#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import collections
import logging
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from dpr.data.biencoder_data import BiEncoderSample
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import CheckpointState

logger = logging.getLogger(__name__)

BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "question_ids",
        "question_segments",
        "context_ids",
        "ctx_segments",
        "is_positive",
        "hard_negatives",
        "cf_negative",
        "encoder_type",
    ],
)

# TODO: it is only used by _select_span_with_token. Move them to utils
rnd = random.Random(0)

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class AdMSoftmaxLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        
        #AM Softmax Loss
        
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        
        #input shape (N, in_features)
        
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
'''

def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def cosine_scores(q_vector: T, ctx_vectors: T) -> T:
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)


class BiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
    ):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False,
        representation_token_pos=0,
        output_hidden_states=False,
        output_attentions=False
    ) -> (T, T, T):
        sequence_output = None
        pooled_output = None
        hidden_states = None
        attns = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states, attns = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                        output_hidden_states=output_hidden_states,
                        output_attentions=output_attentions
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states, attns = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                    output_hidden_states=output_hidden_states,
                    output_attentions=output_attentions
                )

        return sequence_output, pooled_output, hidden_states, attns

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos=0,
        output_hidden_states=False,
        output_attentions = False,
    ) -> Tuple[T, T]:
        q_encoder = self.question_model if encoder_type is None or encoder_type == "question" else self.ctx_model
        _q_seq, q_pooled_out, _q_hidden, _q_attns = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
            output_hidden_states=output_hidden_states,
            output_attentions = output_attentions
        )

        ctx_encoder = self.ctx_model if encoder_type is None or encoder_type == "ctx" else self.question_model
        _ctx_seq, ctx_pooled_out, _ctx_hidden, _ctx_attns = self.get_representation(
            ctx_encoder, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder, 
            output_hidden_states=output_hidden_states, output_attentions = output_attentions
        )

        return q_pooled_out, ctx_pooled_out, _q_attns, _ctx_attns
        #return q_pooled_out, ctx_pooled_out, _q_attns, _ctx_attns, _q_hidden, _ctx_hidden

    def create_biencoder_input(
        self,
        samples: List[BiEncoderSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = True,
        hard_neg_fallback: bool = True,
        query_token: str = None,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only
            
            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            question = sample.query
            # question = normalize_question(sample.query)

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            #all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            all_ctxs = [positive_ctx] + hard_neg_ctxs + neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(ctx.text, title=ctx.title if (insert_title and ctx.title) else None)
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    query_span = _select_span_with_token(question, tensorizer, token_str=query_token)
                    question_tensors.append(query_span)
                else:
                    question_tensors.append(tensorizer.text_to_tensor(" ".join([query_token, question])))
            else:
                question_tensors.append(tensorizer.text_to_tensor(question))

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            None,
            "question",
        )

    def create_biencoder_input_for_counterfactual(
        self,
        samples: List[BiEncoderSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        #cf_pos_indices = []
        cf_neg_indices = []

        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only
            
            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_choice = np.random.choice(len(positive_ctxs))
                positive_ctx = positive_ctxs[positive_choice]
                if len(sample.cf_negative_passages) <= 1:
                    cf_neg_ctx = sample.cf_negative_passages[0]
                else:
                    cf_neg_ctx = sample.cf_negative_passages[positive_choice]
            else:
                positive_ctx = sample.positive_passages[0]
                #cf_neg_ctx = sample.cf_negative_passages[0]

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            cf_negative_ctxs = sample.cf_negative_passages

            question = sample.query
            # question = normalize_question(sample.query)

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)
                #random.shuffle(cf_negative_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]
            cf_neg_ctx = cf_negative_ctxs[0]

            #all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            all_ctxs = [positive_ctx] + hard_neg_ctxs + [cf_neg_ctx] + neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)
            cf_neg_idx = hard_negatives_end_idx

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(ctx.text, title=ctx.title if (insert_title and ctx.title) else None)
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )
            #cf_pos_indices.append(current_ctxs_len+cf_pos_idx)
            cf_neg_indices.append(current_ctxs_len+cf_neg_idx)

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    query_span = _select_span_with_token(question, tensorizer, token_str=query_token)
                    question_tensors.append(query_span)
                else:
                    question_tensors.append(tensorizer.text_to_tensor(" ".join([query_token, question])))
            else:
                question_tensors.append(tensorizer.text_to_tensor(question))

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            cf_neg_indices,
            "question",
        )

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        # TODO: make a long term HF compatibility fix
        # if "question_model.embeddings.position_ids" in saved_state.model_dict:
        #    del saved_state.model_dict["question_model.embeddings.position_ids"]
        #    del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return self.state_dict()

class BiEncoderNllLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)
        #logger.info(scores[0,:])

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)
        
        #pos_neg_scores = [(scores[i, pos], scores[i, neg[0]])  \
        #    for i, (pos, neg) in enumerate(zip(positive_idx_per_question,hard_negative_idx_per_question))]
        #print(pos_neg_scores)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores

class BiEncoderNllLossWithCounterfactuals(BiEncoderNllLoss):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        counterfact_neg_idx_per_question: list = None,
        l1_alpha : float = 0.0 ,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        alpha = [1, 1, 1, 1]

        scores = self.get_scores(q_vectors, ctx_vectors)
        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        hard_negative_indices = [i for li in hard_negative_idx_per_question for i in li]

        # Task scores : positive + negative + hard negative
        #task_scores = BiEncoderNllLossWithCounterfactuals.get_task_mapping(counterfact_neg_idx_per_question, scores.shape, alpha=l1_alpha)
        task_scores = BiEncoderNllLossWithCounterfactuals.get_cf_task_mapping(counterfact_neg_idx_per_question, hard_negative_indices,  scores.shape, alpha=l1_alpha)
        #logger.info('Task scores')
        #logger.info(task_scores)
        task_scores = scores * task_scores.to(device=scores.device)

        # cpp
        masked_scores = BiEncoderNllLossWithCounterfactuals.get_masked_mapping(positive_idx_per_question, hard_negative_indices, scores.shape)
        #logger.info('Masked scores')
        #logger.info(masked_scores)
        masked_scores = scores * masked_scores.to(device=scores.device)
        
        # chn
        counterfactual_scores = BiEncoderNllLossWithCounterfactuals.get_counterfactual_mapping(positive_idx_per_question, counterfact_neg_idx_per_question, scores.shape)
        #logger.info('Counterfactual_scores')
        #logger.info(counterfactual_scores)
        counterfactual_scores = scores * counterfactual_scores.to(device=scores.device)

        # hn
        hardneg_scores = BiEncoderNllLossWithCounterfactuals.get_hardneg_mapping(counterfact_neg_idx_per_question, scores.shape)
        #logger.info('Hardneg scores')
        #logger.info(hardneg_scores)
        hardneg_scores = scores * hardneg_scores.to(device=scores.device)

        # Original Task Loss
        softmax_task_scores = F.log_softmax(task_scores, dim=1)
        task_loss = F.nll_loss(
            softmax_task_scores,
            torch.tensor(positive_idx_per_question).to(softmax_task_scores.device),
            reduction="mean",
        )

        ## Masked Passage Loss
        softmax_masked_scores = F.log_softmax(masked_scores, dim=1)
        masked_loss = F.nll_loss(
            softmax_masked_scores,
            torch.tensor(counterfact_neg_idx_per_question).to(softmax_masked_scores.device),
            reduction='mean',
        )

        ### Counterfactual Contrastive Loss
        
        softmax_counterfactual_scores = F.log_softmax(counterfactual_scores, dim=1)
        counterfactual_loss = F.nll_loss(
            softmax_counterfactual_scores,
            torch.tensor(positive_idx_per_question).to(softmax_counterfactual_scores.device),
            reduction='mean',
        )
        
        # Hard Negative Loss
        softmax_hardneg_scores = F.log_softmax(hardneg_scores, dim=1)
        hardneg_loss = F.nll_loss(
            softmax_hardneg_scores,
            torch.tensor(positive_idx_per_question).to(softmax_hardneg_scores.device),
            reduction='mean',
        )

        loss = task_loss + alpha[1] * masked_loss +  alpha[2] * counterfactual_loss + alpha[3] * hardneg_loss

        max_score, max_idxs = torch.max(softmax_task_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_task_mapping(cf_negative_index : list, shape : torch.Size, alpha : float = 0.0):
        n_questions = shape[0]
        n_scores = shape[1]
        # + All counterfactuals
        #task_mapping = [[int(i != cf_negative_index[j]) for i in range(n_scores)] for j in range(n_questions)]
        #task_mapping = [[alpha if i == cf_negative_index[j] else 1 for i in range(n_scores)] for j in range(n_questions)]
        
        # + Counterfactual of positive passage
        task_mapping = [[0 if i in cf_negative_index else 1 for i in range(n_scores)] for j in range(n_questions)]
        task_mapping = [[alpha if i == cf_negative_index[j] else line[i] for i in range(n_scores)] for j, line in enumerate(task_mapping)]
        
        # No additional counterfactuals
        #task_mapping = [[0 if i in cf_negative_index else 1 for i in range(n_scores)] for j in range(n_questions)]
        return torch.tensor(task_mapping)

    @staticmethod
    def get_cf_task_mapping(cf_negative_index : list, hard_negative_indices : list, shape : torch.Size, alpha : float = 0.0):
        n_questions = shape[0]
        n_scores = shape[1]
        
        # + Counterfactual of positive passage
        task_mapping = [[0 if i in cf_negative_index or i in hard_negative_indices else 1 for i in range(n_scores)] for j in range(n_questions)]
        task_mapping = [[alpha if i == cf_negative_index[j] else line[i] for i in range(n_scores)] for j, line in enumerate(task_mapping)]
        
        return torch.tensor(task_mapping)

    '''
    @staticmethod
    def get_masked_mapping(positive_index : list, shape : torch.Size):
        n_questions = shape[0]
        n_scores = shape[1]
        masked_mapping = [[int(i != positive_index[j]) for i in range(n_scores)] for j in range(n_questions)]
        return torch.tensor(masked_mapping)
    '''

    @staticmethod
    def get_masked_mapping(positive_index : list, hard_negative_indices : list, shape : torch.Size):
        n_questions = shape[0]
        n_scores = shape[1]
        masked_mapping = [[int(i != positive_index[j] and i not in hard_negative_indices) for i in range(n_scores)] for j in range(n_questions)]
        return torch.tensor(masked_mapping)
    
    @staticmethod
    def get_counterfactual_mapping(positive_index : list, cf_negative_index : list, shape : torch.Size):
        n_questions = shape[0]
        n_scores = shape[1]
        counterfactual_mapping = [[int(i == cf_negative_index[j] or i == positive_index[j]) for i in range(n_scores)] for j in range(n_questions)]
        return torch.tensor(counterfactual_mapping)

    @staticmethod
    def get_hardneg_mapping(cf_negative_index : list, shape : torch.Size):
        n_questions = shape[0]
        n_scores = shape[1]
        hardneg_mapping = [[0 if i in cf_negative_index else 1 for i in range(n_scores)] for j in range(n_questions)]
        
        return torch.tensor(hardneg_mapping)

    @staticmethod
    def get_vectors(scores, positive_index : list, cf_negative_index : list, shape : torch.Size) -> T:
        n_questions = shape[0]
        n_scores = shape[1]
        positive_vector = torch.tensor([[scores[i,j] for j in range(n_scores) if j!=cf_negative_index[i]] for i in range(n_questions)])
        cf_vector = torch.tensor([[scores[i,j] for j in range(n_scores) if j!=positive_index[i]] for i in range(n_questions)])
        return positive_vector, cf_vector

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_passage_similarity(ctx_vectors, positive_idx, hard_negative_idx):
        positive_idx = [idx for idx in positive_idx]
        positive_vectors = ctx_vectors[positive_idx]
        hard_negative_idx = [idx for idx in hard_negative_idx]
        negative_vectors = ctx_vectors[hard_negative_idx]
        f = BiEncoderNllLoss.get_similarity_function()
        return f(positive_vectors, negative_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores

class BiEncoderNllLossForAblation(BiEncoderNllLossWithCounterfactuals):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        counterfact_neg_idx_per_question: list = None,
        l1_alpha : float = 0.0 ,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        alpha = [1, 1, 1]

        scores = self.get_scores(q_vectors, ctx_vectors)
        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)
        logger.info(scores)

        # Task scores : positive + negative + hard negative
        task_scores = BiEncoderNllLossWithCounterfactuals.get_task_mapping(counterfact_neg_idx_per_question, scores.shape, alpha=l1_alpha)
        task_scores = scores * task_scores.to(device=scores.device)
        
        # positive score + hard_negative score
        masked_scores = BiEncoderNllLossWithCounterfactuals.get_masked_mapping(positive_idx_per_question, scores.shape)
        masked_scores = scores * masked_scores.to(device=scores.device)
        
        # negative score + hard negative score
        counterfactual_scores = BiEncoderNllLossWithCounterfactuals.get_counterfactual_mapping(positive_idx_per_question, counterfact_neg_idx_per_question, scores.shape)
        counterfactual_scores = scores * counterfactual_scores.to(device=scores.device)
        
        # Original Task Loss
        softmax_task_scores = F.log_softmax(task_scores, dim=1)
        task_loss = F.nll_loss(
            softmax_task_scores,
            torch.tensor(positive_idx_per_question).to(softmax_task_scores.device),
            reduction="mean",
        )

        ## Masked Passage Loss
        softmax_masked_scores = F.log_softmax(masked_scores, dim=1)
        masked_loss = F.nll_loss(
            softmax_masked_scores,
            torch.tensor(counterfact_neg_idx_per_question).to(softmax_masked_scores.device),
            reduction='mean',
        )
        #logger.info(f'Mask Loss : {float(masked_loss)}')

        ### Counterfactual Contrastive Loss
        softmax_counterfactual_scores = F.log_softmax(counterfactual_scores, dim=1)
        counterfactual_loss = F.nll_loss(
            softmax_counterfactual_scores,
            torch.tensor(positive_idx_per_question).to(softmax_counterfactual_scores.device),
            reduction='mean',
        )
        #logger.info(f'Counterfactual Loss : {float(counterfactual_loss)}')
        
        #logger.info('DPR inbatch + cf as hard negative')
        #loss = task_loss #+ alpha[1] * masked_loss +  alpha[2] * counterfactual_loss
        #logger.info('L1+L2')
        #loss = task_loss + alpha[1] * masked_loss
        logger.info('L1+L3')
        loss = task_loss +  alpha[2] * counterfactual_loss
        #logger.info('L2+L3')    
        #loss = alpha[1]*masked_loss + alpha[2]*counterfactual_loss

        max_score, max_idxs = torch.max(softmax_task_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count


def _select_span_with_token(text: str, tensorizer: Tensorizer, token_str: str = "[START_ENT]") -> T:
    id = tensorizer.get_token_id(token_str)
    query_tensor = tensorizer.text_to_tensor(text)

    if id not in query_tensor:
        query_tensor_full = tensorizer.text_to_tensor(text, apply_max_len=False)
        token_indexes = (query_tensor_full == id).nonzero()
        if token_indexes.size(0) > 0:
            start_pos = token_indexes[0, 0].item()
            # add some randomization to avoid overfitting to a specific token position

            left_shit = int(tensorizer.max_length / 2)
            rnd_shift = int((rnd.random() - 0.5) * left_shit / 2)
            left_shit += rnd_shift

            query_tensor = query_tensor_full[start_pos - left_shit :]
            cls_id = tensorizer.tokenizer.cls_token_id
            if query_tensor[0] != cls_id:
                query_tensor = torch.cat([torch.tensor([cls_id]), query_tensor], dim=0)

            from dpr.models.reader import _pad_to_len

            query_tensor = _pad_to_len(query_tensor, tensorizer.get_pad_id(), tensorizer.max_length)
            query_tensor[-1] = tensorizer.tokenizer.sep_token_id
            # logger.info('aligned query_tensor %s', query_tensor)

            assert id in query_tensor, "query_tensor={}".format(query_tensor)
            return query_tensor
        else:
            raise RuntimeError("[START_ENT] toke not found for Entity Linking sample query={}".format(text))
    else:
        return query_tensor
