import os
import glob
import json
import logging
import sys
import pickle
import time
import zlib
import csv
from typing import List, Tuple, Dict, Iterator

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn
import pathlib

from dpr.utils.data_utils import RepTokenSelector
from dpr.data.qa_validation import calculate_matches, calculate_chunked_matches, calculate_matches_from_meta
from dpr.data.retriever_data import KiltCsvCtxSrc, TableChunk
from dpr.models import init_biencoder_components
from dpr.models.biencoder import (
    BiEncoder,
    _select_span_with_token,
)
from dpr.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint


from tqdm import tqdm
logger = logging.getLogger()
setup_logger(logger)
csv.field_size_limit(sys.maxsize)

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

def measure_winrate(score_dicts):
    wins = 0
    for entry in score_dicts:
        wins += int(entry['counterfactual_win'])
    return wins / len(score_dicts) * 100

@hydra.main(config_path="conf", config_name="compare_score")
def main(cfg: DictConfig):
    cfg = setup_cfg_gpu(cfg)
    if cfg.model_file:
        saved_state = load_states_from_checkpoint(cfg.model_file)
        set_cfg_params_from_state(saved_state.encoder_params, cfg)
        
    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)

    logger.info("Loading saved model state ...")
    if cfg.model_file:
        encoder.load_state(saved_state, strict=False)

    question_model = encoder.question_model
    ctx_model = encoder.ctx_model

    question_model, _ = setup_for_distributed_mode(question_model, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16)
    question_model.eval()
    ctx_model, _ = setup_for_distributed_mode(ctx_model, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16)
    ctx_model.eval()

    model_to_load = get_model_obj(question_model)
    vector_size = model_to_load.get_out_size()
    logger.info("Encoder vector_size=%d", vector_size)

    # get questions & answers
    entries = list()
    with open(cfg.input_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            qid, que, pid, psg, title, ans_cf_psg, sent_cf_psg, tr_psg, ans, ans_sent, percentage, tok_diff = line
            entries.append([qid, que, pid, psg, title, ans_cf_psg, sent_cf_psg, tr_psg, ans, ans_sent, percentage, tok_diff])


    score_dicts = list()

    for qid, que, pid, psg, title, ans_cf_psg, sent_cf_psg, tr_psg, ans, ans_sent, percentage, tok_diff in tqdm(entries):

        title = title if cfg.use_title else ''

        que_tensor = tensorizer.text_to_tensor(que)
        q_ids_batch = torch.stack([que_tensor], dim=0).cuda()
        q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
        q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)
        with torch.no_grad():
            _, q_out, hidden, attn = question_model(
                input_ids = q_ids_batch,
                token_type_ids = q_seg_batch, 
                attention_mask = q_attn_mask, 
            )
        q_out = q_out.cpu().detach()

        ctx_tensor = tensorizer.text_to_tensor(psg, title=title) 
        ctx_ids_batch = torch.stack([ctx_tensor], dim=0).cuda()
        ctx_seg_batch = torch.zeros_like(ctx_ids_batch).cuda()
        ctx_attn_mask = tensorizer.get_attn_mask(ctx_ids_batch) 
        with torch.no_grad():
            _, c_out, hidden, attns = ctx_model(
                input_ids=ctx_ids_batch, 
                token_type_ids=ctx_seg_batch, 
                attention_mask=ctx_attn_mask
            )
        c_out = c_out.cpu().detach()

        cf_ctx_tensor = tensorizer.text_to_tensor(sent_cf_psg, title=title) 
        cf_ctx_ids_batch = torch.stack([cf_ctx_tensor], dim=0).cuda()
        cf_ctx_seg_batch = torch.zeros_like(cf_ctx_ids_batch).cuda()
        cf_ctx_attn_mask = tensorizer.get_attn_mask(cf_ctx_ids_batch) 
        with torch.no_grad():
            _, cfc_out, hidden, attns = ctx_model(
                input_ids=cf_ctx_ids_batch, 
                token_type_ids=cf_ctx_seg_batch, 
                attention_mask=cf_ctx_attn_mask
            )
        cfc_out = cfc_out.cpu().detach()

        original_score = float(dot_product_scores(q_out, c_out))
        counterfactual_score = float(dot_product_scores(q_out, cfc_out))
        
        s_dict={
            'qid':qid,
            'question':que,
            'title':title,
            'passage':psg,
            'passage_score':original_score,
            'counterfactual': sent_cf_psg,
            'counterfactual_score':counterfactual_score,
            'counterfactual_win':counterfactual_score>original_score,
        }
        
        if cfg.output_remove_percentage:
            ans_cf_ctx_tensor = tensorizer.text_to_tensor(ans_cf_psg, title=title) 
            ans_cf_ctx_ids_batch = torch.stack([ans_cf_ctx_tensor], dim=0).cuda()
            ans_cf_ctx_seg_batch = torch.zeros_like(ans_cf_ctx_ids_batch).cuda()
            ans_cf_ctx_attn_mask = tensorizer.get_attn_mask(ans_cf_ctx_ids_batch) 
            with torch.no_grad():
                _, ans_cf_ctx_c_out, hidden, attns = ctx_model(
                    input_ids=ans_cf_ctx_ids_batch, 
                    token_type_ids=ans_cf_ctx_seg_batch, 
                    attention_mask=ans_cf_ctx_attn_mask
                )   
            ans_cf_ctx_c_out = ans_cf_ctx_c_out.cpu().detach()
            ans_cf_ctx_score = float(dot_product_scores(q_out, ans_cf_ctx_c_out))

            tr_ctx_tensor = tensorizer.text_to_tensor(tr_psg, title=title) 
            tr_ctx_ids_batch = torch.stack([tr_ctx_tensor], dim=0).cuda()
            tr_ctx_seg_batch = torch.zeros_like(tr_ctx_ids_batch).cuda()
            tr_ctx_attn_mask = tensorizer.get_attn_mask(tr_ctx_ids_batch) 
            with torch.no_grad():
                _, tr_c_out, hidden, attns = ctx_model(
                    input_ids=tr_ctx_ids_batch, 
                    token_type_ids=tr_ctx_seg_batch, 
                    attention_mask=tr_ctx_attn_mask
                )   
            tr_c_out = tr_c_out.cpu().detach()
            trunc_score = float(dot_product_scores(q_out, tr_c_out))

            answer_sent_tensor = tensorizer.text_to_tensor(ans_sent, title=title) 
            answer_sent_ids_batch = torch.stack([answer_sent_tensor], dim=0).cuda()
            answer_sent_seg_batch = torch.zeros_like(answer_sent_ids_batch).cuda()
            answer_sent_attn_mask = tensorizer.get_attn_mask(answer_sent_seg_batch) 
            with torch.no_grad():
                _, ans_sent_out, hidden, attns = ctx_model(
                    input_ids=answer_sent_ids_batch, 
                    token_type_ids=answer_sent_seg_batch, 
                    attention_mask=answer_sent_attn_mask
                )
            ans_sent_out = ans_sent_out.cpu().detach()
            answer_sentence_score = float(dot_product_scores(q_out, ans_sent_out))
            
            s_dict['answer_mask_counterfactual'] = ans_cf_psg
            s_dict['answer_mask_counterfactual_score'] = ans_cf_ctx_score
            s_dict['truncated_psg'] = tr_psg
            s_dict['truncated_psg_score'] = trunc_score
            s_dict['short_answer'] = ans
            s_dict['answer_sentence'] = ans_sent
            s_dict['answer_sentence_score'] = answer_sentence_score
            s_dict['answer_percentage'] = percentage
            s_dict['token_difference'] = tok_diff

        score_dicts.append(s_dict)

    winrate = measure_winrate(score_dicts)
    score_dicts.append({'winrate' : measure_winrate(score_dicts)})
    logger.info(f'AAR : {100-winrate}')

    with open(f'counterfactual_comparison_{cfg.num_ckpt}.json','w') as f:
        json.dump(score_dicts, f, indent=4)
    
if __name__ == "__main__":
    main()
