
'''
Credits to luyug for Condenser implementation
https://github.com/luyug/Condenser
'''

import os
import warnings

import torch
from torch import nn
from torch import Tensor as T
import torch.distributed as dist
import torch.nn.functional as F
from transformers import BertModel, BertConfig, AutoModel, AutoModelForMaskedLM, AutoConfig, PretrainedConfig, \
    RobertaModel
from transformers.models.bert.modeling_bert import BertPooler, BertOnlyMLMHead, BertPreTrainingHeads, BertLayer
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling, MaskedLMOutput
from transformers.models.roberta.modeling_roberta import RobertaLayer

from transformers import TrainingArguments
import logging

from dataclasses import dataclass, field
from .biencoder import BiEncoder
from .hf_models import get_bert_tensorizer, get_optimizer
import os
from transformers import TrainingArguments

logger = logging.getLogger(__name__)

def get_condenser_biencoder_componenets(cfg, inference_only=False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    assert hasattr(cfg.encoder, 'n_head_layers')
    question_encoder = CondenserEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        n_heads=cfg.encoder.n_head_layers,
        project_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )
    ctx_encoder = CondenserEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        n_heads=cfg.encoder.n_head_layers,
        project_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )
    
    fix_ctx_encoder = cfg.encoder.fix_ctx_encoder if hasattr(cfg.encoder, "fix_ctx_encoder") else False
    biencoder = BiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder)

    optimizer = (
        get_optimizer(
            biencoder, 
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train_adam_epis,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, biencoder, optimizer


class CondenserEncoder(BertModel):
    def __init__(self, config, n_heads, project_dim: int = 0):
        BertModel.__init__(self,config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.n_heads = n_heads
        self.c_head = nn.ModuleList(
            [BertLayer(self.config) for _ in range(n_heads)]
        )
        self.encoder_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()
    
    @classmethod
    def init_encoder(
        cls, cfg_name: str, n_heads, project_dim: int = 0, dropout: float = 0.1, pretrained: bool = True, **kwargs
    ):
        cfg = AutoConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        
        if pretrained:
            return cls.from_pretrained(cfg_name, config=cfg, n_heads=n_heads, project_dim=project_dim, **kwargs)
        else:
            return CondenserEncoder(cfg, n_heads=n_heads, project_dim=project_dim)

    def forward(
        self, 
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0,
        output_hidden_states=False,
        output_attentions=False,
    ):

        lm_out = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=output_attentions,
        )

        _, _, hidden_states = lm_out

        cls_hiddens = hidden_states[-1][:, :1]
        skip_hiddens = hidden_states[self.model_args.skip_from]

        hiddens = torch.cat([cls_hiddens, skip_hiddens[:, 1:]], dim=1)

        for layer in self.c_head:
            layer_out = layer(
                hiddens,
                attention_mask,
            )
            hiddens = layer_out[0]
        hidden_states = hiddens
        return None, hidden_states, None, None

    #@classmethod
    #def from_pretrained(
    #        cls, model_args: ModelArguments, data_args: DataTrainingArguments, train_args: TrainingArguments,
    #        *args, **kwargs
    #):
    #    hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
    #    model = cls(hf_model, model_args, data_args, train_args)
    #    path = args[0]
    #    if os.path.exists(os.path.join(path, 'model.pt')):
    #        logger.info('loading extra weights from local files')
    #        model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
    #        load_result = model.load_state_dict(model_dict, strict=False)
    #    return model
'''
class CoCondenser(CondenserEncoder):
    def __init__(
            self,
            bert: BertModel,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            train_args: CoCondenserPreTrainingArguments
    ):
        super(CoCondenser, self).__init__(bert, model_args, data_args, train_args)

        effective_bsz = train_args.per_device_train_batch_size * self._world_size() * 2
        target = torch.arange(effective_bsz, dtype=torch.long).view(-1, 2).flip([1]).flatten().contiguous()

        self.register_buffer(
            'co_target', target
        )

    def _gather_tensor(self, t: T):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        return all_tensors

    def gather_tensors(self, *tt: T):
        tt = [torch.cat(self._gather_tensor(t)) for t in tt]
        return tt

    def forward(self, model_input, labels, grad_cache: T = None, chunk_offset: int = None):
        attention_mask = self.lm.get_extended_attention_mask(
            model_input['attention_mask'],
            model_input['attention_mask'].shape,
            model_input['attention_mask'].device
        )

        lm_out: MaskedLMOutput = self.lm(
            **model_input,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )

        cls_hiddens = lm_out.hidden_states[-1][:, :1]
        if self.train_args.local_rank > -1 and grad_cache is None:
            co_cls_hiddens = self.gather_tensors(cls_hiddens.squeeze().contiguous())[0]
        else:
            co_cls_hiddens = cls_hiddens.squeeze()

        skip_hiddens = lm_out.hidden_states[self.model_args.skip_from]
        hiddens = torch.cat([cls_hiddens, skip_hiddens[:, 1:]], dim=1)

        for layer in self.c_head:
            layer_out = layer(
                hiddens,
                attention_mask,
            )
            hiddens = layer_out[0]

        loss = self.mlm_loss(hiddens, labels)
        if self.model_args.late_mlm:
            loss += lm_out.loss

        if grad_cache is None:
            co_loss = self.compute_contrastive_loss(co_cls_hiddens)
            return loss + co_loss
        else:
            loss = loss * (float(hiddens.size(0)) / self.train_args.per_device_train_batch_size)
            cached_grads = grad_cache[chunk_offset: chunk_offset + co_cls_hiddens.size(0)]
            surrogate = torch.dot(cached_grads.flatten(), co_cls_hiddens.flatten())
            return loss, surrogate

    def contrastive_loss(self, co_cls_hiddens):
        similarities = torch.matmul(co_cls_hiddens, co_cls_hiddens.transpose(0, 1))
        similarities.fill_diagonal_(float('-inf'))
        co_loss = F.cross_entropy(similarities, self.co_target) * self._world_size()
        return co_loss
'''
