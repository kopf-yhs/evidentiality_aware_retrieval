#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import logging
import math
import os
import pathlib
import pickle
import time
from typing import List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from dpr.data.biencoder_data import BiEncoderPassage
from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger

from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)

logger = logging.getLogger()
setup_logger(logger)

def gen_ctx_vectors(
    cfg: DictConfig,
    ctx_rows: List[Tuple[object, BiEncoderPassage]],
    model: nn.Module,
    tensorizer: Tensorizer,
    insert_title: bool = True,
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = cfg.batch_size
    total = 0
    results = []
    attentions = []
    attention_dump = None
    if cfg.output_attentions:
        attention_dump = open('attentions.pkl','wb')

    for j, batch_start in enumerate(range(0, n, bsz)):
        batch = ctx_rows[batch_start : batch_start + bsz]
        batch_token_tensors = [
            tensorizer.text_to_tensor(ctx[1].text, title=ctx[1].title if insert_title else None) for ctx in batch
        ]

        ctx_ids = [r[0] for r in batch]
        ctxs = [r[1].text for r in batch]
        #print(ctx_ids)
        #print(ctxs)

        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0), cfg.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), cfg.device)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch), cfg.device)
        with torch.no_grad():
            _, out, hidden, attns = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask, output_attentions=cfg.output_attentions)
        out = out.cpu()


        extra_info = []
        if len(batch[0]) > 3:
            extra_info = [r[3:] for r in batch]

        assert len(ctx_ids) == out.size(0)
        total += len(ctx_ids)
            
        # TODO: refactor to avoid 'if'
        if cfg.output_attentions:
            ctx_attn_mask = ctx_attn_mask.cpu()
            attns = torch.stack(attns, dim=0).cpu()
            attns = attns.transpose(0,1)
            
            #attentions.extend([(ctx_ids[i],ctxs[i],batch_token_tensors[i],attns[i]) for i in range(out.size(0))])
            pickle.dump(
                [
                    ctx_ids,
                    ctxs,
                    batch_token_tensors,
                    attns,
                ]
                , attention_dump
            )

        if extra_info:
            results.extend([(ctx_ids[i], out[i].view(-1).numpy(), *extra_info[i]) for i in range(out.size(0))])
        #elif cfg.output_attentions:
        #    results.extend([(ctx_ids[i], out[i].view(-1).numpy(), attns[i]) for i in range(out.size(0))])
        else:
            results.extend([(ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))])

        if total % 10 == 0:
            logger.info("Encoded passages %d", total)
    
    if cfg.output_attentions:
        attention_dump.close()
        
    return results


@hydra.main(config_path="conf", config_name="gen_embs")
def main(cfg: DictConfig):

    assert cfg.ctx_src, "Please specify passages source as ctx_src param"

    cfg = setup_cfg_gpu(cfg)
    if cfg.model_file:
        saved_state = load_states_from_checkpoint(cfg.model_file)
        set_cfg_params_from_state(saved_state.encoder_params, cfg)
    
    logger.info("CFG:")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)

    encoder = encoder.ctx_model if cfg.encoder_type == "ctx" else encoder.question_model

    encoder, _ = setup_for_distributed_mode(
        encoder,
        None,
        cfg.device,
        cfg.n_gpu,
        cfg.local_rank,
        cfg.fp16,
        cfg.fp16_opt_level,
    )
    encoder.eval()

    # load weights from the model file
    if cfg.model_file:
        model_to_load = get_model_obj(encoder)
        logger.info("Loading saved model state ...")
        logger.debug("saved model keys =%s", saved_state.model_dict.keys())
    
        prefix_len = len("ctx_model.")
        ctx_state = {
            key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith("ctx_model.")
        }
        model_to_load.load_state_dict(ctx_state, strict=False)

    logger.info("reading data source: %s", cfg.ctx_src)

    ctx_src = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src])
    all_passages_dict = {}
    ctx_src.load_data_to(all_passages_dict)
    all_passages = [(k, v) for k, v in all_passages_dict.items()]

    shard_size = math.ceil(len(all_passages) / cfg.num_shards)
    start_idx = cfg.shard_id * shard_size
    end_idx = start_idx + shard_size

    logger.info(
        "Producing encodings for passages range: %d to %d (out of total %d)",
        start_idx,
        end_idx,
        len(all_passages),
    )
    shard_passages = all_passages[start_idx:end_idx]

    data = gen_ctx_vectors(cfg, shard_passages, encoder, tensorizer, True)

    file = cfg.out_file + "_" + str(cfg.shard_id)
    logger.info('New directory %s on %s' % (os.path.dirname(cfg.out_file), os.getcwd()))
    pathlib.Path(os.path.dirname(cfg.out_file)).mkdir(parents=True, exist_ok=True)
    logger.info("Writing results to %s" % file)
    with open(file, mode="wb") as f:
        pickle.dump(data, f)

    logger.info("Total passages processed %d. Written to %s", len(data), file)


if __name__ == "__main__":
    main()
