# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import torch
import sqlite3
import unicodedata

def set_cfg_params_from_state(state: dict, cfg: DictConfig):
    """
    Overrides some of the encoder config parameters from a give state object
    """
    if not state:
        return

    cfg.do_lower_case = state["do_lower_case"]

    if "encoder" in state:
        saved_encoder_params = state["encoder"]
        # TODO: try to understand why cfg.encoder = state["encoder"] doesn't work
        print(cfg.encoder)
        print(saved_encoder_params)

        for k, v in saved_encoder_params.items():

            # TODO: tmp fix
            if k == "q_wav2vec_model_cfg":
                k = "q_encoder_model_cfg"
            if k == "q_wav2vec_cp_file":
                k = "q_encoder_cp_file"
            if k == "q_wav2vec_cp_file":
                k = "q_encoder_cp_file"

            setattr(cfg.encoder, k, v)
    else:  # 'old' checkpoints backward compatibility support
        pass
        # cfg.encoder.pretrained_model_cfg = state["pretrained_model_cfg"]
        # cfg.encoder.encoder_model_type = state["encoder_model_type"]
        # cfg.encoder.pretrained_file = state["pretrained_file"]
        # cfg.encoder.projection_dim = state["projection_dim"]
        # cfg.encoder.sequence_length = state["sequence_length"]

    ''' TODO : Part of SPAR Implementation
    if not bm25_state:
        return
    
    if "encoder" in bm25_state:
        saved_encoder_params = bm25_state["encoder"]
        # TODO: try to understand why cfg.encoder = state["encoder"] doesn't work

        for k, v in saved_encoder_params.items():

            # TODO: tmp fix
            if k == "q_wav2vec_model_cfg":
                k = "q_encoder_model_cfg"
            if k == "q_wav2vec_cp_file":
                k = "q_encoder_cp_file"
            if k == "q_wav2vec_cp_file":
                k = "q_encoder_cp_file"

            setattr(cfg.bm25_encoder, k, v)
    else:  # 'old' checkpoints backward compatibility support
        pass
        # cfg.encoder.pretrained_model_cfg = state["pretrained_model_cfg"]
        # cfg.encoder.encoder_model_type = state["encoder_model_type"]
        # cfg.encoder.pretrained_file = state["pretrained_file"]
        # cfg.encoder.projection_dim = state["projection_dim"]
        # cfg.encoder.sequence_length = state["sequence_length"]'''

def setup_cfg_gpu(cfg):
    """
    Setup params for CUDA, GPU & distributed training
    """
    logger.info("CFG's local_rank=%s", cfg.local_rank)
    ws = os.environ.get("WORLD_SIZE")
    cfg.distributed_world_size = int(ws) if ws else 1
    logger.info("Env WORLD_SIZE=%s", ws)

    if cfg.distributed_port and cfg.distributed_port > 0:
        logger.info("distributed_port is specified, trying to init distributed mode from SLURM params ...")
        init_method, local_rank, world_size, device = _infer_slurm_init(cfg)

        logger.info(
            "Inferred params from SLURM: init_method=%s | local_rank=%s | world_size=%s",
            init_method,
            local_rank,
            world_size,
        )

        cfg.local_rank = local_rank
        cfg.distributed_world_size = world_size
        cfg.n_gpu = 1

        torch.cuda.set_device(device)
        device = str(torch.device("cuda", device))

        torch.distributed.init_process_group(
            backend="nccl", init_method=init_method, world_size=world_size, rank=local_rank
        )

    elif cfg.local_rank == -1 or cfg.no_cuda:  # single-node multi-gpu (or cpu) mode
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
    else:  # distributed mode
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        import datetime
        torch.distributed.init_process_group(backend="nccl",timeout=datetime.timedelta(seconds=18000))
        cfg.n_gpu = 1

    cfg.device = device

    logger.info(
        "Initialized host %s as d.rank %d on device=%s, n_gpu=%d, world size=%d",
        socket.gethostname(),
        cfg.local_rank,
        cfg.device,
        cfg.n_gpu,
        cfg.distributed_world_size,
    )
    logger.info("16-bits training: %s ", cfg.fp16)
    return cfg

def load_from_dpr(model, path, exact=True):
    
    pass

def load_saved(model, path, exact=True):
    try:
        state_dict = torch.load(path)
    except:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
    
    def filter(x): return x[7:] if x.startswith('module.') else x
    if exact:
        state_dict = {filter(k): v for (k, v) in state_dict.items()}
    else:
        state_dict = {filter(k): v for (k, v) in state_dict.items() if filter(k) in model.state_dict()}
    model.load_state_dict(state_dict)
    return model

def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def convert_to_half(sample):
    if len(sample) == 0:
        return {}

    def _convert_to_half(maybe_floatTensor):
        if torch.is_tensor(maybe_floatTensor) and maybe_floatTensor.type() == "torch.FloatTensor":
            return maybe_floatTensor.half()
        elif isinstance(maybe_floatTensor, dict):
            return {
                key: _convert_to_half(value)
                for key, value in maybe_floatTensor.items()
            }
        elif isinstance(maybe_floatTensor, list):
            return [_convert_to_half(x) for x in maybe_floatTensor]
        else:
            return maybe_floatTensor

    return _convert_to_half(sample)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None):
        self.path = db_path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

def para_has_answer(answer, para, tokenizer):
    assert isinstance(answer, list)
    text = normalize(para)
    tokens = tokenizer.tokenize(text)
    text = tokens.words(uncased=True)
    assert len(text) == len(tokens)
    for single_answer in answer:
        single_answer = normalize(single_answer)
        single_answer = tokenizer.tokenize(single_answer)
        single_answer = single_answer.words(uncased=True)
        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i: i + len(single_answer)]:
                return True
    return False


def complex_ans_recall():
    """
    calculate retrieval metrics for complexwebQ
    """
    import json
    import numpy as np
    from basic_tokenizer import SimpleTokenizer
    tok = SimpleTokenizer()

    predictions = json.load(open("/PATH/code/learning_to_retrieve_reasoning_paths/results/complexwebq_retrieval_res.json"))
    raw_dev = [json.loads(l) for l in open("/PATH/data/ComplexWebQ/complexwebq_dev_qas.txt").readlines()]
    id2qas = {_["id"]:_ for _ in raw_dev}

    assert len(predictions) == len(raw_dev)
    answer_recalls = []
    for item in predictions:
        qid = item["q_id"]
        title2passage = item["context"]
        gold_answers = id2qas[qid]["answer"]

        chain_coverage = []
        for chain in item["topk_titles"]:
            chain_text = " ".join([title2passage[_] for _ in chain])
            chain_coverage.append(para_has_answer(gold_answers, chain_text, tok))
        answer_recalls.append(np.sum(chain_coverage) > 0)
    print(len(answer_recalls))
    print(np.mean(answer_recalls))

if __name__ == "__main__":
    complex_ans_recall()