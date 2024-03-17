import json, sys, random, string
import torch
import faiss
from torch.nn.functional import cosine_similarity as cos_sim
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from itertools import chain, combinations

device = 'cuda'

def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def get_top_vectors(embs, attn_mask, topk=3):
    cls_embedding = embs[:,0,:] # batch_size * 1 * dim
    scores = list()
    for i in range(1, embs.shape[1]):
        if attn_mask[i]:
            token_embedding = embs[:,i,:]
            score = cos_sim(cls_embedding, token_embedding)
            scores.append((i, score))
    scores.sort(key=lambda x : x[1])
    top_indices = [i for i,s in scores[:topk]]
    top_embs = torch.stack([embs[:,i,:].squeeze() for i in top_indices])
    return top_embs, top_indices

def filter_from_questions(
    questions,
    contexts,
    answers,
    cfg
):
    encoder_model = AutoModel.from_pretrained('bert-base-uncased').to(device)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    all_special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens)
    all_punctuation = tokenizer.convert_tokens_to_ids(list(string.punctuation))
    mask_token = tokenizer.convert_tokens_to_ids('[MASK]')

    num_tok_per_passage = sys.maxsize
    entries = list()
    for qid in tqdm(list(answers.keys())):
        que_text = questions[qid]
        tokenized_que = tokenizer(que_text, padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            que_embeddings = encoder_model(
                torch.tensor(tokenized_que.input_ids).unsqueeze(0).to(device),
                attention_mask=torch.tensor(tokenized_que.attention_mask).unsqueeze(0).to(device),
                token_type_ids=torch.tensor(tokenized_que.token_type_ids).unsqueeze(0).to(device),
            ).last_hidden_state.detach().cpu()
        q_ignore_index = [True] + [tok not in all_special_tokens and tok not in all_punctuation for tok in tokenized_que.input_ids[1:]]
        que_embeddings = que_embeddings[:,q_ignore_index,:]
        top_embeddings, top_indices = get_top_vectors(que_embeddings, torch.tensor(tokenized_que.attention_mask))
        top_que_tokens = tokenizer.convert_ids_to_tokens([tokenized_que.input_ids[i] for i in top_indices])

        for pid in answers[qid]:
            if pid in contexts:
                ctx_text = contexts[pid]
                tokenized_ctx = tokenizer(ctx_text, padding=True, truncation=True, max_length=512)
                ctx_input_ids = torch.tensor(tokenized_ctx.input_ids)
                with torch.no_grad():
                    ctx_embeddings = encoder_model(
                        ctx_input_ids.unsqueeze(0).to(device),
                        attention_mask=torch.tensor(tokenized_ctx.attention_mask).unsqueeze(0).to(device),
                        token_type_ids=torch.tensor(tokenized_ctx.token_type_ids).unsqueeze(0).to(device),
                    ).last_hidden_state.squeeze().detach().cpu()
                c_ignore_index = [True] + [tok not in all_special_tokens and tok not in all_punctuation for tok in tokenized_ctx.input_ids[1:]]
                ctx_embeddings = ctx_embeddings[c_ignore_index,:]
                
                index = faiss.IndexFlatL2(ctx_embeddings.shape[-1])
                index.add(ctx_embeddings.numpy())
                _, I = index.search(top_embeddings.numpy(), 3)
                all_mask_indices = set()
                for line in I:
                    all_mask_indices.update([idx for idx in line if ctx_input_ids[idx] not in all_special_tokens and ctx_input_ids[idx] not in all_punctuation])
                toks = [tok for tok in ctx_input_ids[list(all_mask_indices)]]
                toks = list(set(tokenizer.convert_ids_to_tokens(toks)))
                if len(toks) <= 0:
                    continue
                num_tok_per_passage = min(len(toks), num_tok_per_passage)
                set_mask_indices = [[ind for ind in list(all_mask_indices) if random.random() < 0.8] for i in range(3)]
                set_mask_indices = list(all_mask_indices)
                    
                entry = {
                    'question':que_text,
                    'answer':[''],
                    'positive_ctxs':[{
                        'title':'',
                        'text':ctx_text,
                        'id':pid,
                    }],
                    'negative_ctxs':[],
                    'hard_negative_ctxs':[],
                    'question_tokens':top_que_tokens
                }
                for i, mask_id in enumerate(set_mask_indices):
                    masked_ctx_ids = ctx_input_ids.clone()
                    masked_ctx_ids[mask_id] = mask_token
                    masked_ctx = tokenizer.decode(masked_ctx_ids[1:-1], skip_special_tokens=False)
                    entry['hard_negative_ctxs'].append(
                        {
                            'title':'',
                            'text':masked_ctx,
                            'masked_words': toks,
                            'id':pid+'_N',
                        }
                    )
                entries.append(entry)