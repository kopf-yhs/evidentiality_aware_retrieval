# pip install accelerate
import torch
import argparse
import json, csv
import nltk
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn import CrossEntropyLoss
import pickle
import gc
from math import exp
from bs4 import BeautifulSoup

import spacy
nlp = spacy.load("en_core_web_sm")
#nlp.remove_pipe('sentencizer')
config = {
    'punct_chars':[
        '!', '.', '?', '։', '؟', '۔', '܀', '܁', '܂', '߹',
        '।', '॥', '၊', '။', '።', '፧', '፨', '᙮', '᜵', '᜶', '᠃', '᠉', '᥄',
        '᥅', '᪨', '᪩', '᪪', '᪫', '᭚', '᭛', '᭞', '᭟', '᰻', '᰼', '᱾', '᱿',
            '‼', '‽', '⁇', '⁈', '⁉', '⸮', '⸼', '꓿', '꘎', '꘏', '꛳', '꛷', '꡶',
            '꡷', '꣎', '꣏', '꤯', '꧈', '꧉', '꩝', '꩞', '꩟', '꫰', '꫱', '꯫', '﹒',
            '﹖', '﹗', '！', '．', '？', '𐩖', '𐩗', '𑁇', '𑁈', '𑂾', '𑂿', '𑃀',
            '𑃁', '𑅁', '𑅂', '𑅃', '𑇅', '𑇆', '𑇍', '𑇞', '𑇟', '𑈸', '𑈹', '𑈻', '𑈼',
            '𑊩', '𑑋', '𑑌', '𑗂', '𑗃', '𑗉', '𑗊', '𑗋', '𑗌', '𑗍', '𑗎', '𑗏', '𑗐',
            '𑗑', '𑗒', '𑗓', '𑗔', '𑗕', '𑗖', '𑗗', '𑙁', '𑙂', '𑜼', '𑜽', '𑜾', '𑩂',
            '𑩃', '𑪛', '𑪜', '𑱁', '𑱂', '𖩮', '𖩯', '𖫵', '𖬷', '𖬸', '𖭄', '𛲟', '𝪈',
            '｡', '。',','
    ],
    'overwrite':True,
}
nlp.add_pipe('sentencizer',config=config)

CACHE_DIR = '.model_cache/'
torch.hub.set_dir('.model_cache/')

PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
FORCE_RESET = True

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--model_file',
        type=str,
        default='google/flan-t5-xl'
    )
    parser.add_argument(
        '--shard_idx',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--num_shard',
        type=int,
        default=1,
    )
    return parser.parse_args()


def web_parse(text):
    soup = BeautifulSoup(text)
    text = soup.get_text()
    return text

def get_perplexity(
    model,
    tokenizer,
    loss_fct,
    input_texts: str or list, 
    output_texts: str or list, 
    batch: int = None
):
        """ Compute the perplexity on decoder of the seq2seq model.

        :param input_texts: A string or list of input texts for the encoder.
        :param output_texts: A string or list of output texts for the decoder.
        :param batch: Batch size
        :return: A value or list of perplexity.
        """
        assert type(input_texts) is type(output_texts), f"{type(input_texts)} != {type(output_texts)}"

        vocab_size = 32127

        # batch preparation
        single_input = type(input_texts) == str
        input_texts = [input_texts] if single_input else input_texts
        output_texts = [output_texts] if single_input else output_texts
        
        assert len(input_texts) == len(output_texts), f"{len(input_texts)} == {len(output_texts)}"
        batch = len(output_texts) if batch is None else batch
        batch_id = list(range(0, len(input_texts), batch)) + [len(output_texts)]
        batch_id = list(zip(batch_id[:-1], batch_id[1:]))

        loss_list = []
        with torch.no_grad():
            for s, e in tqdm(batch_id):
                # input feature
                model_inputs = tokenizer(input_texts[s:e], return_tensors='pt', padding=True, truncation=True)
                output_encode = tokenizer(text_target=output_texts[s:e], return_tensors='pt', padding=True, truncation=True)

                # shift the label sequence for causal inference
                label = output_encode["input_ids"]
                label[label == tokenizer.pad_token_id] = PAD_TOKEN_LABEL_ID

                model_inputs["labels"] = label
                with torch.no_grad():
                    output = model(**{k: v.cuda() for k, v in model_inputs.items()})
                model_inputs["labels"] = label.to(model.device)

                # model run & loss conversion into likelihood
                logits = output['logits']
                logits = logits[:, :, :-1]
                valid_length = (model_inputs["labels"] != PAD_TOKEN_LABEL_ID).sum(dim=-1)
                loss = loss_fct(logits.view(-1, vocab_size), model_inputs["labels"].view(-1))
                loss = loss.view(len(logits), -1)
                loss = torch.sum(loss, -1) / valid_length
                loss_list += loss.cpu().tolist()

                if FORCE_RESET:
                    del model_inputs
                    del loss
                    del output
                    gc.collect()
                    torch.cuda.empty_cache()

        # conversion to perplexity
        ppl = [exp(i) for i in loss_list]
        return ppl[0] if single_input else ppl

def create_perturb_ctxs(idx, que, ctx, ans, prefix):
    sents = nlp(ctx).sents
    targets=[]
    target=[]
    for sent in sents:
        if len(sent)<=3: # 최소 3단어부터 문장으로 취급함 (휴리스틱ㅋ)
            target.extend([t.text for t in sent])
            if len(target)>6:
                targets.append(' '.join([token for token in target]))
                # print(target, len(target))
                target=[]
                continue
        else:
            targets.append(' '.join([token.text for token in sent]))
            target=[]
    #sents = nltk.sent_tokenize(ctx)
    new_sents = targets

    perturb_ctxs = {i:' '.join(new_sents[:i] + new_sents[i+1:]) for i in range(len(new_sents))}
    perturb_inputs = list()
    for i, ct in perturb_ctxs.items():
        input_text = prefix + '[QUESTION] ' + que + ' [CONTEXT] ' + ct
        #input_ids = tokenizer(input_text, return_tensors="pt")
        #labels = tokenizer(ans, return_tensors="pt", padding=False)
        perturb_inputs.append([idx, que, input_text, ans, ctx, i])
    return perturb_inputs

def main():
    args = get_arguments()
    
    with open(args.data_path, 'r') as f:
        data = json.load(f)

    tokenizer = T5Tokenizer.from_pretrained(args.model_file, cache_dir=CACHE_DIR)
    tokenizer.add_special_tokens({
        'eos_token': '[EOS]',
        'additional_special_tokens': [
            '[QUESTION]',
            '[CONTEXT]',
            '[ANSWER]',
        ]
    })
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = T5ForConditionalGeneration.from_pretrained(args.model_file, device_map="auto", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    #torch.eval()

    loss_fnc= CrossEntropyLoss(reduction='none')
    t5_prefix = 'answer question from context '

    shard_start = args.shard_idx * len(data) // args.num_shard
    shard_end = min((args.shard_idx + 1) * len(data) // args.num_shard, len(data))
    print(str(shard_start) + " " + str(shard_end))

    input_data = list()
    ppl_per_entry = dict()
    for it, entry in tqdm(enumerate(data[shard_start:shard_end])):
        question = entry['question']
        ctx = entry['positive_ctxs'][0]['text']
        answer = entry['answers'][0]
        input_data.append([it, question, ctx, answer, ctx, -1])
        input_data.extend(create_perturb_ctxs(it, question, ctx, answer, t5_prefix))
    
    indices = [[inp[0], inp[1], inp[4], inp[5]] for inp in input_data]
    input_texts = [inp[2] for inp in input_data]
    output_texts = [inp[3] for inp in input_data]
        
    ppls = get_perplexity(model, tokenizer, loss_fnc, input_texts, output_texts, batch=32)
    ppls_indexed = [ind + [pp] for ind, pp in zip(indices, ppls)]

    with open(f'hotpotqa_cleaned_evidentiality_results_uqat5.pkl_{args.shard_idx}','wb') as f:
        pickle.dump(ppls_indexed, f)

if __name__ == '__main__':
    main()