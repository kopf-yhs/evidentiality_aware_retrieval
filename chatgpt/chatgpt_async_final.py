
import os
import openai
import asyncio
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import time

os.environ["OPENAI_API_KEY"] =''
openai.api_key=os.getenv("OPENAI_API_KEY")

async def call_gpt(cur_prompt):
    ans = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                max_tokens=512,
                messages=[{"role": "user", "content": cur_prompt}],
                top_p=0.95,
                n=20,
                temperature=0.5)
    returned = [line['message']['content'] for line in ans['choices']]
    return returned


import json
import tiktoken
async def promptf(row, question, text, title):
  passage = text+' '+title
  
  enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
  passage_len = len(enc.encode(passage))
  if passage_len > 3300: #truncate
    passage = enc.decode(enc.encode(passage)[:2900])

  cur_prompt=f"Please revise the passage. Do not start with 'This passage'.\nPassage: {passage}\nParaphrasead passage: "


  print(f"{row} is generating...")
  ret_texts = await call_gpt(cur_prompt)
  time.sleep(5)
  with open(f"chatgpt_gen_cf_final/{row}.json",'w') as f:
    json.dump(ret_texts, f)
  print('generated at',f"chatgpt_gen_cf_final/{row}.json", flush=True)
  return ret_texts


import pandas as pd

data = pd.read_csv('/home/leedhn/nas2/PiCL_DPR/downloads/data/retriever/qas/nq-test-gold.csv', sep='\t', header=None)
data.columns=['question','passage']
all_data = pd.read_csv('/home/leedhn/nas2/PiCL_DPR/downloads/data/wikipedia_split/psgs_cf.tsv',sep='\t')
all_data.columns=['id','text','title']
ids = list(all_data['id'])
idx = [int(i[8:]) for i in ids]
all_data['idx']=idx
gold_data = pd.merge(data, all_data, left_index=True,right_on='idx', how='right').fillna('')


from tqdm.asyncio import tqdm_asyncio

gen_lists = list(os.listdir('/home/leedhn/nas2/PiCL_DPR/chatgpt_gen_cf_final'))
gen_lists = [int(name[:-5]) for name in gen_lists if 'json' in name]
gen_lists.sort()
rest_idx = list(set(list(gold_data['idx']))-set(gen_lists))
print('Rests num : ',len(rest_idx))

async def run_chatgpt(indexes):
  chat_gen = [promptf(line['idx'],line['question'],line['text'],line['title']) for index, line in (gold_data.loc[gold_data['idx'].isin(indexes)].iterrows())]
  await tqdm_asyncio.gather(*chat_gen)



async def main(indexes):
      await run_chatgpt(indexes)



if __name__=="__main__":
  if len(rest_idx) > 300:
    for start_idx in tqdm(range(0,len(rest_idx), 300)):
      cur_idx = rest_idx[start_idx:start_idx+300]
      asyncio.run(main(cur_idx))
  else:
    asyncio.run(main(rest_idx))


