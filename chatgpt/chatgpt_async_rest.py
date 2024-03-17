
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
    # ans = openai.Completion.create(
                # model="text-davinci-002",
                # model='text-davinci-003',
                model="gpt-3.5-turbo",
                max_tokens=512,
                # stop=['[STOP]'],
                messages=[{"role": "user", "content": cur_prompt}],
                # prompt=cur_prompt,
                top_p=0.95,
                n=20,
                temperature=0.5)

    # print(ans)
    # returned = [line['text'] for line in ans['choices']]#ans['choices'][0]['text']
    returned = [line['message']['content'] for line in ans['choices']]
    # print( greenify(returned), end='')
    # print(ans)
    

    return returned


import json
import tiktoken
async def promptf(row, question, text, title):
  passage = text+' '+title
  
  enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
  passage_len = len(enc.encode(passage))
  if passage_len > 3300: #truncate
    passage = enc.decode(enc.encode(passage)[:2900])

  # cur_prompt=f"Given question and passage\n1. find an answer\n2. paraphrase the passage with removing the answer from it.\nQuestion: {question} \nPassage: {passage} \nAnswer: "
  cur_prompt=f"Please paraphrase the passage. Not explaining.\nPassage: {passage}\nParaphrasead passage: "

  

  # ret_text= await call_gpt(cur_prompt)
  # cur_prompt = cur_prompt + ret_text[0]+'\n Paraphrased passage: '
  print(f"{row} is generating...")
  ret_texts = await call_gpt(cur_prompt)
  time.sleep(5)
  with open(f"chatgpt_gen_cf_new/{row}.json",'w') as f:
    json.dump(ret_texts, f)
  print('generated at',f"chatgpt_gen_cf_new/{row}.json")
  return ret_texts

# question="who got the first nobel prize in physics"
# text="The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen , of Germany , who received 150,782 SEK , which is equal to 7,731,004 SEK in December 2007 . John Bardeen is the only laureate to win the prize twice -- in 1956 and 1972 . Maria Skłodowska - Curie also won two Nobel Prizes , for physics in 1903 and chemistry in 1911 . William Lawrence Bragg was , until October 2014 , the youngest ever Nobel laureate ; he won the prize in 1915 at the age of 25 . Two women have won the prize : Curie and Maria Goeppert - Mayer ( 1963 ) . As of 2017 , the prize has been awarded to 206 individuals . There have been six years in which the Nobel Prize in Physics was not awarded ( 1916 , 1931 , 1934 , 1940 -- 1942 ) ."
# title="List of Nobel laureates in Physics"
# promptf(0,question,text,title)


import pandas as pd

# data = pd.read_csv('/home/leedhn/nas2/PiCL_DPR/downloads/data/wikipedia_split/qrels_gold_passages.csv', sep='\t', header=None)
# data.columns=['idx','question','psg_idx','passage']

data = pd.read_csv('/home/leedhn/nas2/PiCL_DPR/downloads/data/wikipedia_split/qrels_gold_passages.csv', sep='\t', header=None)
data.columns=['idx','question','psg_idx','passage']
# all_data = pd.read_csv('/home/leedhn/nas2/PiCL_DPR/downloads/data/wikipedia_split/psgs_gold.csv',sep='\t')
all_data = pd.read_csv('/home/leedhn/nas2/PiCL_DPR/downloads/data/wikipedia_split/psgs_cf.tsv',sep='\t')
all_data.columns=['id','text','title']
ids = list(all_data['id'])
idx = [int(i[8:]) for i in ids]
all_data['idx']=idx
print(all_data.head())
# gold_data = pd.merge(data, all_data, left_on='idx',right_index=True, how='left').fillna('')
gold_data = pd.merge(data, all_data, left_on='idx',right_on='idx', how='left').fillna('')


# gold_data['chatgpt_passage']=[[] for j in range(gold_data.shape[0])]

nq_test = pd.read_csv('/home/leedhn/nas2/PiCL_DPR/downloads/data/retriever/qas/nq-test.csv',sep='\t',header=None)
nq_test.columns=['question','answers']

gold_data_with_ans = pd.merge(gold_data, nq_test['answers'], left_on='idx',right_index=True)
print(gold_data_with_ans.head())
# all_data = pd.read_csv('/home/leedhn/nas2/PiCL_DPR/downloads/data/wikipedia_split/psgs_gold.csv',sep='\t')
# gold_data = pd.merge(data, all_data, left_on='idx',right_index=True, how='left')
# gold_data['chatgpt_passage']=[[] for j in range(gold_data.shape[0])]
# print(gold_data.shape)


from tqdm.asyncio import tqdm_asyncio

gen_lists = list(os.listdir('/home/leedhn/nas2/PiCL_DPR/chatgpt_gen_cf_new'))
gen_lists = [int(name[:-5]) for name in gen_lists if 'json' in name]
gen_lists.sort()
rest_idx = list(set(gold_data_with_ans['idx'])-set(gen_lists))
print('Rests num : ',len(rest_idx))

async def run_chatgpt(indexes):
    # for index, line in (gold_data.iterrows()):
        # chatgpt_psg = []
    # chat_gen = [promptf(index,line['question'],line['passage'],'') for index, line in (data.loc[data['idx'].isin(list(set(list(data['idx']))^set(gen_lists)))].iterrows())]
    chat_gen = [promptf(line['idx'],line['question'],line['text'],line['title']) for index, line in (gold_data_with_ans.loc[gold_data_with_ans['idx'].isin(indexes)].iterrows())]
        # gold_data[f'chatgpt_passage'][index]+=chat_gen
    #, passage=f"{} {}"))
    # print(gold_data[f'chatgpt_passage'][index])
    # break
    
    await tqdm_asyncio.gather(*chat_gen)


# async def main():
#     await run_chatgpt()


# asyncio.run(main())

async def main(indexes):
      await run_chatgpt(indexes)



if __name__=="__main__":
  if len(rest_idx) > 300:
    for start_idx in tqdm(range(0,len(rest_idx), 300)):
      cur_idx = rest_idx[start_idx:start_idx+300]
      asyncio.run(main(cur_idx))
  else:
    asyncio.run(main(rest_idx))


