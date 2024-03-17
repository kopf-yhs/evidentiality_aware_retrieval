import pandas as pd
import json
import os


PATH='/home/leedhn/nas2/PiCL_DPR/chatgpt_gen_cf_final'
file_list = os.listdir(PATH)
file_list = [item for item in file_list if 'json' in item]
file_list = [int(item[:-5]) for item in file_list]
file_list.sort()
gold_psgs = pd.read_csv('/home/leedhn/nas2/PiCL_DPR/downloads/data/wikipedia_split/grels_gold_psg_title.csv',sep='\t')

data=[[] for _ in range(20)]
for item in file_list:
    with open(f"{PATH}/{item}.json",'r') as f:
        chat_data = json.load(f)
    for i, chat in enumerate(chat_data):
        if f'gold:{item}' in list(gold_psgs['id']):
            data[i].append(
                {
                    'id':f'chat{i}:{item}',
                    'text': chat,
                    'title': list(gold_psgs.loc[gold_psgs['id']==f'gold:{item}']['title'])[0]
                }
            )
        else:
            data[i].append(
                {
                    'id':f'chat{i}:{item}',
                    'text': chat,
                    'title': ''
                }
            )

for i, d in enumerate(data):
    df = pd.DataFrame.from_dict(d)
    print(df.head())
    print(df.shape)
    df.to_csv(f'/home/leedhn/nas2/PiCL_DPR/downloads/data/wikipedia_split/chat_gen_cf_final_{i}.csv',sep='\t',index=False)
    