# %%
import os
os.environ["OPENAI_API_KEY"] = ""

# %%
from langchain.llms import OpenAI
llm = OpenAI(openai_api_key="OPENAI_API_KEY")

# %%
from langchain.llms import OpenAI

# %%
llm = OpenAI(temperature=0.9)

# %%
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["question","passage"],
    template="Given a passage and a question, revisit the passage below so as not to include an answer to the question.\nQuestion: {question} \nPassage: {passage}",
)

# %%
print(prompt.format(question="colorful socks", passage="rainbow"))

# %%
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# %%
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(temperature=0.5)

template = "paraphrase the passage below so as can not answer to the question."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "Question: {question} \nPassage: {passage}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# get a chat completion from the formatted messages
# chat(chat_prompt.format_prompt(question="I love programming.", passage=" ranran").to_messages())
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})

chain = LLMChain(llm=chat, prompt=chat_prompt)
chain.run(question="who got the first nobel prize in physics", passage="The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen , of Germany , who received 150,782 SEK , which is equal to 7,731,004 SEK in December 2007 . John Bardeen is the only laureate to win the prize twice -- in 1956 and 1972 . Maria Skłodowska - Curie also won two Nobel Prizes , for physics in 1903 and chemistry in 1911 . William Lawrence Bragg was , until October 2014 , the youngest ever Nobel laureate ; he won the prize in 1915 at the age of 25 . Two women have won the prize : Curie and Maria Goeppert - Mayer ( 1963 ) . As of 2017 , the prize has been awarded to 206 individuals . There have been six years in which the Nobel Prize in Physics was not awarded ( 1916 , 1931 , 1934 , 1940 -- 1942 ) .")

# %%
import pandas as pd

data = pd.read_csv('/home/leedhn/nas2/PiCL_DPR/downloads/data/wikipedia_split/qrels_gold_passages.csv', sep='\t', header=None)
data.columns=['idx','question','psg_idx','passage']
data.head()
# chain.run(question="who got the first nobel prize in physics", passage="The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen , of Germany , who received 150,782 SEK , which is equal to 7,731,004 SEK in December 2007 . John Bardeen is the only laureate to win the prize twice -- in 1956 and 1972 . Maria Skłodowska - Curie also won two Nobel Prizes , for physics in 1903 and chemistry in 1911 . William Lawrence Bragg was , until October 2014 , the youngest ever Nobel laureate ; he won the prize in 1915 at the age of 25 . Two women have won the prize : Curie and Maria Goeppert - Mayer ( 1963 ) . As of 2017 , the prize has been awarded to 206 individuals . There have been six years in which the Nobel Prize in Physics was not awarded ( 1916 , 1931 , 1934 , 1940 -- 1942 ) .")

# %%
data['chatgpt_passage']=None
data.head()

# %%
all_data = pd.read_csv('/home/leedhn/nas2/PiCL_DPR/downloads/data/wikipedia_split/psgs_gold.csv',sep='\t')

# %%
all_data.head()

# %%
gold_data = pd.merge(data, all_data, left_on='psg_idx', right_on='id', how='left')

# %%
gold_data.shape, all_data.shape, data.shape

# %%
gold_data.head()

# %%
for i in range(10):
    gold_data[f'chatgpt_passage_{i}']=[[] for j in range(gold_data.shape[0])]

# %%
gold_data.head()

# %%
gold_data['text'][1376]

# %%
from tqdm import tqdm
chat = ChatOpenAI(temperature=0.5)
chain = LLMChain(llm=chat, prompt=chat_prompt)

for index, line in tqdm(gold_data.iterrows(), total=gold_data.shape[0]):
    # chatgpt_psg = []
    for i in range(10):
        gold_data[f'chatgpt_passage_{i}'][index].append(chain.run(question=line['question'], passage=f"{line['text']} {line['title']}"))
        # print(chatgpt_psg[-1])
    

gold_data.head()

# %%
gold_data.to_csv("chatgpt_gen_psgs_gold.csv",sep='\t')

# %%



