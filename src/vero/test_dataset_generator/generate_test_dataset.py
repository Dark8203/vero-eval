#!/usr/bin/env python
# coding: utf-8

# In[49]:

import os
import pandas as pd
import numpy as np
from openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
import json
# import sys
# sys.path.insert(0, r"C:\Users\Alakh Agrawal\OneDrive\New folder (2)\rag_rt")
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
import time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from typing import List, Literal
from pydantic import BaseModel, Field, conlist
from tqdm import tqdm


# In[50]:


pd.set_option('display.max_colwidth', 200)


# In[6]:


load_dotenv(find_dotenv())


# In[7]:



# In[8]:

api_key = os.getenv('VERO_API_KEY')
openai_api_key = api_key[19:-23]
client = OpenAI(api_key=openai_api_key)

# print('Hey')
# ## semantic chunking

# In[9]:


import codex_working


# In[10]:


import importlib


# In[11]:


importlib.reload(codex_working)
import codex_working


# In[12]:


from prompts import system_prompt_basic, system_prompt_chunk_boundary, system_prompt_chunk_length, system_prompt_query_intent, user_prompt


# In[13]:


### 1-2 marks questions - take some random parts of docs and generate 1000s of factual one-on-one questions like facts numbers
### 5 marks ques - more complicated, collated chunks (4-5) - 100s
### 10 marks complex questions - elaborate questions with layers of thinking needed and a lot of information far away in docs - 1s
### ques - not in docs - 10s


def get_openai_resp_struct(system_prompt: str, user_prompt: str, info_chunk_inp: dict, resp_format, model_id: str = "o3-mini-2025-01-31"):
    """
    Returns a Pydantic-validated structure (QAResponse) with 5 Q&A items.
    """
    formatted_user = user_prompt.format(info_chunk=info_chunk_inp)
    response = client.responses.parse(
        model=model_id,
        input=formatted_user,          # user prompt (runtime-filled)
        instructions=(
            system_prompt
            + "\n\n[STRUCTURE] Respond ONLY as JSON matching the provided schema. "
        ),
        text_format=resp_format,    # <-- Pydantic model (schema)
        max_output_tokens=50000
    )
    return getattr(response, "output_parsed", response)


# ## 1/2 marks ques

# In[33]:


class QAItem_basic(BaseModel):
    question: str = Field(..., description="The question text. Must NOT mention chunk IDs.")
    answer: str = Field(..., description="A single, unambiguous factual answer supported by the passages.")
    chunk_ids: List[str] = Field(
        ..., description="List of relevant chunk IDs (strings) that support the answer."
    )
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        ..., description="Declared difficulty level."
    )


# In[34]:


class QAResponse_basic(BaseModel):
    # Exactly 5 items (1 Easy, 2 Medium, 2 Hard). The model is told to respect this in the instructions.
    items: conlist(QAItem_basic, min_length=5, max_length=5)


# In[35]:


def qaresponse_to_df_basic(qa_response):
    """
    Convert a QAResponse (Pydantic object) into a pandas DataFrame.
    Each row = one Q&A item.
    """
    records = [
        {
            "Question": item.question,
            "Answer": item.answer,
            "Chunk IDs": ", ".join(item.chunk_ids),  # flatten list into string
            "Difficulty": item.difficulty
        }
        for item in qa_response.items
    ]
    return pd.DataFrame(records)


# In[36]:


def get_QA_basic(dfct, n=10):
    df_r = pd.DataFrame(columns=['Question', 'Answer', 'Chunk IDs', 'Difficulty'])
    for i1 in range(int(np.ceil(n/5))):
        sample_chunks = dfct.sample(20)
        cd = sample_chunks[['chunk_id', 'text']].to_dict(orient='records')
        # for i in range(20):
        #     cd[str(i)] = sample_chunks[i].model_dump()['page_content']
        resp1 = get_openai_resp_struct(system_prompt_basic, user_prompt, json.dumps(cd), QAResponse_basic)
        df1 = qaresponse_to_df_basic(resp1)
        df_r = pd.concat([df_r, df1])
        time.sleep(2)
    return df_r


# In[ ]:





# ## challenge chunking startegy

# In[ ]:





# #### retrieval bias towards chunk length

# In[ ]:





# In[37]:


class QAItem_len_bias(BaseModel):
    question: str = Field(..., description="The question text. Must NOT mention chunk IDs.")
    answer: str = Field(..., description="A single, unambiguous factual answer supported by the passages.")
    more_relevant_chunk_ids: List[str] = Field(
        ..., description="List of more relevant chunk IDs (strings) that support the answer."
    )
    less_relevant_chunk_ids: List[str] = Field(
        ..., description="List of less relevant chunk IDs (strings) that support the answer."
    )
    short_rationale: str = Field(..., description="short reasoning highlighting discriminator favouring shorter chunks")
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        ..., description="Declared difficulty level."
    )


# In[38]:


class QAResponse_len_bias(BaseModel):
    # Exactly 5 items (1 Easy, 2 Medium, 2 Hard). The model is told to respect this in the instructions.
    items: conlist(QAItem_len_bias, min_length=8, max_length=8)


# In[39]:


def qaresponse_to_df_len_bias(qa_response):
    """
    Convert a QAResponse_len_bias object into a DataFrame.
    Each row = one QAItem_len_bias.
    """
    rows = []
    for item in getattr(qa_response, "items", []):
        rows.append({
            "Question": item.question,
            "Answer": item.answer,
            "Chunk IDs": ", ".join(map(str, getattr(item, "more_relevant_chunk_ids", []))),
            "Less Relevant Chunk IDs": ", ".join(map(str, getattr(item, "less_relevant_chunk_ids", []))),
            "Difficulty": item.difficulty,
            "Rationale": getattr(item, "short_rationale", None),
        })
    return pd.DataFrame(
        rows,
        columns=["Question", "Answer", "Chunk IDs",
                 "Less Relevant Chunk IDs", "Difficulty", "Rationale"]
    )


# In[40]:


def get_QA_chunk_length(dfct, n=10):
    dfr = pd.DataFrame(columns=["Question", "Answer", "Chunk IDs", "Less Relevant Chunk IDs", "Difficulty", "Rationale"])
    for i1 in range(int(np.ceil(n/10))):
        lst = []
        
        for i in dfct['cluster_id'].unique():    ## find cluster of chunks with atleast 2 small chunks, long_chunks>=small_chunks and has not already been considered
            dft = dfct[dfct['cluster_id']==i]
            avg_len = np.average(dft['token_len'])
            dft1 = dft[dft['token_len']<50]
            if len(dft1)>1 and len(dft1)/len(dft)<0.5 and i not in lst:
                lst.append(i)
                break

        cd = dft[['chunk_id', 'text']].to_dict(orient='records')
        # for i in range(len(dft)):
        #     cd[i] = dft.iloc[i]['text']

        resp1 = get_openai_resp_struct(system_prompt_chunk_length, user_prompt, json.dumps(cd), QAResponse_len_bias)
        df1 = qaresponse_to_df_len_bias(resp1)

        dfr = pd.concat([dfr, df1])
        time.sleep(2)
        if len(lst)==dfct['cluster_id'].nunique():
            return dfr
    return dfr


# In[ ]:





# #### challenge info in boundary of chunks

# In[ ]:





# In[41]:


class QAItem_boundary(BaseModel):
    question: str = Field(..., description="The question text. Must NOT mention chunk IDs.")
    answer: str = Field(..., description="A single, unambiguous factual answer supported by the passages.")
    chunk_ids: List[str] = Field(
        ..., description="List of relevant chunk IDs (strings) that support the answer."
    )
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        ..., description="Declared difficulty level."
    )
    rationale: str = Field(..., description="1-2 sentences about why this questions tests bounday/synthesis")


# In[42]:


class QAResponse_boundary(BaseModel):
    # Exactly 5 items (1 Easy, 2 Medium, 2 Hard). The model is told to respect this in the instructions.
    items: conlist(QAItem_boundary, min_length=10, max_length=10)


# In[43]:


def qaresponse_to_df_boundary(qa_response):
    """Each row = one QAItem_boundary. Joins chunk_ids; includes rationale."""
    rows = []
    for item in getattr(qa_response, "items", []):
        rows.append({
            "Question": item.question,
            "Answer": item.answer,
            "Chunk IDs": ", ".join(map(str, getattr(item, "chunk_ids", []))),
            "Difficulty": item.difficulty,
            "Rationale": getattr(item, "rationale", None),
        })
    return pd.DataFrame(rows, columns=["Question","Answer","Chunk IDs","Difficulty","Rationale"])


# In[44]:


def get_QA_chunk_boundary(dfct, n=10):
    dfr = pd.DataFrame(columns=["Question","Answer","Chunk IDs","Difficulty","Rationale"])
    for i1 in range(int(np.ceil(n/10))):
        lst = []
        
        for i in dfct['cluster_id'].unique():    ## find cluster of chunks with atleast 2 small chunks, long_chunks>=small_chunks and has not already been considered
            dft = dfct[dfct['cluster_id']==i]
            avg_len = np.average(dft['token_len'])
            dft1 = dft[dft['token_len']<50]
            if len(dft1)>1 and len(dft1)/len(dft)<0.5 and i not in lst:
                lst.append(i)
                break

        cd = dft[['chunk_id', 'text']].to_dict(orient='records')
        # for i in range(len(dft)):
        #     cd[i] = dft.iloc[i]['text']

        resp1 = get_openai_resp_struct(system_prompt_chunk_boundary, user_prompt, json.dumps(cd), QAResponse_boundary)
        df1 = qaresponse_to_df_boundary(resp1)

        dfr = pd.concat([dfr, df1])
        time.sleep(2)
        if len(lst)==dfct['cluster_id'].nunique():
            return dfr
    return dfr


# In[ ]:





# #### challenging user query intent understanding - complex domain terms, complicated/unclear/poor language queries

# In[ ]:





# In[45]:


class QAItem_intent(BaseModel):
    question: str = Field(..., description="The question text. Must NOT mention chunk IDs.")
    answer: str = Field(..., description="A single, unambiguous factual answer supported by the passages.")
    chunk_ids: List[str] = Field(
        ..., description="List of relevant chunk IDs (strings) that support the answer."
    )
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        ..., description="Declared difficulty level."
    )
    rationale: str = Field(..., description="1-2 sentences about why this questions tests bounday/synthesis")


# In[46]:


class QAResponse_intent(BaseModel):
    # Exactly 5 items (1 Easy, 2 Medium, 2 Hard). The model is told to respect this in the instructions.
    items: conlist(QAItem_intent, min_length=10, max_length=10)


# In[47]:


def qaresponse_to_df_intent(qa_response):
    """Each row = one QAItem_intent. Joins chunk_ids; includes rationale."""
    rows = []
    for item in getattr(qa_response, "items", []):
        rows.append({
            "Question": item.question,
            "Answer": item.answer,
            "Chunk IDs": ", ".join(map(str, getattr(item, "chunk_ids", []))),
            "Difficulty": item.difficulty,
            "Rationale": getattr(item, "rationale", None),
        })
    return pd.DataFrame(rows, columns=["Question","Answer","Chunk IDs","Difficulty","Rationale"])


# In[48]:


def get_QA_query_intent(dfct, n=10):
    dfr = pd.DataFrame(columns=["Question","Answer","Chunk IDs","Difficulty","Rationale"])
    for i1 in range(int(np.ceil(n/10))):
        lst = []
        
        for i in dfct['cluster_id'].unique():    ## find cluster of chunks with atleast 2 small chunks, long_chunks>=small_chunks and has not already been considered
            dft = dfct[dfct['cluster_id']==i]
            avg_len = np.average(dft['token_len'])
            dft1 = dft[dft['token_len']<50]
            if len(dft1)>1 and len(dft1)/len(dft)<0.5 and i not in lst:
                lst.append(i)
                break

        cd = dft[['chunk_id', 'text']].to_dict(orient='records')
        # for i in range(len(dft)):
        #     cd[i] = dft.iloc[i]['text']

        resp1 = get_openai_resp_struct(system_prompt_query_intent, user_prompt, json.dumps(cd), QAResponse_intent)
        df1 = qaresponse_to_df_intent(resp1)

        dfr = pd.concat([dfr, df1])
        time.sleep(2)
        if len(lst)==dfct['cluster_id'].nunique():
            return dfr
    return dfr


# In[ ]:





# In[48]:

def generate_and_save(data_path, usecase, save_path='test_dataset_generator.csv', n_queries=50):
    
    # data_path = data_path
    loader = DirectoryLoader(data_path, glob = '*.pdf', loader_cls = PyPDFLoader)
    docs = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 400, length_function = len, add_start_index = True)
    # chunks = text_splitter.split_documents(docs)


    # In[14]:


    chunks = codex_working.semantically_chunk_documents(
        docs,                                  # same input as before
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # small, fast model
        min_tokens=80,                         # prevent overly small chunks
        max_tokens=350,                        # keep chunks within retriever budget
        similarity_threshold=0.6,              # cohesion control; higher = stricter
        overlap_sentences=1,                   # carry 1 sentence into next chunk
    )

    dfc = codex_working.chunks_to_df(chunks)

    dfc['chunk_id'] = dfc.index


    dfc1 = codex_working.cluster_chunks_df(dfc)
    
    for iterationi in tqdm(range(4)):
        if iterationi==0:
            df1 = get_QA_basic(dfc1.copy(), n=np.ceil(n_queries/4).astype(int))
        elif iterationi==1:
            df2 = get_QA_chunk_length(dfc1.copy(), n=np.ceil(n_queries/4).astype(int))
        elif iterationi==2:
            df3 = get_QA_chunk_boundary(dfc1.copy(), n=np.ceil(n_queries/4).astype(int))
        elif iterationi==3:
            df4 = get_QA_query_intent(dfc1.copy(), n=np.ceil(n_queries/4).astype(int))

    df1['Rationale'] = 'None'
    df1['check_metric'] = 'general'
    df2['check_metric'] = 'chunk_length'
    df3['check_metric'] = 'chunk_boundary'
    df4['check_metric'] = 'user_intent'
    
    df1['Less Relevant Chunk IDs'] = np.nan
    df3['Less Relevant Chunk IDs'] = np.nan
    df4['Less Relevant Chunk IDs'] = np.nan
    
    df1 = df1[['Question', 'Answer', 'Chunk IDs', 'Less Relevant Chunk IDs', 'Difficulty', 'Rationale', 'check_metric']]
    df2 = df2[['Question', 'Answer', 'Chunk IDs', 'Less Relevant Chunk IDs', 'Difficulty', 'Rationale', 'check_metric']]
    df3 = df3[['Question', 'Answer', 'Chunk IDs', 'Less Relevant Chunk IDs', 'Difficulty', 'Rationale', 'check_metric']]
    df4 = df4[['Question', 'Answer', 'Chunk IDs', 'Less Relevant Chunk IDs', 'Difficulty', 'Rationale', 'check_metric']]

    df_test = pd.concat([df1, df2, df3, df4])

    df_test = df_test.reset_index(drop=True)

    # df_test



    df_test.to_csv(save_path)
    return "Created and Saved successfully"

# generate_and_save(data_path=r'../data/',
#                 usecase='usecase_1',
#                 save_path='trial_test_set_1.csv',
#                 n_queries=30)
