import pandas as pd
import numpy as np
import os
import re
from openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import random
import json
from dotenv import load_dotenv, find_dotenv
import langgraph
from datetime import date
from fastapi import FastAPI
from pydantic import BaseModel
import time
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
import sys
import requests
from pprint import pprint
import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import asyncio
from typing import List, Literal
from pydantic import BaseModel, Field, conlist
from tqdm import tqdm

pd.set_option('display.max_colwidth', 200)
load_dotenv(find_dotenv())

client = OpenAI()

import chunking_utilities
import importlib
importlib.reload(chunking_utilities)
import chunking_utilities

from prompts import system_prompt_basic, system_prompt_chunk_boundary, system_prompt_chunk_length, \
    system_prompt_query_intent, user_prompt

### 1-2 marks questions - take some random parts of docs and generate 1000s of factual one-on-one questions like facts numbers
### 5 marks ques - more complicated, collated chunks (4-5) - 100s
### 10 marks complex questions - elaborate questions with layers of thinking needed and a lot of information far away in docs - 1s
### ques - not in docs - 10s


def get_openai_resp_struct(system_prompt: str, user_prompt: str, info_chunk_inp: dict, resp_format,
                           model_id: str = "o3-mini-2025-01-31"):
    """
    Build and call the OpenAI Responses API, returning a Pydantic-validated structure.

    Purpose:
    - Format the provided user prompt with `info_chunk_inp`, call the responses.parse
      endpoint and return the parsed output (preferably a Pydantic model instance).
    Inputs:
    - system_prompt: The system-level instruction string to bias behavior.
    - user_prompt: The user-level prompt template (must include '{info_chunk}' when used that way).
    - info_chunk_inp: The chunk payload (already JSON-serializable or a string) to inject.
    - resp_format: A Pydantic model class or structure that the SDK will use to validate the response.
    - model_id: Model identifier string used for the Responses API call.
    Returns:
    - The parsed output (usually a Pydantic model instance) if parsing succeeded, otherwise the raw response.
    Notes:
    - The function relies on the `client` OpenAI instance already created.
    """
    formatted_user = user_prompt.format(info_chunk=info_chunk_inp)
    response = client.responses.parse(
        model=model_id,
        input=formatted_user,  # user prompt (runtime-filled)
        instructions=(
                system_prompt
                + "\n\n[STRUCTURE] Respond ONLY as JSON matching the provided schema. "
        ),
        text_format=resp_format,  # <-- Pydantic model (schema)
        max_output_tokens=50000
    )
    return getattr(response, "output_parsed", response)


class QAItem_basic(BaseModel):
    """
    Schema for a single basic QA item.

    Fields:
    - question: The question text. Must NOT mention chunk IDs.
    - answer: A single, unambiguous factual answer supported by the passages.
    - chunk_ids: List of relevant chunk IDs that support the answer.
    - difficulty: Declared difficulty level: "Easy", "Medium", or "Hard".
    """
    question: str = Field(..., description="The question text. Must NOT mention chunk IDs.")
    answer: str = Field(..., description="A single, unambiguous factual answer supported by the passages.")
    chunk_ids: List[str] = Field(
        ..., description="List of relevant chunk IDs (strings) that support the answer."
    )
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        ..., description="Declared difficulty level."
    )

class QAResponse_basic(BaseModel):
    """
    Response container for a group of basic QA items.

    Constraints:
    - Contains exactly 5 items (1 Easy, 2 Medium, 2 Hard) as enforced by the calling prompts.
    """
    # Exactly 5 items (1 Easy, 2 Medium, 2 Hard). The model is told to respect this in the instructions.
    items: conlist(QAItem_basic, min_length=5, max_length=5)


def qaresponse_to_df_basic(qa_response):
    """
    Convert a QAResponse (Pydantic object) into a pandas DataFrame.

    Each row corresponds to a single QAItem_basic. This helper flattens the
    chunk ID lists into comma-separated strings for CSV-friendly output.
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


def get_QA_basic(dfct, n=10):
    """
    Generate basic QA items by sampling chunks and calling the model.

    Workflow:
    - Repeats calls to the Responses API until `n` items are gathered (requests are batched,
      each response is expected to contain exactly 5 items).
    - Samples a small set of chunks per call to provide grounding contexts.
    Inputs:
    - dfct: DataFrame of chunks with at least 'chunk_id' and 'text' columns.
    - n: Desired number of questions to produce.
    Returns:
    - DataFrame with generated QA items (columns: Question, Answer, Chunk IDs, Difficulty).
    """
    df_r = pd.DataFrame(columns=['Question', 'Answer', 'Chunk IDs', 'Difficulty'])
    for i1 in range(int(np.ceil(n / 5))):
        sample_chunks = dfct.sample(20)
        cd = sample_chunks[['chunk_id', 'text']].to_dict(orient='records')
        # for i in range(20):
        #     cd[str(i)] = sample_chunks[i].model_dump()['page_content']
        resp1 = get_openai_resp_struct(system_prompt_basic, user_prompt, json.dumps(cd), QAResponse_basic)
        df1 = qaresponse_to_df_basic(resp1)
        df_r = pd.concat([df_r, df1])
        time.sleep(2)
    return df_r

class QAItem_len_bias(BaseModel):
    """
    Schema for QA items focusing on chunk-length bias.

    Fields:
    - question: The question text. Must NOT mention chunk IDs.
    - answer: A single, unambiguous factual answer supported by the passages.
    - more_relevant_chunk_ids: Chunks judged more relevant for the QA pair.
    - less_relevant_chunk_ids: Chunks judged less relevant.
    - short_rationale: Short reasoning that explains the discriminator favoring shorter chunks.
    - difficulty: Declared difficulty level.
    """
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


class QAResponse_len_bias(BaseModel):
    """
    Response container for QA items that test chunk-length bias.

    Constraints:
    - Contains a fixed number of items (here min_length=8, max_length=8 as used in prompts).
    """
    # Exactly 5 items (1 Easy, 2 Medium, 2 Hard). The model is told to respect this in the instructions.
    items: conlist(QAItem_len_bias, min_length=8, max_length=8)


def qaresponse_to_df_len_bias(qa_response):
    """
    Convert a QAResponse_len_bias object into a DataFrame.

    Flattens both more/less relevant chunk id lists and includes the rationale field.
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



def get_QA_chunk_length(dfct, n=10):
    """
    Generate QA items that test sensitivity to chunk length.

    Workflow:
    - Scans cluster groups looking for clusters with multiple short chunks and
      satisfies other heuristics, then calls the model to produce QA items.
    Inputs:
    - dfct: DataFrame containing chunk metadata including 'cluster_id' and 'token_len'.
    - n: Number of questions requested (function batches calls tuned to prompt sizes).
    Returns:
    - DataFrame of generated QA items with length-bias annotations.
    """
    dfr = pd.DataFrame(
        columns=["Question", "Answer", "Chunk IDs", "Less Relevant Chunk IDs", "Difficulty", "Rationale"])
    for i1 in range(int(np.ceil(n / 10))):
        lst = []

        for i in dfct[
            'cluster_id'].unique():  ## find cluster of chunks with atleast 2 small chunks, long_chunks>=small_chunks and has not already been considered
            dft = dfct[dfct['cluster_id'] == i]
            avg_len = np.average(dft['token_len'])
            dft1 = dft[dft['token_len'] < 50]
            if len(dft1) > 1 and len(dft1) / len(dft) < 0.5 and i not in lst:
                lst.append(i)
                break

        cd = dft[['chunk_id', 'text']].to_dict(orient='records')
        # for i in range(len(dft)):
        #     cd[i] = dft.iloc[i]['text']

        resp1 = get_openai_resp_struct(system_prompt_chunk_length, user_prompt, json.dumps(cd), QAResponse_len_bias)
        df1 = qaresponse_to_df_len_bias(resp1)

        dfr = pd.concat([dfr, df1])
        time.sleep(2)
        if len(lst) == dfct['cluster_id'].nunique():
            return dfr
    return dfr

class QAItem_boundary(BaseModel):
    """
    Schema for QA items that probe chunk boundary/synthesis behavior.

    Fields:
    - question: The question text. Must NOT mention chunk IDs.
    - answer: A single factual answer supported by passages.
    - chunk_ids: Relevant chunk identifiers.
    - difficulty: Declared difficulty level.
    - rationale: 1-2 sentences describing why this question tests boundary or synthesis behavior.
    """
    question: str = Field(..., description="The question text. Must NOT mention chunk IDs.")
    answer: str = Field(..., description="A single, unambiguous factual answer supported by the passages.")
    chunk_ids: List[str] = Field(
        ..., description="List of relevant chunk IDs (strings) that support the answer."
    )
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        ..., description="Declared difficulty level."
    )
    rationale: str = Field(..., description="1-2 sentences about why this questions tests bounday/synthesis")


class QAResponse_boundary(BaseModel):
    """
    Container for boundary-testing QA items.

    Constraints:
    - Expected number of items as enforced by the calling prompt (here min_length=10, max_length=10).
    """
    # Exactly 5 items (1 Easy, 2 Medium, 2 Hard). The model is told to respect this in the instructions.
    items: conlist(QAItem_boundary, min_length=10, max_length=10)



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
    return pd.DataFrame(rows, columns=["Question", "Answer", "Chunk IDs", "Difficulty", "Rationale"])


def get_QA_chunk_boundary(dfct, n=10):
    """
    Generate QA items that stress chunk boundary and synthesis handling.

    Workflow:
    - Looks for clusters with a mix of short/long chunks and requests boundary-testing items
      from the model using the boundary-focused system prompt.
    Inputs:
    - dfct: DataFrame of chunks with clustering information.
    - n: Number of questions requested.
    Returns:
    - DataFrame with boundary-focused QA items.
    """
    dfr = pd.DataFrame(columns=["Question", "Answer", "Chunk IDs", "Difficulty", "Rationale"])
    for i1 in range(int(np.ceil(n / 10))):
        lst = []

        for i in dfct[
            'cluster_id'].unique():  ## find cluster of chunks with atleast 2 small chunks, long_chunks>=small_chunks and has not already been considered
            dft = dfct[dfct['cluster_id'] == i]
            avg_len = np.average(dft['token_len'])
            dft1 = dft[dft['token_len'] < 50]
            if len(dft1) > 1 and len(dft1) / len(dft) < 0.5 and i not in lst:
                lst.append(i)
                break

        cd = dft[['chunk_id', 'text']].to_dict(orient='records')
        # for i in range(len(dft)):
        #     cd[i] = dft.iloc[i]['text']

        resp1 = get_openai_resp_struct(system_prompt_chunk_boundary, user_prompt, json.dumps(cd), QAResponse_boundary)
        df1 = qaresponse_to_df_boundary(resp1)

        dfr = pd.concat([dfr, df1])
        time.sleep(2)
        if len(lst) == dfct['cluster_id'].nunique():
            return dfr
    return dfr


class QAItem_intent(BaseModel):
    """
    Schema for QA items that examine query intent understanding.

    Fields:
    - question: The question text. Must NOT mention chunk IDs.
    - answer: A single factual answer supported by the passages.
    - chunk_ids: Relevant chunk identifiers.
    - difficulty: Declared difficulty level.
    - rationale: 1-2 sentences about why the question tests intent understanding.
    """
    question: str = Field(..., description="The question text. Must NOT mention chunk IDs.")
    answer: str = Field(..., description="A single, unambiguous factual answer supported by the passages.")
    chunk_ids: List[str] = Field(
        ..., description="List of relevant chunk IDs (strings) that support the answer."
    )
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        ..., description="Declared difficulty level."
    )
    rationale: str = Field(..., description="1-2 sentences about why this questions tests bounday/synthesis")


class QAResponse_intent(BaseModel):
    """
    Container for intent-focused QA items.

    Constraints:
    - Expected set size as defined by the prompt (here min_length=10, max_length=10).
    """
    # Exactly 5 items (1 Easy, 2 Medium, 2 Hard). The model is told to respect this in the instructions.
    items: conlist(QAItem_intent, min_length=10, max_length=10)



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
    return pd.DataFrame(rows, columns=["Question", "Answer", "Chunk IDs", "Difficulty", "Rationale"])


def get_QA_query_intent(dfct, n=10):
    """
    Generate QA items designed to test query intent coverage.

    Workflow:
    - Selects cluster candidates similar to other generation helpers and asks the model to
      generate items that reveal whether intent is preserved or lost across chunking.
    Inputs:
    - dfct: DataFrame of chunks with clustering and token length metadata.
    - n: Number of questions requested.
    Returns:
    - DataFrame with intent-focused QA items.
    """
    dfr = pd.DataFrame(columns=["Question", "Answer", "Chunk IDs", "Difficulty", "Rationale"])
    for i1 in range(int(np.ceil(n / 10))):
        lst = []

        for i in dfct[
            'cluster_id'].unique():  ## find cluster of chunks with atleast 2 small chunks, long_chunks>=small_chunks and has not already been considered
            dft = dfct[dfct['cluster_id'] == i]
            avg_len = np.average(dft['token_len'])
            dft1 = dft[dft['token_len'] < 50]
            if len(dft1) > 1 and len(dft1) / len(dft) < 0.5 and i not in lst:
                lst.append(i)
                break

        cd = dft[['chunk_id', 'text']].to_dict(orient='records')
        # for i in range(len(dft)):
        #     cd[i] = dft.iloc[i]['text']

        resp1 = get_openai_resp_struct(system_prompt_query_intent, user_prompt, json.dumps(cd), QAResponse_intent)
        df1 = qaresponse_to_df_intent(resp1)

        dfr = pd.concat([dfr, df1])
        time.sleep(2)
        if len(lst) == dfct['cluster_id'].nunique():
            return dfr
    return dfr


def generate_and_save(data_path, usecase, save_path='test_dataset.csv', n_queries=50):
    """
    Orchestrate dataset generation from PDFs and save the resulting QA CSV.

    Steps performed:
    - Load PDF files from `data_path` using DirectoryLoader / PyPDFLoader.
    - Semantically chunk documents (calls chunking_utilities.semantically_chunk_documents).
    - Convert chunks to a DataFrame and cluster them.
    - Generate 4 classes of QA items (basic, chunk_length, chunk_boundary, user_intent)
      by batching calls to the OpenAI Responses API.
    - Concatenate all generated QA items, set some metadata columns and persist to CSV.
    Inputs:
    - data_path: Directory containing PDF files.
    - usecase: Logical label for the use case (not used programmatically here but kept for callers).
    - save_path: Destination CSV path.
    - n_queries: Number of queries to request in total (split across generation modes).
    Returns:
    - A short success string on completion.
    Notes:
    - The function expects `chunking_utilities` helpers (semantically_chunk_documents, chunks_to_df, cluster_chunks_df)
      to be importable and available in the runtime.
    """
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    docs = loader.load()

    chunks = chunking_utilities.semantically_chunk_documents(
        docs,  # same input as before
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # small, fast model
        min_tokens=80,  # prevent overly small chunks
        max_tokens=350,  # keep chunks within retriever budget
        similarity_threshold=0.6,  # cohesion control; higher = stricter
        overlap_sentences=1,  # carry 1 sentence into next chunk
    )

    dfc = chunking_utilities.chunks_to_df(chunks)

    dfc['chunk_id'] = dfc.index

    dfc1 = chunking_utilities.cluster_chunks_df(dfc)

    for iterationi in tqdm(range(4)):
        if iterationi == 0:
            df1 = get_QA_basic(dfc1.copy(), n=np.ceil(n_queries / 4).astype(int))
        elif iterationi == 1:
            df2 = get_QA_chunk_length(dfc1.copy(), n=np.ceil(n_queries / 4).astype(int))
        elif iterationi == 2:
            df3 = get_QA_chunk_boundary(dfc1.copy(), n=np.ceil(n_queries / 4).astype(int))
        elif iterationi == 3:
            df4 = get_QA_query_intent(dfc1.copy(), n=np.ceil(n_queries / 4).astype(int))

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
