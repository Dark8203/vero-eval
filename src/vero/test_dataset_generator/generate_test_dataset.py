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

from vero.test_dataset_generator import chunking_utilities

from vero.test_dataset_generator.prompts import system_prompt_basic, system_prompt_chunk_boundary, system_prompt_chunk_length, \
    system_prompt_query_intent, user_prompt_chunks, system_prompt_persona_generation, \
        user_prompt_personas, system_prompt_persona_specific_QA


def get_openai_resp_struct(system_prompt: str, user_prompt: str, info_inp: dict, \
                           resp_format, model_id: str = "o3-mini-2025-01-31"):
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
    if info_inp and user_prompt:
        formatted_user = user_prompt.format(inp=info_inp)
    else:
        formatted_user = "Start the task"
    response = client.responses.parse(
        model=model_id,
        input=formatted_user,  # user prompt (runtime-filled)
        instructions=(
                system_prompt
                + "\n\n[STRUCTURE] Respond ONLY as JSON matching the provided schema. "
        ),
        text_format=resp_format,  # Pydantic model (schema)
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
    for i1 in range(int(np.ceil(n/5))):
        sample_chunks = dfct.sample(20)
        cd = sample_chunks[['chunk_id', 'text']].to_dict(orient='records')
        resp1 = get_openai_resp_struct(system_prompt_basic, user_prompt_chunks, json.dumps(cd), QAResponse_basic)
        df1 = qaresponse_to_df_basic(resp1)
        df_r = pd.concat([df_r, df1])
        time.sleep(1)
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

        resp1 = get_openai_resp_struct(system_prompt_chunk_length, user_prompt_chunks, json.dumps(cd), QAResponse_len_bias)
        df1 = qaresponse_to_df_len_bias(resp1)

        dfr = pd.concat([dfr, df1])
        time.sleep(1)
        if len(lst)==dfct['cluster_id'].nunique():
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

        resp1 = get_openai_resp_struct(system_prompt_chunk_boundary, user_prompt_chunks, json.dumps(cd), QAResponse_boundary)
        df1 = qaresponse_to_df_boundary(resp1)

        dfr = pd.concat([dfr, df1])
        time.sleep(1)
        if len(lst)==dfct['cluster_id'].nunique():
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

        resp1 = get_openai_resp_struct(system_prompt_query_intent, user_prompt_chunks, json.dumps(cd), QAResponse_intent)
        df1 = qaresponse_to_df_intent(resp1)

        dfr = pd.concat([dfr, df1])
        time.sleep(1)
        if len(lst)==dfct['cluster_id'].nunique():
            return dfr
    return dfr

class TechSavviness(BaseModel):
    """
    Schema defining a persona's comfort and proficiency with technology.

    Fields:
    - level: Declared technology familiarity level — must be one of: 'Low', 'Medium', or 'High'.
    - justification: A concise, 1-sentence explanation describing why the persona fits the chosen tech level.
    """
    level: Literal["Low", "Medium", "High"] = Field(..., description="One of: 'Low', 'Medium', 'High'.")
    justification: str = Field(..., description="1-sentence explanation for their tech savviness level.")

class LanguageProficiency(BaseModel):
    """
    Schema describing a persona’s fluency and communication style in English.

    Fields:
    - level: Language proficiency category — must be one of: 'Native', 'Proficient', 'Conversational', or 'Limited'.
    - characteristic: A 1-sentence description of how this proficiency manifests in communication or writing style.
    """
    level: Literal["Native", "Proficient", "Conversational", "Limited"] = Field(..., description="Language level.")
    characteristic: str = Field(..., description="1-sentence description of language use.")

class UserPersona(BaseModel):
    """
    Schema representing a single user persona archetype used for simulation or evaluation.

    Fields:
    - title: Short, descriptive archetype name (e.g., “Data-Driven Analyst”, “Casual Browser”).
    - key_traits_summary: 2–3 sentence overview summarizing core traits, behaviors, and mindset.
    - background: 1–2 sentence summary providing professional or demographic context.
    - tech_savviness: Nested object describing technology familiarity and rationale.
    - primary_goal: The main objective, motivation, or job-to-be-done for this persona.
    - language_proficiency: Nested object detailing English fluency and communication characteristics.
    - behavioral_quirks: 1–2 sentence outline of notable behavioral or interaction quirks.
    """
    title: str = Field(..., description="Short, descriptive archetype.")
    key_traits_summary: str = Field(..., description="2–3 sentence summary of core traits, behaviors, mindset.")
    background: str = Field(..., description="1–2 sentence professional/demographic context.")
    tech_savviness: TechSavviness
    primary_goal: str = Field(..., description="Specific job-to-be-done or primary task.")
    language_proficiency: LanguageProficiency
    behavioral_quirks: str = Field(..., description="1–2 sentence interaction style and behavior.")

class PersonasPayload(BaseModel):
     """
    Schema for the complete persona payload containing a fixed set of persona objects.

    Fields:
    - user_personas: List containing exactly 15 persona entries.
      Each persona object defines traits, background, tech savviness, goals, language proficiency,
      and behavioral quirks, collectively forming a diverse evaluation or simulation set.
    """
     user_personas: conlist(UserPersona, min_length=15, max_length=15) = Field(
        ..., description="List of 15 persona objects."
    )

def qaresponse_to_df_personas(qa_response):
    """
    Convert a PersonasPayload object into a flat DataFrame.

    Each row corresponds to a single UserPersona entry from PersonasPayload.user_personas.

    The function:
    - Accepts either a Pydantic v1/v2 model instance or a plain dictionary.
    - Flattens nested persona fields (tech_savviness, language_proficiency) into top-level DataFrame columns.
    - Produces a tabular structure suitable for inspection, export, or analysis.

    Returns:
        pd.DataFrame: A DataFrame with one row per persona and columns for:
        Title, Key Traits Summary, Background, Tech Savviness Level, Tech Savviness Justification,
        Primary Goal, Language Proficiency Level, Language Proficiency Characteristic, and Behavioral Quirks.
    """
    # Get the list of personas from either model or dict
    if hasattr(qa_response, "user_personas"):
        personas = qa_response.user_personas
    elif isinstance(qa_response, dict):
        personas = qa_response.get("user_personas", [])
    else:
        personas = []

    def _val(obj, attr, default=None):
        # Robustly read attribute from pydantic model or dict
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    rows = []
    for item in personas:
        tech = _val(item, "tech_savviness", {})  # nested: TechSavviness
        lang = _val(item, "language_proficiency", {})  # nested: LanguageProficiency

        # Support both dict and model objects for nested fields
        tech_level = _val(tech, "level")
        tech_just = _val(tech, "justification")
        lang_level = _val(lang, "level")
        lang_char = _val(lang, "characteristic")

        rows.append({
            "Title": _val(item, "title", ""),
            "Key Traits Summary": _val(item, "key_traits_summary", ""),
            "Background": _val(item, "background", ""),
            "Tech Savviness Level": tech_level,
            "Tech Savviness Justification": tech_just,
            "Primary Goal": _val(item, "primary_goal", ""),
            "Language Proficiency Level": lang_level,
            "Language Proficiency Characteristic": lang_char,
            "Behavioral Quirks": _val(item, "behavioral_quirks", ""),
        })

    return pd.DataFrame(rows, columns=[
        "Title",
        "Key Traits Summary",
        "Background",
        "Tech Savviness Level",
        "Tech Savviness Justification",
        "Primary Goal",
        "Language Proficiency Level",
        "Language Proficiency Characteristic",
        "Behavioral Quirks",
    ])

def get_personas(business_use_case):
    """
    Generate a structured set of user personas for a given business use case.

    Workflow:
    - Sends the provided business use case description to the model with a persona generation prompt.
    - Receives a structured PersonasPayload (validated Pydantic model) containing 15 personas.
    - Converts the structured response into a flat DataFrame for analysis or export.

    Inputs:
    - business_use_case: A text description outlining the product, target audience, or business context
      used to guide persona generation.

    Returns:
    - DataFrame of 15 personas, each with fields such as Title, Key Traits Summary, Background,
      Tech Savviness, Language Proficiency, Primary Goal, and Behavioral Quirks.
    """
    resp1 = get_openai_resp_struct(system_prompt_persona_generation, user_prompt_personas, business_use_case, PersonasPayload)
    df1 = qaresponse_to_df_personas(resp1)

    return df1



### generate questions based on each persona

def get_QA_personas(dfct, df_personas, n=5):
    """
    Generate QA items conditioned on different user personas.

    Workflow:
    - Iterates through each persona from the provided persona DataFrame.
    - Dynamically formats a persona-specific system prompt using persona attributes
      (traits, background, goals, tech savviness, and language proficiency).
    - For each persona, samples random document chunks and queries the model
      to generate persona-aligned QA items.
    - Aggregates responses across personas into a unified DataFrame with
      a 'check_metric' column identifying the generating persona.

    Inputs:
    - dfct: DataFrame containing text chunks with 'chunk_id' and 'text' fields.
    - df_personas: DataFrame of persona definitions (flattened from PersonasPayload).
    - n: Number of questions to generate per persona (function batches calls accordingly).

    Returns:
    - DataFrame of generated QA items tagged by persona, with columns:
      ['Question', 'Answer', 'Chunk IDs', 'Difficulty', 'check_metric'].
    """
    df_r = pd.DataFrame(columns=['Question', 'Answer', 'Chunk IDs', 'Difficulty', 'check_metric'])
    for persona_gen in tqdm(json.loads(df_personas.to_json(orient='records')), desc='Running different personas'):
        system_prompt_t = system_prompt_persona_specific_QA.format(
            title=persona_gen['Title'],
            key_traits_summary=persona_gen['Key Traits Summary'],
            background=persona_gen['Background'],
            tech_savviness_level=persona_gen['Tech Savviness Level'],
            tech_savviness_justification=persona_gen['Tech Savviness Justification'],
            primary_goal=persona_gen['Primary Goal'],
            language_proficiency_level=persona_gen['Language Proficiency Level'],
            language_proficiency_characteristic=persona_gen['Language Proficiency Characteristic'],
            behavioural_quirks=persona_gen['Behavioral Quirks']
        )
        for i1 in range(int(np.ceil(n/5))):
            sample_chunks = dfct.sample(20)
            cd = sample_chunks[['chunk_id', 'text']].to_dict(orient='records')
            resp1 = get_openai_resp_struct(system_prompt_t, user_prompt_chunks, json.dumps(cd), QAResponse_basic)
            df1 = qaresponse_to_df_basic(resp1)
            df1['check_metric'] = 'persona_'+str(persona_gen['Title'])
            df_r = pd.concat([df_r, df1])
            time.sleep(1)
    return df_r


def generate_and_save(data_path, usecase = None, save_path_dir='test_dataset', n_queries=100):
    """
    Orchestrate comprehensive test dataset generation from PDFs, including persona-based QA creation,
    and save all intermediate and final artifacts to disk.

    Steps performed:
    - Load PDF files from `data_path` using DirectoryLoader / PyPDFLoader.
    - Semantically chunk documents (via chunking_utilities.semantically_chunk_documents).
    - Convert chunks to a DataFrame, assign chunk IDs, and perform semantic clustering.
    - Persist the clustered dataset as 'chunked_dataset.csv'.
    - Generate multiple classes of QA items:
        (1) Basic QA items (general comprehension),
        (2) Chunk-length sensitivity QA,
        (3) Chunk-boundary synthesis QA,
        (4) Query-intent evaluation QA,
        (5) Persona-conditioned QA (if `usecase` is provided).
    - Automatically generate 15 personas (from the given business use case) and save them as
      'personas_generated.csv' before creating persona-aligned QA items.
    - Concatenate all QA outputs, standardize columns, and save the unified dataset as
      'test_data_generated.csv' under `save_path_dir`.

    Inputs:
    - data_path: Directory path containing PDF source documents.
    - usecase: Optional business use case description used to generate personas and persona-specific QA.
    - save_path_dir: Output directory where all intermediate and final CSVs will be saved.
    - n_queries: Total number of QA items to generate (distributed across all generation modes).

    Returns:
    - A success message string after successfully generating and saving all datasets.

    Notes:
    - Requires helper functions from `chunking_utilities` (semantically_chunk_documents, chunks_to_df, cluster_chunks_df).
    - The persona generation step executes only if `usecase` is provided.
    - The function dynamically balances generation volume across five QA types to achieve variety and coverage.
    """
    os.makedirs(save_path_dir, exist_ok=True)
    loader = DirectoryLoader(data_path, glob = '*.pdf', loader_cls = PyPDFLoader)
    docs = loader.load()
    
    chunks = chunking_utilities.semantically_chunk_documents(
        docs,                                  # same input as before
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # small, fast model
        min_tokens=80,                         # prevent overly small chunks
        max_tokens=350,                        # keep chunks within retriever budget
        similarity_threshold=0.6,              # cohesion control; higher = stricter
        overlap_sentences=1,                   # carry 1 sentence into next chunk
    )

    dfc = chunking_utilities.chunks_to_df(chunks)

    dfc['chunk_id'] = dfc.index


    dfc1 = chunking_utilities.cluster_chunks_df(dfc)
    dfc1.to_csv(save_path_dir+'/'+'chunked_dataset.csv', index=False)
    num_t = 20 ### number of cohorst of generation = 15 personas + 4 edge cases = 19 ~= 20
    for iterationi in tqdm(range(5), desc='Generating test data'):
        if iterationi==0:
            df1 = get_QA_basic(dfc1.copy(), n=np.ceil(n_queries/num_t).astype(int))
        elif iterationi==1:
            df2 = get_QA_chunk_length(dfc1.copy(), n=np.ceil(n_queries/num_t).astype(int))
        elif iterationi==2:
            df3 = get_QA_chunk_boundary(dfc1.copy(), n=np.ceil(n_queries/num_t).astype(int))
        elif iterationi==3:
            df4 = get_QA_query_intent(dfc1.copy(), n=np.ceil(n_queries/num_t).astype(int))
        elif usecase:
            business_use_case = usecase
            df_personas = get_personas(business_use_case)
            df_personas.to_csv(save_path_dir+'/'+'personas_generated.csv', index=False)
            df5 = get_QA_personas(dfc1.copy(), df_personas.copy(), n=np.ceil(n_queries/num_t).astype(int))

    df1['Rationale'] = 'None'
    df1['check_metric'] = 'general'
    df2['check_metric'] = 'chunk_length'
    df3['check_metric'] = 'chunk_boundary'
    df4['check_metric'] = 'user_intent'
    df5['Rationale'] = 'None'
    
    df1['Less Relevant Chunk IDs'] = np.nan
    df3['Less Relevant Chunk IDs'] = np.nan
    df4['Less Relevant Chunk IDs'] = np.nan
    df5['Less Relevant Chunk IDs'] = np.nan
    
    df1 = df1[['Question', 'Answer', 'Chunk IDs', 'Less Relevant Chunk IDs', 'Difficulty', 'Rationale', 'check_metric']]
    df2 = df2[['Question', 'Answer', 'Chunk IDs', 'Less Relevant Chunk IDs', 'Difficulty', 'Rationale', 'check_metric']]
    df3 = df3[['Question', 'Answer', 'Chunk IDs', 'Less Relevant Chunk IDs', 'Difficulty', 'Rationale', 'check_metric']]
    df4 = df4[['Question', 'Answer', 'Chunk IDs', 'Less Relevant Chunk IDs', 'Difficulty', 'Rationale', 'check_metric']]
    df5 = df5[['Question', 'Answer', 'Chunk IDs', 'Less Relevant Chunk IDs', 'Difficulty', 'Rationale', 'check_metric']]

    df_test = pd.concat([df1, df2, df3, df4, df5])

    df_test = df_test.reset_index(drop=True)

    df_test.to_csv(save_path_dir+'/'+'test_data_generated.csv', index=False)
    return "Created and Saved successfully"

# generate_and_save(data_path=r'../data1/',
#                 usecase='chatbot to guide data scientists on how to test their AI Agents',
#                 save_path='test_dataset',
#                 n_queries=100)
