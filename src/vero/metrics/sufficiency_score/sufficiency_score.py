from vero.metrics import MetricBase
import json
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from .prompt import prompt_sufficiency
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

custom_rag_prompt = PromptTemplate.from_template(prompt_sufficiency)

#TODO: still need to refactor for api and model and change import for prompt
class SufficiencyScore(MetricBase):
    '''
        Calculates the sufficiency score of chunks for the user query.

        :param api_key: Pass the api key for OpenAI client.

        Methods
        ---------
        1. __init__(context, question,api_key)
            Accepts the context and question parameters and the api key.
        2. evaluate() -> float
            Returns the sufficiency score.

        :returns: float
        '''
    name:str = 'sufficiency_score'

    def __init__(self, context:list|str, question:list|str, api_key:str=None):
        self.context = context
        self.question = question
        api_key = os.getenv('OPENAI_API_KEY')
        self.api_key = api_key

    def evaluate(self):
        llm = ChatOpenAI(
            model='gpt-4o-mini',
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=self.api_key,
        )
        messages = custom_rag_prompt.invoke({"question": self.question, "context": self.context})
        response = llm.invoke(messages)
        response_json = json.loads(response.content)
        return response_json['score']