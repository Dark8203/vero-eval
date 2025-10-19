from vero.metrics import MetricBase
import gc
import os
import numpy as np
from typing import Optional
from openai import OpenAI
import torch
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# from tracing_components import logger


# Check if a GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

#TODO:expose openai client for model config
class GEvalScore(MetricBase):
    '''
        A unique implementation of g-eval.

        :param api_key: Pass the api key for OpenAI client.

        Methods
        ---------
        1. __init__(api_key)
            Initializes the client.

        2. evaluate() -> float
            Returns the g-eval score.
        '''
    name = 'g_eval_score'

    def __init__(self,api_key:str | None = None):
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)

    def __enter__(self):
        return self

    # tiktoken for tokens, extracting logits
    def evaluate(self,reference, candidate, metric, metric_description:str =None, rubric :str =None, custom_prompt: Optional[str] = None, polling: bool = False, polling_calls: int = 5) -> float | None:
        '''
        :param reference: Pass the reference chunks.
        :param candidate: Pass the answer.
        :param metric: Pass the metric.
        :param metric_description: Pass the metric description. (Optional)
        :param rubric: Pass the rubric. (Optional)
        :param custom_prompt: Pass the custom prompt. (Optional)
        :param polling: Pass if polling is enabled. (Optional)
        :param polling_calls: Pass the number of polling calls. (Optional)

        :return: float
        '''
        client = self.client
        if rubric is None:
            rubric = '1 to 5; 1 being worst and 5 being best'
        score = 0
        pc = polling_calls
        log_list = {}
        g_eval_score = 0
        logits = {32: 10, 33: 10, 34: 10, 35: 10, 36: 10}

        meta_prompt = f'''
                You are an expert AI prompt engineer specializing in creating evaluation prompts for Large Language Models (LLMs) based on the G-Eval framework. Your task is to generate a comprehensive and structured evaluation prompt template based on user-provided requirements.

                Instructions:
                    Role-Play: The generated prompt must begin by assigning a clear role to the evaluator LLM (e.g., "You are an expert evaluator.").

                    Task Definition: Clearly define the evaluation task using the user's Metric Name and Metric Description (description is optional).

                    Context Placeholders:

                    Always include placeholders(give curly braces so placeholders can be filled) for Retrieved Context (i.e. ref) and Generated Output (i.e. candidate).

                    Evaluation Criteria: Integrate the user's Rating Scale. Provide a detailed definition for what each level of the scale represents.

                    Reasoning Requirement: MANDATE a step-by-step reasoning process (Chain-of-Thought) to evaluate and provide the final score.

                    Structured Output: Instruct the LLM to provide its final output in a clean, easily parsable form and the output should ONLY include the scale provided by the user and nothing else.

                These are the requirements given by the user:
                Metric - {metric}
                Metric Description - {metric_description}
                Rating rubric - {rubric}      

                **Crucial** only include the prompt don't include something like:'Sure, Here is the prompt for this' or 'Here's a comprehensive and structured evaluation prompt template based on the provided requirements:' or 'Fill in the placeholders ref and candidate with the appropriate context and generated output to utilize this prompt effectively.', etc
                Also dont include final score in curly braces as it will become a placeholder and throw error when formatting.
                **Crucial** include this line in the generated prompt - 'Your final output must be ONLY this score. Do not output any reasoning, explanations, introductory text, quotation marks, asterisks, or any other characters. The entire response must be the raw score value.'
            '''

        try:
            if custom_prompt == None:
                prompt_call = client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[{'role': 'user', 'content': meta_prompt}],
                )
                prompt = prompt_call.choices[0].message.content
                prompt = prompt.format(ref=reference, candidate=candidate)
                print(prompt)
            else:
                prompt = custom_prompt

            while True:
                completion = client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[{'role': 'user', 'content': prompt}],
                    logprobs=True,
                    top_logprobs=10,
                    logit_bias=logits,

                )
                if polling:
                    score += int(completion.choices[0].message.content.strip())
                    if polling_calls > 1:
                        polling_calls -= 1
                        continue
                    else:
                        break
                else:
                    break

            if polling:
                g_eval_score = score / pc
                return g_eval_score
            else:
                top_two_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
                for logprob in top_two_logprobs:
                    num = logprob.token.strip()
                    if num in ['1', '2', '3', '4', '5']:
                        log_score = np.round(np.exp(logprob.logprob), 4)
                        score += log_score
                        log_list[int(num)] = log_score

                if score == 0:
                    print('Error in calculating g-eval score')
                    #logger.error('Error in generating g-eval score')
                else:
                    for i, j in log_list.items():
                        g_eval_score += i * float((j / score))
                    return g_eval_score

        except Exception as e:
            # logger.error('Error in generating g-eval score\nError', e)
            return None


    def __exit__(self, exc_type, exc_value, traceback):
        del self.client
        self.client = None
        gc.collect()
        torch.cuda.empty_cache()