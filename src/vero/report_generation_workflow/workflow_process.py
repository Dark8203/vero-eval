import os
import json
import pandas as pd
from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph, END
from langchain_openai import ChatOpenAI
from .agent_prompts import *
from langchain_community.document_loaders import CSVLoader

OPENAI_API = os.getenv('OPENAI_API_KEY')

print('Report Starting')
def merge_dicts(dict1, dict2):
    return {**dict1, **dict2}

class State(TypedDict):
    data: Annotated[dict,merge_dicts]


class ReportGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(
                model='o4-mini',
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key= OPENAI_API
            )

    def generate_report(self, pipe_config_data: str | None = None, generation_scores_path: str | None = None, retrieval_scores_path: str | None = None, reranker_scores_path: str | None = None):

        initial_state: State = {'data': {}}

        with open(pipe_config_data, 'r') as json_file:
            pipeline_configuration = json.load(json_file)
        with open('vero/report_generation_workflow/report_data/metrics_definition.json', 'r') as json_file:
            metrics_definition = json.load(json_file)

        loader = CSVLoader(file_path=retrieval_scores_path)
        docs = loader.load()
        retriever_evaluation_results = "\n\n".join([doc.page_content for doc in docs])
        loader = CSVLoader(file_path=generation_scores_path)
        docs = loader.load()
        generation_evaluation_results = "\n\n".join([doc.page_content for doc in docs])
        loader = CSVLoader(file_path=reranker_scores_path)
        docs = loader.load()
        reranker_evaluation_results = "\n\n".join([doc.page_content for doc in docs])

        #TODO: remove data parser agent

        def data_parser(state: State):
            prompt = PromptTemplate.from_template(parser_prompt)

            # pipeline_configuration = pd.read_csv('PipelineConfiguration.csv')
            # metrics_definition = pd.read_csv('MetricsDefinition.csv')

            # evaluation_results = pd.read_csv('EvaluationResults.csv')
            evaluation_results = {'Generation_Evaluation_Results':generation_evaluation_results,'Reranked_Evaluation_Results':reranker_evaluation_results,'Retrieval_Evaluation_Results':retriever_evaluation_results}

            # data = pd.read_csv('Scores_with_remarks.csv')
            messages = prompt.invoke({'pipeline_configuration':pipeline_configuration,'metrics_definitions':metrics_definition,'evaluation_results':evaluation_results})

            result = self.llm.invoke(messages)
            # print(result)
            return {'data': {'parser_agent':result.content}}

        #TODO:make specialised heuristics and review definitions for stress test; Added mean to retriever and reranker metrics

        def retriever_analysis(state: State):
            prompt = PromptTemplate.from_template(retriever_analyst_prompt)

            # retriever_evaluation_results = json.loads(state['data']['parser_agent'])['retriever_evaluation_results']


            with open('vero/report_generation_workflow/report_data/retriever_heuristics.json', 'r') as json_file:
                retriever_heuristics = json.load(json_file)

            messages = prompt.invoke({'retriever_evaluation_results':retriever_evaluation_results,'metrics_definitions':metrics_definition,'heuristics':retriever_heuristics})

            result = self.llm.invoke(messages)
            # print(result.content)
            return {'data': {'retriever_analyst_agent':result.content}}


        def generation_analysis(state: State):
            prompt = PromptTemplate.from_template(generation_analyst_prompt)

            # generation_evaluation_results = json.loads(state['data']['parser_agent'])['generation_evaluation_results']
            # generation_evaluation_results = pd.read_csv('Scores_with_remarks.csv')


            with open('vero/report_generation_workflow/report_data/generator_heuristics.json', 'r') as json_file:
                generator_heuristics = json.load(json_file)

            messages = prompt.invoke(
                {'generation_evaluation_results': generation_evaluation_results, 'metrics_definitions': metrics_definition['evaluation_metrics']['generation_metrics'],'heuristics':generator_heuristics})
            result = self.llm.invoke(messages)
            # print(result.content)
            return {'data': {'generation_analyst_agent':result.content}}


        def reranker_analysis(state: State):
            prompt = PromptTemplate.from_template(reranker_analyst_prompt)

            # reranker_evaluation_results = json.loads(state['data']['parser_agent'])['reranker_evaluation_results']


            with open('vero/report_generation_workflow/report_data/retriever_heuristics.json', 'r') as json_file:
                reranker_heuristics = json.load(json_file)

            messages = prompt.invoke(
                {'reranker_evaluation_results': reranker_evaluation_results, 'metrics_definitions': metrics_definition,'heuristics':reranker_heuristics})

            result = self.llm.invoke(messages)
            # print(result.content)
            return {'data': {'reranker_analyst_agent':result.content}}



        def synthesis_diagnosis(state: State):
            prompt = PromptTemplate.from_template(diagnosis_prompt)

            evaluation_context = json.loads(state['data']['parser_agent'])
            retriever_analysis = json.loads(state['data']['retriever_analyst_agent'])
            generation_analysis = json.loads(state['data']['generation_analyst_agent'])
            reranker_analysis = json.loads(state['data']['reranker_analyst_agent'])
            messages = prompt.invoke({'evaluation_context':evaluation_context,
                                      'retriever_analysis':retriever_analysis,
                                      'generator_analysis':generation_analysis,
                                      'reranker_analysis':reranker_analysis})

            result = self.llm.invoke(messages)
            return {'data': {'diagnosis_agent':result.content}}



        def strategist_recommender(state: State):
            prompt = PromptTemplate.from_template(recommender_prompt)

            evaluation_context = json.loads(state['data']['parser_agent'])
            diagnosis_context = json.loads(state['data']['diagnosis_agent'])
            messages = prompt.invoke({'evaluation_context': evaluation_context,
                                      'diagnosis_context': diagnosis_context})

            result = self.llm.invoke(messages)
            return {'data': {'recommender_agent':result.content}}


        def report_generation(state: State):
            prompt = PromptTemplate.from_template(report_generation_prompt)
            messages = prompt.invoke({'data':state['data']})
            result = self.llm.invoke(messages)
            # print(result.content)
            return {'data': {'report_generation_agent':result.content}}


        def report_reviewer(state: State):
            prompt = PromptTemplate.from_template(report_review_prompt)

            filtered_data = state['data'].copy()
            filtered_data.pop('report_generation_agent',None)
            messages = prompt.invoke({'data':filtered_data,'generated_report':state['data']['report_generation_agent']})

            result = self.llm.invoke(messages)#feedback flag

            return {'data': {'reviewer_agent':result.content}}


        #functions for triggering conditional edges
        def parallel_executing_agents(state: State):
            return ['generation_analyst_agent','reranker_analyst_agent','retriever_analyst_agent']

        def feedback_flag(state: State):
            # flag = state['data']['reviewer_agent']['regeneration_needed']
            flag = json.loads(state['data']['reviewer_agent'])['regeneration_needed']
            if flag:
                return 'report_generation_agent'
            else:
                return END


        graph_builder = StateGraph(State)
        graph_builder.add_node('parser_agent',data_parser)
        graph_builder.add_node('generation_analyst_agent',generation_analysis)
        graph_builder.add_node('reranker_analyst_agent',reranker_analysis)
        graph_builder.add_node('retriever_analyst_agent',retriever_analysis)
        graph_builder.add_node('diagnosis_agent',synthesis_diagnosis,)
        graph_builder.add_node('recommender_agent',strategist_recommender,)
        graph_builder.add_node('report_generation_agent',report_generation,)
        graph_builder.add_node('reviewer_agent',report_reviewer,)

        # graph_builder.add_conditional_edges(START, parallel_executing_agents)
        graph_builder.add_edge(START, 'parser_agent')
        graph_builder.add_conditional_edges('parser_agent',parallel_executing_agents)
        graph_builder.add_edge('generation_analyst_agent','diagnosis_agent')
        graph_builder.add_edge('reranker_analyst_agent','diagnosis_agent')
        graph_builder.add_edge('retriever_analyst_agent','diagnosis_agent')
        graph_builder.add_edge('diagnosis_agent','recommender_agent')
        graph_builder.add_edge('recommender_agent','report_generation_agent')
        graph_builder.add_edge('report_generation_agent','reviewer_agent')

        graph_builder.add_conditional_edges('reviewer_agent',feedback_flag)


        graph = graph_builder.compile()
        result = graph.invoke(initial_state)
        print(result)

        with open('report.md','w') as f:
            f.write(result['data']['report_generation_agent'])
            f.close()


