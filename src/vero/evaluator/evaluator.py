from vero.metrics import *
from tqdm import tqdm
import pandas as pd
import ast
import re
import math
from typing import Any

METRICS_REGISTRY = {
    cls.name: cls for cls in [
        SufficiencyScore,
        SemScore,
        BleurtScore,
        BartScore,
        BertScore,
        AlignScore,
        CitationScore,
        CumulativeNDCG,
        OverlapScore,
        GEvalScore,
        MeanRR,
        MeanAP,
        RerankerNDCG,
        NumericalHallucinationScore,
        PrecisionScore,
        RecallScore,
        RougeScore,
    ]
}

MODEL_METRICS = {
    'align_score',
    'bart_score',
    'bert_score',
    'bleurt_score',
    'rouge_score',
    'sem_score',
}

MATH_METRICS = {
    'citation_score',
    'cumulative_ndcg',
    'overlap_score',
    'precision_score',
    'recall_score',
    'reranker_ndcg',
    'numerical_hallucination_score',
    'mean_average_precision',
    'mean_reciprocal_rank',

}

LLM_METRICS = {
    'g_eval_score',
    'sufficiency_score',
}


# eval = Evaluator()
# eval.evaluate(data)
# with BertScore() as bs:
#     bert_results = [bs.evaluate(chunk, ans) for chunk, ans in tqdm(zip(chunks_list, answers_list), total=len(df_new))]
# bert_dicts = [{'Precision': p, 'Recall': r, 'F1score': f} for p, r, f in bert_results]
# print(bert_dicts)

class Evaluator:
    name = 'evaluator'

    def __init__(self, metrics: list[str] | None = None):
        pass
        # if metrics is None:
        #     metrics_to_run = list(METRICS_REGISTRY.keys())
        # else:
        #     metrics_to_run = metrics
        #
        # self.evaluators = []
        # for metric_name in metrics_to_run:
        #     if metric_name not in METRICS_REGISTRY:
        #         raise ValueError(f"Metric '{metric_name}' not supported currently.")
        #
        #     evaluator_class = METRICS_REGISTRY[metric_name]
        #     self.evaluators.append(evaluator_class())

    # TODO: data parsing first then continue with this
    def evaluate_math(self, reference_list, answers_list, metrics: list[str] | None = None):
        pass

    #TODO: include evaluation with ground truth answers also
    #ground truth path is placeholder for now
    def evaluate_generation(self, ground_truth_path: str | None = None, data_path: str | None = None):
        if data_path is None:
            raise ValueError("Data path must be provided for generation evaluation.")

        def extract_page_content(text_blob):
            pattern = r"page_content='(.*?)'"
            matches = re.findall(pattern, text_blob)
            return matches

        df = pd.read_csv(data_path)
        reference_list = df['Context Retrieved'].apply(extract_page_content).tolist()
        answers_list = df['Answer'].tolist()

        score_df = pd.DataFrame()

        with SemScore() as sem_score:
            sem_results = [sem_score.evaluate(chunk, ans) for chunk, ans in
                           tqdm(zip(reference_list, answers_list), total=len(df))]
        score_df['SemScore'] = sem_results

        with BertScore() as bs:
            bert_results = [bs.evaluate(chunk, ans) for chunk, ans in
                            tqdm(zip(reference_list, answers_list), total=len(df))]
        bert_dicts = [{'Precision': p, 'Recall': r, 'F1score': f} for p, r, f in bert_results]
        score_df['BertScore'] = bert_dicts

        with RougeScore() as rouge:
            rouge_results = [rouge.evaluate(chunk, ans) for chunk, ans in
                             tqdm(zip(reference_list, answers_list), total=len(df))]
        rouge_dicts = [{'Precision': p, 'Recall': r, 'F1score': f} for p, r, f in rouge_results]
        score_df['RougeLScore'] = rouge_dicts

        with BartScore() as bart_score:
            bart_score = [bart_score.evaluate(chunk, ans) for chunk, ans in
                     tqdm(zip(reference_list, answers_list), total=len(df))]
        score_df['BARTScore'] = bart_score

        with BleurtScore() as bleurt:
            bl_results = [bleurt.evaluate(chunk, ans) for chunk, ans in
                          tqdm(zip(reference_list, answers_list), total=len(df))]
        score_df['BLUERTScore'] = bl_results

        print("\nProcessing G-Eval...")
        with GEvalScore() as g_eval:
            g_eval_results = [g_eval.evaluate(chunk, ans, metric='Faithfulness') for chunk, ans in
                              tqdm(zip(reference_list, answers_list), total=len(df))]
        score_df['G-Eval (Faithfulness)'] = g_eval_results


        print(score_df.head())
        score_df.to_csv('Generation_Scores.csv')


        return score_df

    def parse_retriever_data(self,ground_truth_path:str, data_path: str):
        def extract_page_content(text_blob):
            pattern = r"id='(.*?)'"
            matches = re.findall(pattern, text_blob)
            match = [int(x) for x in matches]
            return match

        def clean_ids(value):
            if pd.isna(value):
                return []

            if isinstance(value, (int, float)):
                return [int(value)]

            if isinstance(value, str):
                ids = [int(s.strip()) for s in value.split(',')]
                return ids

        df = pd.read_csv(ground_truth_path)
        df['Chunk IDs'] = df['Chunk IDs'].apply(clean_ids)
        df['Less Relevant Chunk IDs'] = df['Less Relevant Chunk IDs'].apply(clean_ids)

        df['all_chunk_ids'] = df['Chunk IDs'] + df['Less Relevant Chunk IDs']


        df_new = pd.read_csv(data_path)
        chunks_list = df_new['Context Retrieved'].apply(extract_page_content).tolist()

        df_new_2 = pd.DataFrame(columns=['Retrieved Chunk IDs','True Chunk IDs'])
        df_new_2['Retrieved Chunk IDs'] = chunks_list
        df_new_2['True Chunk IDs'] = df['all_chunk_ids']
        df_new_2['True Chunk IDs'] = df_new_2['True Chunk IDs'].apply(lambda x: None if len(x) == 0 else x)
        df_new_2.dropna(subset=['True Chunk IDs'], inplace=True)

        df['Chunk IDs'] = df['Chunk IDs'].apply(lambda x: None if len(x) == 0 else x)
        # df['Less Relevant Chunk IDs'] = df['Less Relevant Chunk IDs'].apply(lambda x: None if len(x) == 0 else x)
        df.dropna(subset=['Chunk IDs'], inplace=True)
        df_new_2['Ranked True Chunk IDs'] = [{key:2 for key in chunk_id_list} for chunk_id_list in df['Chunk IDs'].tolist()]
        df_new_2['Ranked Less Relevant Chunk IDs'] = df['Less Relevant Chunk IDs'].apply(lambda x :{key:1 for key in x})
        df_new_2['Ranked All Chunk IDs'] = df_new_2.apply(lambda row: {**row['Ranked True Chunk IDs'], **row['Ranked Less Relevant Chunk IDs']},
            axis=1
        )

        df_new_2.to_csv('ranked_chunks_data.csv', index=False)

    #here also ground truth path is placeholder for now
    def evaluate_reranker(self, ground_truth_path: str | None = None, retriever_data_path: str | None = None):
        if retriever_data_path is None:
            raise ValueError("Data path must be provided for reranker evaluation.")

        df = pd.read_csv(retriever_data_path)
        ret_chunks = df['Retrieved Chunk IDs'].apply(ast.literal_eval).tolist()
        true_chunks = df['True Chunk IDs'].apply(ast.literal_eval).tolist()
        true_ranked_chunks = df['Ranked All Chunk IDs'].apply(ast.literal_eval).tolist()

        print(ret_chunks, true_chunks, true_ranked_chunks)

        mean_ap = MeanAP(ret_chunks, true_chunks)
        mean_ap_result = mean_ap.evaluate()

        print(mean_ap_result)

        mean_rr = MeanRR(ret_chunks, true_chunks)
        mean_rr_result = mean_rr.evaluate()

        reranker_ndcg = RerankerNDCG(ret_chunks, true_ranked_chunks)
        reranker_ndcg_result = reranker_ndcg.evaluate()
        reranker_ndcg_result_avg = round(sum(reranker_ndcg_result) / len(reranker_ndcg_result), 2)

        cumulative_ndcg = CumulativeNDCG(ret_chunks, true_ranked_chunks)
        cumulative_ndcg_result = cumulative_ndcg.evaluate()
        cumulative_ndcg_result_avg = round(sum(cumulative_ndcg_result) / len(cumulative_ndcg_result), 2)

        score_df = pd.DataFrame(columns=['Mean Average Precision', 'Mean Reciprocal Rank', 'Reranker NDCG', 'Cumulative NDCG'])
        score_df.loc[0] = [mean_ap_result,mean_rr_result, reranker_ndcg_result_avg, cumulative_ndcg_result_avg]

        score_df.to_csv('Reranked_Scores.csv')

        return score_df



    def evaluate_retrieval(self, data_path: str | None = None, retriever_data_path: str | None = None):
        if retriever_data_path is None:
            raise ValueError("Data path must be provided for retrieval evaluation.")

        df = pd.read_csv(retriever_data_path)
        ret_chunks = df['Retrieved Chunk IDs'].tolist()
        true_chunks = df['True Chunk IDs'].tolist()


        retrieval_recall_score = [round(RecallScore(ret_chunk, true_chunk).evaluate(),2) for ret_chunk, true_chunk in
                                  zip(ret_chunks, true_chunks)]


        retrieval_precision_score = [round(PrecisionScore(ret_chunk, true_chunk).evaluate(),2) for ret_chunk, true_chunk in
                                     zip(ret_chunks, true_chunks)]


        def extract_page_content(text_blob):
            pattern = r"page_content='(.*?)'"
            matches = re.findall(pattern, text_blob)
            return matches

        df_new = pd.read_csv(data_path)
        contexts = df_new['Context Retrieved'].apply(extract_page_content).tolist()
        question = df_new['Question'].tolist()

        retrieval_sufficiency_score = [SufficiencyScore(context, ques).evaluate() for context, ques in
                                       tqdm(zip(contexts, question), total=len(df_new))]


        score_df = pd.DataFrame(
            columns=['Recall Score', 'Precision Score', 'Context Sufficiency Score'])
        score_df['Recall Score'] = retrieval_recall_score
        score_df['Precision Score'] = retrieval_precision_score
        score_df['Context Sufficiency Score'] = retrieval_sufficiency_score


        score_df.to_csv('Retrieval_Scores.csv')

        return score_df

    def evaluate(self, reference_list, answers_list, metrics: list[str] | None = None):
        pass
        # for metric in MODEL_METRICS:
        #
        #

        #
        # # TODO: Figure this fuck out
        #
        # print("Processing AlignScore...")
        # with AlignScore() as align:
        #     al_results = [align.evaluate(chunk, ans) for chunk, ans in tqdm(zip(reference_list, answers_list), total=len(df_new))]
        # print(al_results)
        #
        #
        # print("\nProcessing G-Eval...")
        # with GEvalScore() as g_eval:
        #     g_eval_results = [g_eval.evaluate(chunk, ans, metric='Faithfulness') for chunk, ans in
        #                       tqdm(zip(reference_list, answers_list), total=len(df_new))]
        # print(g_eval_results)
