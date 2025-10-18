# from src.vero.metrics import RecallScore, PrecisionScore, SufficiencyScore, CitationScore, MeanRR, MeanAP, RerankerNDCG, CumulativeNDCG
# from src.vero.metrics import OverlapScore, BertScore, RougeScore, SemScore, BartScore, BleurtScore, AlignScore, GEvalScore, NumericalHallucinationScore
#
#
# ch_r=[1,2,3,5,6]
# ch_t=[2,3,4]
# rs = RecallScore(ch_r,ch_t)
# ps = PrecisionScore(ch_r,ch_t)
# ss = SufficiencyScore(ch_r,ch_t)
# cs = CitationScore(ch_r,ch_t)
# print(rs.evaluate())
# print(ps.evaluate())
# print(ss.evaluate())
# print(cs.evaluate())
# #
# rr = [[1,2,3,5,6],[1,2,3,5,6]]
# gt = [[2,3,6],[2,3,6]]
# ranks = [{2: 2, 3: 2},{2: 2, 3: 2, 6: 1}]
# mrr = MeanRR(rr,gt)
# map = MeanAP(rr,gt)
# rndcg = RerankerNDCG(rr,ranks)
# cndcg = CumulativeNDCG(rr,ranks)
# print(mrr.evaluate())
# print(map.evaluate())
# print(rndcg.evaluate())
# print(cndcg.evaluate())
#
# kt = ['vitamins','neurotoxin']
# ans = 'You are advised to increase the intake of vitamins to counter the neurotoxin that you have ingested.'
# os = OverlapScore(ans,kt)
# print(os.evaluate())

#
# import re
# import pandas as pd
# from tqdm import tqdm
# import torch
#
# def extract_page_content(text_blob):
#     pattern = r"page_content='(.*?)'"
#     matches = re.findall(pattern, text_blob)
#     return matches
# df_new = pd.read_csv(r'testing.csv')
# chunks_list = df_new['Context'].apply(extract_page_content).tolist()
# answers_list = df_new['Answer'].tolist()

# print("Processing SemScore...")
# with SemScore() as sem_score:
#     sem_results = [sem_score.evaluate(chunk, ans) for chunk, ans in tqdm(zip(chunks_list, answers_list), total=len(df_new))]
# # print(sem_results)
#
#
# print("\nProcessing BERT Score...")
# bs = BertScore()
# with BertScore() as bs:
#     bert_results = [bs.evaluate(chunk, ans) for chunk, ans in tqdm(zip(chunks_list, answers_list), total=len(df_new))]
# bert_dicts = [{'Precision': p, 'Recall': r, 'F1score': f} for p, r, f in bert_results]
# print(bert_dicts)
#
#
# print("\nProcessing RougeL Score...")
# with RougeScore() as rouge:
#     rouge_results = [rouge.evaluate(chunk, ans) for chunk, ans in tqdm(zip(chunks_list, answers_list), total=len(df_new))]
# rouge_dicts = [{'Precision': p, 'Recall': r, 'F1score': f} for p, r, f in rouge_results]
# print(rouge_dicts)
#
#
#
# print('BartScore')
# with BartScore() as bart_score:
#     score = [bart_score.evaluate(chunk, ans) for chunk, ans in tqdm(zip(chunks_list, answers_list), total=len(df_new))]
# print(score)
#
# print("Processing BLUERTScore...")
# with BleurtScore() as bleurt:
#     bl_results = [bleurt.evaluate(chunk, ans) for chunk, ans in tqdm(zip(chunks_list, answers_list), total=len(df_new))]
# print(bl_results)

#TODO: Figure this out

# allocated = torch.cuda.memory_allocated(device=0)
# print(allocated)
# print("Processing AlignScore...")
# with AlignScore() as align:
#     al_results = [align.evaluate(chunk, ans) for chunk, ans in tqdm(zip(chunks_list, answers_list), total=len(df_new))]
# print(al_results)
# allocated = torch.cuda.memory_allocated(device=0)
# print(allocated)

#
# print("\nProcessing G-Eval...")
# with GEvalScore(api_key) as g_eval:
#     g_eval_results = [g_eval.evaluate(chunk,ans, metric='Faithfulness') for chunk, ans in tqdm(zip(chunks_list, answers_list), total=len(df_new))]
# print(g_eval_results)

from src.vero import Evaluator
import pandas as pd

evaluator = Evaluator()

evaluator.evaluate_generation('', r'testing.csv')
