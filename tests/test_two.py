from vero.evaluator import Evaluator
import torch


if torch.cuda.is_available():
    device = 'cuda'
    print(device)

evaluator = Evaluator()

# data_path must point to a CSV with columns "Context Retrieved" and "Answer"
df_scores = evaluator.evaluate_generation(data_path='tests/testing.csv')
print(df_scores.head())




# ground_truth_path: dataset with 'Chunk IDs' and 'Less Relevant Chunk IDs' columns
# data_path: retriever output with 'Context Retrieved' containing "id='...'"
evaluator.parse_retriever_data(
    ground_truth_path='test_dataset_generator.csv',
    data_path='testing.csv'
)
# This will produce 'ranked_chunks_data.csv'


df_retrieval_scores = evaluator.evaluate_retrieval(
    data_path='tests/testing.csv',
    retriever_data_path='tests/ranked_data.csv'
)
print(df_retrieval_scores.head())



df_reranker_scores = evaluator.evaluate_reranker(
    ground_truth_path='test_dataset_generator.csv',
    retriever_data_path='ranked_chunks_data.csv'
)
print(df_reranker_scores)




from vero.metrics import SemScore, BertScore
from tqdm import tqdm
import re
import pandas as pd

def extract_page_content(text_blob):
    pattern = r"page_content='(.*?)'"
    matches = re.findall(pattern, text_blob)
    return matches
df_new = pd.read_csv(r'tests/testing.csv')
chunks_list = df_new['Context Retrieved'].apply(extract_page_content).tolist()
answers_list = df_new['Answer'].tolist()

print("Processing SemScore...")
with SemScore() as sem_score:
    sem_results = [sem_score.evaluate(chunk, ans) for chunk, ans in tqdm(zip(chunks_list, answers_list), total=len(df_new))]
print(sem_results)


print("\nProcessing BERT Score...")
bs = BertScore()
with BertScore() as bs:
    bert_results = [bs.evaluate(chunk, ans) for chunk, ans in tqdm(zip(chunks_list, answers_list), total=len(df_new))]
bert_dicts = [{'Precision': p, 'Recall': r, 'F1score': f} for p, r, f in bert_results]
print(bert_dicts)


from vero.report_generation_workflow import ReportGenerator

# Initialize the report generator
report_generator = ReportGenerator()

# Generate the final report by providing:
# - Pipeline configuration JSON file
# - Generation, Retrieval, and Reranker evaluation CSV files
report_generator.generate_report(
    'tests/pipe_config_data.json',
    'tests/Generation_Scores.csv',
    'tests/Retrieval_Scores.csv',
    'tests/Reranked_Scores.csv'
)


from vero.test_dataset_generator import generate_and_save

# Generate 100 queries from PDFs stored in ./data/pdfs directory and save outputs in test_dataset directory
generate_and_save(
    data_path='tests/data/',
    usecase='Vitamin chatbot catering to general users for their daily queries',
    save_path_dir='test_dataset',
    n_queries=20
)