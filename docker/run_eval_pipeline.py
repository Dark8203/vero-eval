from vero.evaluator import Evaluator
import torch
import yaml
import os

config_path = os.getenv('VERO_CONFIG_PATH', '../vero-deploy/config.yaml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

if torch.cuda.is_available():
    device = 'cuda'
    print(device)

evaluator = Evaluator()

# data_path must point to a CSV with columns "Context Retrieved" and "Answer"
df_scores = evaluator.evaluate_generation(data_path=config['input_file_path'])
df_scores.to_csv(config['generation_output_file_path'])

# ground_truth_path: dataset with 'Chunk IDs' and 'Less Relevant Chunk IDs' columns
# data_path: retriever output with 'Context Retrieved' containing "id='...'"
evaluator.parse_retriever_data(
    ground_truth_path=config['ground_truth_file_path'],
    data_path=config['input_file_path']
)
# This will produce 'ranked_chunks_data.csv'


df_retrieval_scores = evaluator.evaluate_retrieval(
    data_path=config['input_file_path'],
    retriever_data_path='ranked_chunks_data.csv'
)
df_retrieval_scores.to_csv(config['retriever_output_file_path'])



df_reranker_scores = evaluator.evaluate_reranker(
    ground_truth_path=config['ground_truth_file_path'],
    retriever_data_path='ranked_chunks_data.csv'
)
df_reranker_scores.to_csv(config['reranker_output_file_path'])