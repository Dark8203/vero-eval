---
id: evaluator
title: Evaluator
---

# Evaluator

## **Overview**
- The Evaluator is a convenience wrapper to run multiple metrics over model outputs and retrieval results.
- It orchestrates generation evaluation (text-generation metrics), retrieval evaluation (precision/recall/sufficiency), and reranker evaluation (NDCG, MAP, MRR). It produces CSV summaries by default.

> Quick notes
> - The Evaluator uses project metric classes (e.g., BartScore, BertScore, RougeScore, SemScore, PrecisionScore, RecallScore, MeanAP, MeanRR, RerankerNDCG, CumulativeNDCG, etc.). These metrics are in vero.metrics and are referenced internally.
> - Many methods expect particular CSV column names (see "Expected CSV schemas").

### Steps to evaluate your pipeline

**Step 1 - Generation evaluation** 
- Input: a CSV with "Context Retrieved" and "Answer" columns.
- Result: Generation_Scores.csv with columns such as SemScore, BertScore, RougeLScore, BARTScore, BLUERTScore, G-Eval (Faithfulness).

Example:
```py
from vero.evaluator import Evaluator

evaluator = Evaluator()
# data_path must point to a CSV with columns "Context Retrieved" and "Answer"
df_scores = evaluator.evaluate_generation(data_path='testing.csv')
print(df_scores.head())
```

**Step 2 - Preparing reranker inputs (parse ground truth + retriever output)**
- Use parse_retriever_data to convert ground-truth chunk ids and retriever outputs into a ranked_chunks_data.csv suitable for reranker evaluation.

Example:
```py
from vero.evaluator import Evaluator

evaluator = Evaluator()
# ground_truth_path: dataset with 'Chunk IDs' and 'Less Relevant Chunk IDs' columns
# data_path: retriever output with 'Context Retrieved' containing "id='...'"
evaluator.parse_retriever_data(
    ground_truth_path='test_dataset_generator.csv',
    data_path='testing.csv'
)
# This will produce 'ranked_chunks_data.csv'
```


**Step 3 - Retrieval evaluation (precision, recall, sufficiency)**
- Inputs:
  - retriever_data_path: a CSV that contains 'Retrieved Chunk IDs' and 'True Chunk IDs' columns (lists or strings).
  - data_path: the generation CSV with 'Context Retrieved' and 'Question' (for sufficiency).
- Result: Retrieval_Scores.csv

Example:
```py
from vero.evaluator import Evaluator

evaluator = Evaluator()
df_retrieval_scores = evaluator.evaluate_retrieval(
    data_path='testing.csv',
    retriever_data_path='ranked_chunks_data.csv'
)
print(df_retrieval_scores.head())
```

**Step 4 - Reranker evaluation (MAP, MRR, NDCG)**
Example:
```py
from vero.evaluator import Evaluator

evaluator = Evaluator()
df_reranker_scores = evaluator.evaluate_reranker(
    ground_truth_path='test_dataset_generator.csv',
    retriever_data_path='ranked_chunks_data.csv'
)
print(df_reranker_scores)
```


#### Lower-level metric usage
To run a single metric directly you can instantiate the metric class. For example, to compute BARTScore or BertScore per pair:
```py
from vero.metrics import BartScore, BertScore

with BartScore() as bs:
    bart_results = [bs.evaluate(context, answer) for context, answer in zip(contexts, answers)]

with BertScore() as bert:
    bert_results = [bert.evaluate(context, answer) for context, answer in zip(contexts, answers)]
```

