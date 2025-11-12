---
id: test-dataset-generation
title: Test Dataset Generation
---

# Test Dataset Generation

## **Overview**

- The Test Dataset Generation module creates high-quality question-answer pairs derived from your document collection. It generates challenging queries designed to reveal retrieval and reasoning failures in RAG systems (e.g., boundary synthesis, chunk-length bias, intent understanding).
- Internally it chunks documents, clusters related chunks, and uses an LLM to produce QA items with ground-truth chunk IDs and metadata.

### When to use

- To create evaluation sets for retriever, reranker, and generation metrics.
- To stress-test retrieval behavior (chunk-length bias, boundary content) and query understanding.

### Core function

- `generate_and_save(data_path, usecase, save_path_dir='test_dataset', n_queries=100)`

### Parameters

- `data_path (str)`: Path to a directory containing source PDFs. The generator uses a PDF loader to read documents.
- `usecase (str)` : Description of the dataset/use-case.
- `save_path_dir (str)`: Output directory path (default: 'test_dataset').
- `n_queries (int)`: Number of QA queries to generate (min 100). The generator groups queries across several prompt styles.

### Return

- Writes test dataset generated and relevant files to `save_path_dir` and returns a short success message.

### **Example**

```py
from vero.test_dataset_generator import generate_and_save

# Generate 40 queries from PDFs stored in ./data/pdfs and save as test_dataset.csv
generate_and_save(
    data_path='./data/pdfs',
    usecase='Vitamin chatbot catering to general users for their daily queries',
    save_path_dir='test_dataset',
    n_queries=100
)
```

> **Using the dataset with the Evaluator**
>
> The Evaluator expects a ground truth CSV with chunk IDs when running reranker / retrieval evaluations. The produced CSV is directly usable as `ground_truth_path` in `Evaluator.parse_retriever_data` or `evaluate_reranker`.

### Practical tips

- Recommended `n_queries`: 100–500 depending on evaluation budget.
- Reproducibility: You can re-run with the same document set; consider saving intermediate chunking outputs if you need stable chunk IDs across runs.
