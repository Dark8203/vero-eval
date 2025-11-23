# Vero

<p align="center">
  <a href="https://discord.gg/KeJeuzZm">
    <img src="https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white" alt="Join our Discord">
  </a>
</p>

**Vero** is an open platform for evaluating and monitoring AI pipelines with real-world rigor.
<br>
It goes beyond standard benchmarking by understanding your business use-case to generate edge-case **user personas** and **stress-test** your Agent across challenging scenarios, help identify risks and build highly reliable AI systems.

> Most eval tools say "You're broken". Vero tells you where, and how to fix it

> ⭐ If you find this project helpful, consider giving it a star. It genuinely helps the project grow and encourages us to keep improving.

# Index

- [Key Features](#key-features)
- [Flowchart](#flowchart)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Metrics Overview](#metrics-overview)
- [Evaluator](#evaluator)
- [Test Dataset Generation](#test-dataset-generation)
- [Report Generation](#report-generation)
- [Containerised evaluation (Docker)](#containerised-evaluation-docker)

<br>

# Key Features

- **Trace & Log Execution**: Each query runs through the RAG pipeline is logged into an SQLite database, capturing the user query, retrieved context, reranked items, and the model’s output.
- **Component-level Metrics**: Evaluate intermediate pipeline stages using metrics like Precision, Recall, Sufficiency, Citation, Overlap, and Ranking metrics (e.g. MRR, MAP, NDCG).
- **Generation Metrics**: Measure semantic, factual, and alignment quality of generated outputs using metrics such as BERTScore, ROUGE, SEMScore, AlignScore, BLEURT, and G-Eval.
- **Modular & Extensible**: Easily plug in new metric classes or custom scoring logic; the framework is designed to grow with your needs.
- **End-to-End Evaluation**: Combine component metrics to understand the holistic performance of your RAG system — not just individual parts.

# Flowchart

![flowchart(v2).png](docs/flowchart%28v2%29.png)

# Project Structure

```
.
├── src/
|   ├── vero/
│   │   ├── evaluator         # Main package for evaluation
│   │   ├── report generation workflow  # Report generation workflow
│   │   ├── test dataset generator  # Test dataset generator
|   │   ├── metrics/          # Main package for metrics
|   └── └──  all the metrics  # All the metrics are in here
└── tests/
│   └── test_main.py/         # file for all the testing
└── docker/
│   ├── run_eval_pipeline.py  # script to run the evaluation pipeline
│   └── Dockerfile/           # dockerfile for building the image    
└── vero-deploy/
│   ├── data/                 # folder for input and output data
│   │   ├── inputs            # folder for input data
│   │   │   ├── input.csv     # example input csv file
│   │   │   └── ground_truth.csv # example ground truth csv file
│   │   └── outputs           # folder for output data
│   ├── config.yaml           # yaml file for configuration
└── └── docker-compose.yaml   # docker compose file

```

# Getting Started

### Setup

Install via pip (recommended inside a virtualenv):

```py
pip install vero-eval
```

### Example Usage

```py
from vero.rag import SimpleRAGPipeline
from vero.trace import TraceDB
from vero.eval import Evaluator

trace_db = TraceDB(db_path="runs.db")
pipeline = SimpleRAGPipeline(retriever="faiss", generator="openai", trace_db=trace_db)

# Run your pipeline
run = pipeline.run("Who invented the transistor?")
print("Answer:", run.answer)

# Later, compute metrics for all runs
evaluator = Evaluator(trace_db=trace_db)
results = evaluator.evaluate()
print(results)
```

## Metrics Overview

The RAG Evaluation Framework supports three classes of metrics:

- Generation Metrics — measure semantic/factual quality of answers.
- Ranking Metrics — measure ranking quality of rerankers.
- Retrieval Metrics — measure the quality and sufficiency of the retrieved context.

# Evaluator

## **Overview**

- The Evaluator is a convenience wrapper to run multiple metrics over model outputs and retrieval results.
- It orchestrates generation evaluation (text-generation metrics), retrieval evaluation (precision/recall/sufficiency), and reranker evaluation (NDCG, MAP, MRR). It produces CSV summaries by default.

> Quick notes
>
> - The Evaluator uses project metric classes (e.g., BartScore, BertScore, RougeScore, SemScore, PrecisionScore, RecallScore, MeanAP, MeanRR, RerankerNDCG, CumulativeNDCG, etc.). These metrics are in vero.metrics and are referenced internally.
> - Many methods expect particular CSV column names (see "Expected CSV schemas").
> - We highly recommend to install **gpu version** of torch library.

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
from vero.evaluator.evaluator import Evaluator

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

# Test Dataset Generation

## **Overview**

- The Test Dataset Generation module creates high-quality question-answer pairs derived from your document collection. It generates challenging queries designed to reveal retrieval and reasoning failures in RAG systems, and considering edge-case user personas.
- Internally it chunks documents, clusters related chunks, and uses an LLM to produce QA items with ground-truth chunk IDs and metadata.

### **Example**

```py
from vero.test_dataset_generator import generate_and_save

# Generate 100 queries from PDFs stored in ./data/pdfs directory and save outputs in test_dataset directory
generate_and_save(
    data_path='./data/pdfs/',
    usecase='Vitamin chatbot catering to general users for their daily queries',
    save_path_dir='test_dataset',
    n_queries=100
)
```

# Report Generation

## Overview

- The Report Generation module consolidates evaluation outputs from generation, retrieval, and reranking into a final report.
- It orchestrates a stateful workflow that processes CSV results from various evaluators and synthesizes comprehensive insights and recommendations.

## Example Usage

```py
from vero.report_generation_workflow import ReportGenerator

# Initialize the report generator
report_generator = ReportGenerator()

# Generate the final report by providing:
# - Pipeline configuration JSON file
# - Generation, Retrieval, and Reranker evaluation CSV files
report_generator.generate_report(
    'pipe_config_data.json',
    'Generation_Scores.csv',
    'Retrieval_Scores.csv',
    'Reranked_Scores.csv'
)
```


# Containerised evaluation (Docker)

## Prequisites
- Install [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/).
- Disk spcae >= 50 GB
- CPU >= 4 cores (GPU recommended for faster metric computation)
- RAM >= 8 GB

**Step 1 - Clone the repository and switch to the deploy folder**
```bash
git clone https://github.com/vero-eval/vero-eval.git
cd vero-deploy
```

**Step 2 - Inspect example inputs**
- Open the `data/inputs` folder inside `vero-deploy`.
- Review the example CSV provided (for example `data/inputs/input.csv`) and structure your input CSV the same way. Common column names used across the project include `Context Retrieved`, `Answer`, `Question`, `Retrieved Chunk IDs`, `True Chunk IDs`, `Chunk IDs`, and `Less Relevant Chunk IDs`.

**Step-3 Configure the input filename**
- Open `config.yaml` in the `vero-deploy` directory and check the key that references the input CSV filename. Update it to point to your CSV if needed (ensure the filename you pick matches the file in `data/inputs`).

**Step-4 Set your OpenAI key in Docker Compose**
- Open `docker-compose.yaml` and set the `OPEN_AI_KEY` environment variable for the evaluation service. Example snippet:
```yaml
x-base-config: &base-config
  image: crimsonceres/demo-test
  environment:
    - HF_HOME=/root/.cache/huggingface
    - TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
    - OPENAI_API_KEY=
```
You can also use environment substitution (for example `OPEN_AI_KEY=${OPEN_AI_KEY}`) and export the variable in your shell before running Docker.

**Step-5 Run the evaluation container**
- If you have a GPU available (recommended):
```bash
docker compose run evaluation-runner-gpu
```
- If you are using CPU only:
```bash
docker compose run evaluation-runner-cpu
```

>That is all required to run a containerised evaluation. Ensure `data/inputs` contains the CSV referenced in `config.yaml` and that `docker-compose.yaml` has `OPEN_AI_KEY` set before running the appropriate `docker compose run` command.

