---
id: quickstart
title: QuickStart Guide
---

# **Vero-Eval: A Quick Introduction**

**Vero-Eval** is an open-source evaluation framework designed to rigorously assess the performance of **Retrieval-Augmented Generation (RAG)** pipelines. It provides built-in tracing, logging, and a rich suite of metrics to evaluate each component in the pipeline — from retrieval and reranking to generation — all integrated end to end.

Key features of Vero-Eval:

- **Trace & Log Execution**: Each query runs through the RAG pipeline is logged into an SQLite database, capturing the user query, retrieved context, reranked items, and the model’s output.  
- **Component-level Metrics**: Evaluate intermediate pipeline stages using metrics like Precision, Recall, Sufficiency, Citation, Overlap, and Ranking metrics (e.g. MRR, MAP, NDCG).  
- **Generation Metrics**: Measure semantic, factual, and alignment quality of generated outputs using metrics such as BERTScore, ROUGE, SEMScore, AlignScore, BLEURT, and G-Eval.  
- **Modular & Extensible**: Easily plug in new metric classes or custom scoring logic; the framework is designed to grow with your needs.  
- **End-to-End Evaluation**: Combine component metrics to understand the holistic performance of your RAG system — not just individual parts.


## Starting with Vero-Eval

### Setup 
Install via pip (recommended inside a virtualenv):

```bash
pip install vero-eval
```

### Example Usage
Here’s how you might use it in a minimal workflow:

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
