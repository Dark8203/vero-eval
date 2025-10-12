---
id: end-to-end-eval
title: End-to-end evaluation tutorial
---

This tutorial shows how to run an evaluation suite against a small dataset.

1. Prepare your dataset (CSV with columns: query, ground_truth_answer, ground_truth_docs_ids)
2. Run a script that executes runs for each query and stores the trace
3. Compute metrics

> **Note:** This tutorial is yet to be implemented as a ready-to-run notebook. The code snippets below illustrate the key steps, that will be reproducible.


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
