---
id: report-generation
title: Report Generation
---

# Report Generation

## Overview
- The Report Generation module consolidates evaluation outputs from generation, retrieval, and reranking into a final report.
- It orchestrates a stateful workflow that processes CSV results from various evaluators and synthesizes comprehensive insights and recommendations.

## When to use
- To merge multiple evaluation results into a single, review-ready report.
- To get an overall analysis including metric evaluations, analysis synthesis, and recommendations from your evaluation pipeline.

## Core Functionality
- Reads CSV files containing generation, retrieval, and reranker scores.
- Invokes a series of agent prompts using a state graph to analyze and interpret the results.
- Consolidates the analysis into a final Markdown report.

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
