---
id: rougescore
title: ROUGEScore
---

# **ROUGE-L Score**

ROUGE-L focuses on the Longest Common Subsequence (LCS) between generated and reference texts.
* **Inputs:** candidate (generated) text and reference text.
* **Returns:** precision, recall, and F1-score based on LCS overlap.

### **Example**
```py
from vero import RougeScore

#example inputs
#chunks_list = ["The cat sat on the mat.", "The dog barked at the mailman."]
#answers_list = ["A cat is sitting on a mat and a dog is barking at the mailman."]
with RougeScore() as rs:
    rouge_results = [rs.evaluate(chunk, ans) for chunk, ans in zip(chunks_list, answers_list)]
rouge_dicts = [{'Precision': p, 'Recall': r, 'F1score': f} for p, r, f in rouge_results]
print(rouge_dicts)
```

### **Output**
```text
{'Precision': 0.78, 'Recall': 0.74, 'F1score': 0.76}
```