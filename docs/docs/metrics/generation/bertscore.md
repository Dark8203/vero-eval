---
id: bertscore
title: BERTScore
---

# **BERTScore**

It is an automatic evaluation metric for text generation tasks that measures the similarity between candidate and reference texts using contextual embeddings from pre-trained BERT models.
* **Inputs:** candidate (generated) text and reference text.
* **Returns:** precision, recall, and F1-score based on token-level similarity.

### **Example**
```py
from vero.metrics import BertScore

#example inputs
#chunks_list = ["The cat sat on the mat.", "The dog barked at the mailman."]
#answers_list = ["A cat is sitting on a mat and a dog is barking at the mailman."]
with BertScore() as bs:
    bert_results = [bs.evaluate(chunk, ans) for chunk, ans in tqdm(zip(chunks_list, answers_list), total=len(df_new))]
bert_dicts = [{'Precision': p, 'Recall': r, 'F1score': f} for p, r, f in bert_results]
print(bert_dicts)
```

### **Output**
```text
{Precision': 0.85, 'Recall': 0.80, 'F1score': 0.825}

