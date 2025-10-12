---
id: rerank-ndcg
title: Reranker NDCG
---

# **Reranker NDCG**

Normalized Discounted Cumulative Gain for ranking metrics using graded relevance scores.

* **Inputs:** list of lists of retrieved items, list of dicts mapping item → relevance score  
* **Returns:** NDCG score (float)

### **Example**
```py
from vero import RerankerNDCG

#example inputs
#rr is the reranked results from the retriever
#ranks is the relevance scores for the items retrieved by the retriever
rr = [[1,2,3,5,6],[1,2,3,5,6]]
ranks = [{2:2, 3:2},{2:2, 3:2, 6:1}]
rndcg = RerankerNDCG(rr, ranks)
print(rndcg.evaluate())
```

### **Output**
```text
0.89
