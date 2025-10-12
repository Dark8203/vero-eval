---
id: cumulative-ndcg
title: Cumulative NDCG
---

# **Cumulative NDCG**

Unique implementation of NDCG@k that can be used to evaluate the cumulative performance of retriever and reranker.

* **Inputs:** list of lists of retrieved items, list of dicts mapping item → relevance score  
* **Returns:** cumulative NDCG (float)

### **Example**
```py
from vero import CumulativeNDCG

#example inputs
#rr is the reranked results from the retriever
#ranks is the relevance scores for the items retrieved by the retriever
rr = [[1,2,3,5,6],[1,2,3,5,6]]
ranks = [{2:2, 3:2},{2:2, 3:2, 6:1}]
cndcg = CumulativeNDCG(rr, ranks)
print(cndcg.evaluate())
```

### **Output**
```text
0.69