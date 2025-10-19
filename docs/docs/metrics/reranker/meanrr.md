---
id: meanrr
title: MeanRR
---

# **Mean Reciprocal Rank (MeanRR)**

Mean Reciprocal Rank: for each query, take reciprocal rank of first relevant item, then average.

* **Inputs:** list of lists of retrieved items, list of lists of ground truth items  
* **Returns:** mean reciprocal rank (float)

### **Example**
```py
from vero.metrics import MeanRR

#example inputs
#rr is the reranked results from the retriever
#gt is the ground truth relevant items for each query
rr = [[1,2,3,5,6],[1,2,3,5,6]]
gt = [[2,3,6],[2,3,6]]
mrr = MeanRR(rr, gt)
print(mrr.evaluate())
```

### **Output**
```text
0.67