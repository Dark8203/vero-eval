---
id: meanap
title: MeanAP
---

# **Mean Average Precision (MeanAP)**

Mean Average Precision: average over queries of the average precision value.

* **Inputs:** list of lists of retrieved items, list of lists of ground truth items  
* **Returns:** mean average precision (float)

### **Example**
```py
from vero.metrics import MeanAP

#example inputs
#rr is the reranked results from the retriever
#gt is the ground truth relevant items for each query
rr = [[1,2,3,5,6],[1,2,3,5,6]]
gt = [[2,3,6],[2,3,6]]
map = MeanAP(rr, gt)
print(map.evaluate())
```

### **Output**
```text
0.78