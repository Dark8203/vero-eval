---
id: sufficiencyscore
title: Sufficiency Score
---

# **Sufficiency Score**

Determines whether the retrieved set is sufficient to cover all ground truth items.

* **Inputs:** retrieved list and ground truth list  
* **Returns:** sufficiency score (often 1 or 0)

### **Example**
```py
from vero import SufficiencyScore

#example inputs
#ch_r is the retrieved citations from the retriever
#ch_t is the ground truth citations
ch_r = [1,2,3,5,6]
ch_t = [2,3,4]
ss = SufficiencyScore(ch_r, ch_t)
print(ss.evaluate())
```

### **Output**
```text
0.77