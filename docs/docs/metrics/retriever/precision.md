---
id: precisionscore
title: Precision Score
---

# **Precision Score**

Measures how many retrieved items are relevant (i.e. in ground truth).

* **Inputs:** retrieved list and ground truth list  
* **Returns:** precision as a float between 0 and 1

### **Example**
```py
from vero import PrecisionScore

#example inputs
#ch_r is the retrieved citations from the retriever
#ch_t is the ground truth citations
ch_r = [1,2,3,5,6]
ch_t = [2,3,4]
ps = PrecisionScore(ch_r, ch_t)
print(ps.evaluate())
```

### **Output**
```text
0.60