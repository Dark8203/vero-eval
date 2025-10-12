---
id: citationscore
title: Citation Score
---

# **Citation Score**

Measures the overlap of retrieved items with ground truth citations.

* **Inputs:** retrieved list and ground truth list  
* **Returns:** citation score as a float between 0 and 1

### **Example**
```py
from vero import CitationScore

#example inputs
#ch_r is the retrieved citations from the retriever
#ch_t is the ground truth citations
ch_r = [1,2,3,5,6]
ch_t = [2,3,4]
cs = CitationScore(ch_r, ch_t)
print(cs.evaluate())
```

### **Output**
```text
0.60