---
id: recallscore
title: Recall Score
---

# **Recall Score**

Measures how many ground truth items are retrieved.

* **Inputs:** retrieved list and ground truth list  
* **Returns:** recall as a float between 0 and 1

### **Example**
```py
from vero import RecallScore

#example inputs
#ch_r is the retrieved citations from the retriever
#ch_t is the ground truth citations
ch_r = [1,2,3,5,6]
ch_t = [2,3,4]
rs = RecallScore(ch_r, ch_t)
print(rs.evaluate())
```

### **Output**
```text
0.76