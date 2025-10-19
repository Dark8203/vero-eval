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
from vero.metrics import SufficiencyScore

#example inputs
#context
#user_query
query = 'When was the Eiffel Tower built and where is it located?'
context_retrieved = ['The Eiffel Tower was built between 1887 and 1889 and is located in Paris, France.','Paris is the capital of France and known for the Louvre museum.','The Great Wall of China is more than 13,000 miles long and was built across northern China.']
ss = SufficiencyScore(context_retrieved,query,api_key)
print(ss.evaluate())
```

### **Output**
```text
0.77