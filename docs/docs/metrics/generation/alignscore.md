---
id: alignscore
title: AlignScore
---

# **AlignScore**

A factual consistency / alignment metric that evaluates how well a **claim** (generated or asserted text) is supported by a **context** (retrieved or original text).  
It uses a unified alignment function (often built on a pretrained alignment model) to score alignment between context chunks and claim sentences.

* **Inputs:** context (text) and claim (text)  
* **Returns:** a single AlignScore value (higher = stronger factual consistency)

### **Insights**
| AlignScore       | Inference     |
| -------------- | ------------- |
| closer to 1    | high factual consistency  |
| closer to 0    | low factual consistency  |


### **Example**
```py
from vero import AlignScore

with AlignScore() as ascore:
    # evaluate many (context, claim) pairs
    results = [ascore.evaluate(context, claim) for context, claim in zip(contexts_list, claims_list)]
print(results)
```
### **Output**
```text
0.91