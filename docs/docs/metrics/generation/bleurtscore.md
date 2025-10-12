---
id: bleurtscore
title: BleurtScore
---

# **BleurtScore (Weighted Semantic Similarity)**

An advanced metric based on BLEURT that produces a more nuanced weighted similarity score.  
* **Inputs:** candidate (generated) text and reference text (or query, when used as retriever metric).
* **Returns:** a single weighted BLEURT score.

### **Use Cases**
- As a generation metric → highlights which chunks contribute more to the output.  
- As a retriever metric → measures semantic relationships even if exact matches are missing.

> **Note:**
> 
>Can be very useful for debugging:
> - If Context Recall is low, but Weighted Semantic Similarity score is high, it tells the developer: "Your retriever is finding documents that are about the right topic, but it's failing to find the specific sentence or fact needed for the answer"
> - If both scores are low, the retriever is failing at a more fundamental level

### **Insights**
| BluertScore       | Inference     |
| -------------- | ------------- |
| closer to 1    | high semantic similarity  |
| closer to 0    | low semantic similarity  | 

### **Example**
```py
from vero import BleurtScore

#example inputs
#chunks_list = ["The cat sat on the mat.", "The dog barked at the mailman."]
#answers_list = ["A cat is sitting on a mat and a dog is barking at the mailman."]
with BleurtScore() as bls:
    bleurt_results = [bls.evaluate(chunk, ans) for chunk, ans in zip(chunks_list, answers_list)]
print(bleurt_results)
```

### **Output**
```text
0.89

