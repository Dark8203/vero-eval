---
id: bartscore
title: BARTScore
---

# **BARTScore**

A generation evaluation metric that uses a pretrained BART model to assess the quality of generated text against reference text and is a type of comparison score   .
* **Inputs:** candidate (generated) text and reference text.
* **Returns:** a numerical score (higher = better alignment with reference).

### **Example**
```py
from vero import BartScore

#example inputs
#chunks_list = ["The cat sat on the mat.", "The dog barked at the mailman."]
#answers_list = ["A cat is sitting on a mat and a dog is barking at the mailman."]
with BartScore() as bs:
    bart_results = [bs.evaluate(chunk, ans) for chunk, ans in zip(chunks_list, answers_list)]
print(bart_results)
```

### **Output**
```text
0.75
```

_Note: This score does not hold any meaning in itself, it can be used to compare two models or versions of RAG pipelines and comparision can done as - higher the score better the generation capabilites of that pipeline compared to another._