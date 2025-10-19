---
id: semscore
title: SEMScore
---

# **SEMScore**

A metric that measures semantic similarity between candidate (generated) text and reference text using embeddings and cosine similarity.
* **Inputs:** candidate (generated) text and reference text.
* **Returns:** a single SEMScore value.

### **Insights**
| SEMScore       | Inference     |
| -------------- | ------------- |
| closer to 1    | more semantically similar  |
| closer to 0    | unrelated  |
| negative score | semantically opposite |

### **Example**
```py
from vero.metrics import SEMScore

#example inputs
#chunks_list = ["The cat sat on the mat.", "The dog barked at the mailman."]
#answers_list = ["A cat is sitting on a mat and a dog is barking at the mailman."]
with SEMScore() as ss:
    sem_results = [ss.evaluate(chunk, ans) for chunk, ans in zip(chunks_list, answers_list)]
print(sem_results)
```

### **Output**
```text
0.92