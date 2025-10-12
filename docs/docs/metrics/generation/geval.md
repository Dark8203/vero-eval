---
id: geval
title: G-Eval
---

# **G-Eval**

An LLM-based evaluation framework where a large language model directly scores the generated output against criteria such as relevance, consistency, and fluency.
* **Inputs:** candidate (generated) text, reference text, and evaluation prompt/criteria.
* **Returns:** a numerical score (or multiple scores across evaluation dimensions).

## **Capabilities**
- A unique implementation of g-eval where we calculate the weighted sum of all the possible scores with their linear probabilities and get the average of it as the final score.
- We provide the prompting capability where if you want you can provide your own custom prompt for evaluation or you can pass the metric name, metric description(optional) and we will generate the prompt for you.
- We also provide the polling capability which basically runs the g-eval any given number of times(default is 5) and get an average score as final score.
- Pass the references and candidate (optional : custom prompt, metric name, metric description, polling flag and polling number).
- Returns a final G-Eval score for the passed metric or prompt.

### **Example**
```py
from vero import GEvalScore

#example inputs
#chunks_list = ["The cat sat on the mat.", "The dog barked at the mailman."]
#answers_list = ["A cat is sitting on a mat and a dog is barking at the mailman."]
with GEvalScore(api_key) as g_eval:
    g_eval_results = [g_eval.evaluate(chunk,ans, metric='Faithfulness') for chunk, ans in tqdm(zip(chunks_list, answers_list), total=len(df_new))]
print(g_eval_results)
```
### **Output**
```text
Faithfulness : 0.94
