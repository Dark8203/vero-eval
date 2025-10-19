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
#example inputs
#answer
#context_retrieved
answer = 'The Eiffel Tower was built in 1889 and is located in louvre museum, France.'
context_retrieved = ['The Eiffel Tower was built between 1887 and 1889 and is located in Paris, France.','Paris is the capital of France and known for the Louvre museum.','The Great Wall of China is more than 13,000 miles long and was built across northern China.']

with GEvalScore(api_key) as ge:
    result = ge.evaluate(context_retrieved,answer,'Correctness',polling=True)
    print(result)
```
### **Output**
```text
Faithfulness : 0.94
