# ML-or-DL-or-LLM-An-Empirical-Study-on-Multi-Label-Requirements-Classification
## ❗ Declaration
This repository contains a summary overview of this paper, as well as the open-source code, data, and experimental results that we have released. 
**We strictly adhere to the double-blind policy. The GitHub account name is just a randomly chosen online name and has nothing to do with anyone!**

## 📑 Overview
- 🧠 Objective: This paper conducts a comprehensive empirical comparison of ML, DL, and LLMbased approaches to assess their effectiveness in requirements
classification. 
- 🎈 Method: The experiments specifically focus on the most challenging multi-label classification (MLC) task, employing the same dataset and four evaluation metrics across all approaches. This study represents the first systematic application of LLMs to automated MLC in the requirement domain.
Result: 
  - The results show that larger parameter sizes do not necessarily result in better performance. The effectiveness of LLMs depends more on their architectural design than on parameter scale alone. 
  - LLMs exhibit significant potential in MLC of requirements under zeroshot scenarios. 
  - Furthermore, in cases of imbalanced multilabel distributions, more complex transfer learning neural networks do not necessarily yield superior performance.
- 🎉 Conclusion: 
These findings suggest that LLM-based MLC of requirements methods represent a promising research direction that may transcend the inherent limitations of
conventional techniques.

## 📊 Dataset
We restructure the EMSE dataset to ensure its compatibility with LLMs.
- The dataset has been processed in Alpaca format and stored in the "\data\llm" folder.
- The EMSE dataset used for the deep learning baseline is stored in the "data\dl" folder.

## 📌 Baselines
Our implemented deep learning baseline codes are stored in the "\Deep learning" folder.

## 📔 Results
The experimental results of all baseline models have been placed in the "\results" folder.

## We will keep updating!
