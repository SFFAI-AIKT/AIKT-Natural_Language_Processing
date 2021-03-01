# Daily arXiv: Machine Translation - February, 2021

# Index

- [2021-03-01](#2021-03-01)
  - [1. Automated essay scoring using efficient transformer-based language models](#2021-03-01-1)
  - [2. Learning Chess Blindfolded: Evaluating Language Models on State Tracking](#2021-03-01-2)
  - [3. Gradient-guided Loss Masking for Neural Machine Translation](#2021-03-01-3)
- [Other Columns](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-03-01

[Return to Index](#Index)



<h2 id="2021-03-01-1">1. Automated essay scoring using efficient transformer-based language models</h2>

Title: [Automated essay scoring using efficient transformer-based language models](https://arxiv.org/abs/2102.13136)

Authors: [Christopher M Ormerod](https://arxiv.org/search/cs?searchtype=author&query=Ormerod%2C+C+M), [Akanksha Malhotra](https://arxiv.org/search/cs?searchtype=author&query=Malhotra%2C+A), [Amir Jafari](https://arxiv.org/search/cs?searchtype=author&query=Jafari%2C+A)

> Automated Essay Scoring (AES) is a cross-disciplinary effort involving Education, Linguistics, and Natural Language Processing (NLP). The efficacy of an NLP model in AES tests it ability to evaluate long-term dependencies and extrapolate meaning even when text is poorly written. Large pretrained transformer-based language models have dominated the current state-of-the-art in many NLP tasks, however, the computational requirements of these models make them expensive to deploy in practice. The goal of this paper is to challenge the paradigm in NLP that bigger is better when it comes to AES. To do this, we evaluate the performance of several fine-tuned pretrained NLP models with a modest number of parameters on an AES dataset. By ensembling our models, we achieve excellent results with fewer parameters than most pretrained transformer-based models.

| Comments: | 11 pages, 1 figure, 3 tables                                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2102.13136](https://arxiv.org/abs/2102.13136) [cs.CL]** |
|           | (or **[arXiv:2102.13136v1](https://arxiv.org/abs/2102.13136v1) [cs.CL]** for this version) |





<h2 id="2021-03-01-2">2. Learning Chess Blindfolded: Evaluating Language Models on State Tracking</h2>

Title: [Learning Chess Blindfolded: Evaluating Language Models on State Tracking](https://arxiv.org/abs/2102.13249)

Authors: [Shubham Toshniwal](https://arxiv.org/search/cs?searchtype=author&query=Toshniwal%2C+S), [Sam Wiseman](https://arxiv.org/search/cs?searchtype=author&query=Wiseman%2C+S), [Karen Livescu](https://arxiv.org/search/cs?searchtype=author&query=Livescu%2C+K), [Kevin Gimpel](https://arxiv.org/search/cs?searchtype=author&query=Gimpel%2C+K)

> Transformer language models have made tremendous strides in natural language understanding tasks. However, the complexity of natural language makes it challenging to ascertain how accurately these models are tracking the world state underlying the text. Motivated by this issue, we consider the task of language modeling for the game of chess. Unlike natural language, chess notations describe a simple, constrained, and deterministic domain. Moreover, we observe that the appropriate choice of chess notation allows for directly probing the world state, without requiring any additional probing-related machinery. We find that: (a) With enough training data, transformer language models can learn to track pieces and predict legal moves with high accuracy when trained solely on move sequences. (b) For small training sets providing access to board state information during training can yield significant improvements. (c) The success of transformer language models is dependent on access to the entire game history i.e. "full attention". Approximating this full attention results in a significant performance drop. We propose this testbed as a benchmark for future work on the development and analysis of transformer language models.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2102.13249](https://arxiv.org/abs/2102.13249) [cs.CL]** |
|           | (or **[arXiv:2102.13249v1](https://arxiv.org/abs/2102.13249v1) [cs.CL]** for this version) |



<h2 id="2021-03-01-3">3. Gradient-guided Loss Masking for Neural Machine Translation</h2>

Title: [Gradient-guided Loss Masking for Neural Machine Translation](https://arxiv.org/abs/2102.13549)

Authors: [Xinyi Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Ankur Bapna](https://arxiv.org/search/cs?searchtype=author&query=Bapna%2C+A), [Melvin Johnson](https://arxiv.org/search/cs?searchtype=author&query=Johnson%2C+M), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O)

> To mitigate the negative effect of low quality training data on the performance of neural machine translation models, most existing strategies focus on filtering out harmful data before training starts. In this paper, we explore strategies that dynamically optimize data usage during the training process using the model's gradients on a small set of clean data. At each training step, our algorithm calculates the gradient alignment between the training data and the clean data to mask out data with negative alignment. Our method has a natural intuition: good training data should update the model parameters in a similar direction as the clean data. Experiments on three WMT language pairs show that our method brings significant improvement over strong baselines, and the improvements are generalizable across test data from different domains.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2102.13549](https://arxiv.org/abs/2102.13549) [cs.CL]** |
|           | (or **[arXiv:2102.13549v1](https://arxiv.org/abs/2102.13549v1) [cs.CL]** for this version) |