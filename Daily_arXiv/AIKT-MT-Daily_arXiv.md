# Daily arXiv: Machine Translation - February, 2021

# Index

- [2021-03-02](#2021-03-02)
  - [1. Generative Adversarial Transformers](#2021-03-02-1)
  - [2. Token-Modification Adversarial Attacks for Natural Language Processing: A Survey](#2021-03-02-2)
  - [3. M6: A Chinese Multimodal Pretrainer](#2021-03-02-3)
- [2021-03-01](#2021-03-01)
  - [1. Automated essay scoring using efficient transformer-based language models](#2021-03-01-1)
  - [2. Learning Chess Blindfolded: Evaluating Language Models on State Tracking](#2021-03-01-2)
  - [3. Gradient-guided Loss Masking for Neural Machine Translation](#2021-03-01-3)
- [Other Columns](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-03-02

[Return to Index](#Index)



<h2 id="2021-03-02-1">1. Generative Adversarial Transformers</h2>

Title: [Generative Adversarial Transformers](https://arxiv.org/abs/2103.01209)

Authors: [Drew A. Hudson](https://arxiv.org/search/cs?searchtype=author&query=Hudson%2C+D+A), [C. Lawrence Zitnick](https://arxiv.org/search/cs?searchtype=author&query=Zitnick%2C+C+L)

> We introduce the GANsformer, a novel and efficient type of transformer, and explore it for the task of visual generative modeling. The network employs a bipartite structure that enables long-range interactions across the image, while maintaining computation of linearly efficiency, that can readily scale to high-resolution synthesis. It iteratively propagates information from a set of latent variables to the evolving visual features and vice versa, to support the refinement of each in light of the other and encourage the emergence of compositional representations of objects and scenes. In contrast to the classic transformer architecture, it utilizes multiplicative integration that allows flexible region-based modulation, and can thus be seen as a generalization of the successful StyleGAN network. We demonstrate the model's strength and robustness through a careful evaluation over a range of datasets, from simulated multi-object environments to rich real-world indoor and outdoor scenes, showing it achieves state-of-the-art results in terms of image quality and diversity, while enjoying fast learning and better data-efficiency. Further qualitative and quantitative experiments offer us an insight into the model's inner workings, revealing improved interpretability and stronger disentanglement, and illustrating the benefits and efficacy of our approach. An implementation of the model is available at [this http URL](http://github.com/dorarad/gansformer).

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2103.01209](https://arxiv.org/abs/2103.01209) [cs.CV]** |
|           | (or **[arXiv:2103.01209v1](https://arxiv.org/abs/2103.01209v1) [cs.CV]** for this version) |







<h2 id="2021-03-02-2">2. Token-Modification Adversarial Attacks for Natural Language Processing: A Survey</h2>

Title: [Token-Modification Adversarial Attacks for Natural Language Processing: A Survey](https://arxiv.org/abs/2103.00676)

Authors: [Tom Roth](https://arxiv.org/search/cs?searchtype=author&query=Roth%2C+T), [Yansong Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+Y), [Alsharif Abuadbba](https://arxiv.org/search/cs?searchtype=author&query=Abuadbba%2C+A), [Surya Nepal](https://arxiv.org/search/cs?searchtype=author&query=Nepal%2C+S), [Wei Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+W)

> There are now many adversarial attacks for natural language processing systems. Of these, a vast majority achieve success by modifying individual document tokens, which we call here a \textit{token-modification} attack. Each token-modification attack is defined by a specific combination of fundamental \textit{components}, such as a constraint on the adversary or a particular search algorithm. Motivated by this observation, we survey existing token-modification attacks and extract the components of each. We use an attack-independent framework to structure our survey which results in an effective categorisation of the field and an easy comparison of components. We hope this survey will guide new researchers to this field and spark further research into the individual attack components.

| Comments: | 8 pages, 1 figure                                            |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Cryptography and Security (cs.CR); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2103.00676](https://arxiv.org/abs/2103.00676) [cs.CL]** |
|           | (or **[arXiv:2103.00676v1](https://arxiv.org/abs/2103.00676v1) [cs.CL]** for this version) |







<h2 id="2021-03-02-3">3. M6: A Chinese Multimodal Pretrainer</h2>

Title: [M6: A Chinese Multimodal Pretrainer](https://arxiv.org/abs/2103.00823)

Authors: [Junyang Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+J), [Rui Men](https://arxiv.org/search/cs?searchtype=author&query=Men%2C+R), [An Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+A), [Chang Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+C), [Ming Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+M), [Yichang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Peng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+P), [Ang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+A), [Le Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+L), [Xianyan Jia](https://arxiv.org/search/cs?searchtype=author&query=Jia%2C+X), [Jie Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+J), [Jianwei Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+J), [Xu Zou](https://arxiv.org/search/cs?searchtype=author&query=Zou%2C+X), [Zhikang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Xiaodong Deng](https://arxiv.org/search/cs?searchtype=author&query=Deng%2C+X), [Jie Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Jinbao Xue](https://arxiv.org/search/cs?searchtype=author&query=Xue%2C+J), [Huiling Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+H), [Jianxin Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+J), [Jin Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+J), [Yong Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Wei Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+W), [Jingren Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J), [J ie Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+J+i), [Hongxia Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+H)

> In this work, we construct the largest dataset for multimodal pretraining in Chinese, which consists of over 1.9TB images and 292GB texts that cover a wide range of domains. We propose a cross-modal pretraining method called M6, referring to Multi-Modality to Multi-Modality Multitask Mega-transformer, for unified pretraining on the data of single modality and multiple modalities. We scale the model size up to 10 billion and 100 billion parameters, and build the largest pretrained model in Chinese. We apply the model to a series of downstream applications, and demonstrate its outstanding performance in comparison with strong baselines. Furthermore, we specifically design a downstream task of text-guided image generation, and show that the finetuned M6 can create high-quality images with high resolution and abundant details.

| Comments: | 12 pages, technical report                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2103.00823](https://arxiv.org/abs/2103.00823) [cs.CL]** |
|           | (or **[arXiv:2103.00823v1](https://arxiv.org/abs/2103.00823v1) [cs.CL]** for this version) |







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