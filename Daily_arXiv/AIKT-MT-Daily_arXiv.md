# Daily arXiv: Machine Translation - Oct., 2019

# Index

- [2019-10-10](#2019-10-10)
  - [1. Novel Applications of Factored Neural Machine Translation](#2019-10-10-1)
- [2019-10-09](#2019-10-09)
  - [1. Improving Neural Machine Translation Robustness via Data Augmentation: Beyond Back Translation](#2019-10-09-1)
  - [2. Make Up Your Mind! Adversarial Generation of Inconsistent Natural Language Explanations](#2019-10-09-2)
  - [3. One-To-Many Multilingual End-to-end Speech Translation](#2019-10-09-3)
  - [4. An Interactive Machine Translation Framework for Modernizing Historical Documents](#2019-10-09-4)
  - [5. Overcoming the Rare Word Problem for Low-Resource Language Pairs in Neural Machine Translation](#2019-10-09-5)
  - [6. Controlled Text Generation for Data Augmentation in Intelligent Artificial Agents](#2019-10-09-6)
- [2019-10-08](#2019-10-08)
  - [1. How Transformer Revitalizes Character-based Neural Machine Translation: An Investigation on Japanese-Vietnamese Translation Systems](#2019-10-08-1)
  - [2. Domain Differential Adaptation for Neural Machine Translation](#2019-10-08-2)
  - [3. On Leveraging the Visual Modality for Neural Machine Translation](#2019-10-08-3)
  - [4. Adversarial reconstruction for Multi-modal Machine Translation](#2019-10-08-4)
- [2019-10-07](#2019-10-07)
  - [1. Distilling Transformers into Simple Neural Networks with Unlabeled Transfer Data](#2019-10-07-1)
  - [2. Modeling Confidence in Sequence-to-Sequence Models](#2019-10-07-2)
  - [3. Can I Trust the Explainer? Verifying Post-hoc Explanatory Methods](#2019-10-07-3)
- [2019-10-04](#2019-10-04)
  - [1. Cracking the Contextual Commonsense Code: Understanding Commonsense Reasoning Aptitude of Deep Contextual Representations](#2019-10-04-1)
  - [2. Linking artificial and human neural representations of language](#2019-10-04-2)
- [2019-10-03](#2019-10-03)
  - [1. Speech-to-speech Translation between Untranscribed Unknown Languages](2019-10-03-1)
  - [2. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](2019-10-03-2)
- [2019-10-02](#2019-10-02)
  - [1. Interrogating the Explanatory Power of Attention in Neural Machine Translation](#2019-10-02-1)
  - [2. Improved Word Sense Disambiguation Using Pre-Trained Contextualized Word Representations](#2019-10-02-2)
  - [3. Multilingual End-to-End Speech Translation](#2019-10-02-3)
  - [4. When and Why is Document-level Context Useful in Neural Machine Translation?](#2019-10-02-4)
  - [5. Grammatical Error Correction in Low-Resource Scenarios](#2019-10-02-5)
  - [6. Application of Low-resource Machine Translation Techniques to Russian-Tatar Language Pair](#2019-10-02-6)
  - [7. A Survey of Methods to Leverage Monolingual Data in Low-resource Neural Machine Translation](#2019-10-02-7)
  - [8. Machine Translation for Machines: the Sentiment Classification Use Case](#2019-10-02-8)
  - [9. Putting Machine Translation in Context with the Noisy Channel Model](#2019-10-02-9)
- [2019-10-01](#2019-10-01)
  - [1. Revisiting Self-Training for Neural Sequence Generation](2019-10-01-1)
  - [2. The Source-Target Domain Mismatch Problem in Machine Translation](2019-10-01-2)
  - [3. Controllable Data Synthesis Method for Grammatical Error Correction](2019-10-01-3)
  - [4. Regressing Word and Sentence Embeddings for Regularization of Neural Machine Translation](2019-10-01-4)
  - [5. Simple and Effective Paraphrastic Similarity from Parallel Translations](2019-10-01-5)
- [2019-09](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-09.md)
- [2019-08](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-08.md)
- [2019-07](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-07.md)
- [2019-06](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-06.md)
- [2019-05](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-05.md)
- [2019-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-04.md)
- [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)



# 2019-10-10

[Return to Index](#Index)

<h2 id="2019-10-10-1">1. Novel Applications of Factored Neural Machine Translation</h2>

Title: [Novel Applications of Factored Neural Machine Translation](https://arxiv.org/abs/1910.03912)

Authors: [Patrick Wilken](https://arxiv.org/search/cs?searchtype=author&query=Wilken%2C+P), [Evgeny Matusov](https://arxiv.org/search/cs?searchtype=author&query=Matusov%2C+E)

*(Submitted on 9 Oct 2019)*

> In this work, we explore the usefulness of target factors in neural machine translation (NMT) beyond their original purpose of predicting word lemmas and their inflections, as proposed by Garcìa-Martìnez et al., 2016. For this, we introduce three novel applications of the factored output architecture: In the first one, we use a factor to explicitly predict the word case separately from the target word itself. This allows for information to be shared between different casing variants of a word. In a second task, we use a factor to predict when two consecutive subwords have to be joined, eliminating the need for target subword joining markers. The third task is the prediction of special tokens of the operation sequence NMT model (OSNMT) of Stahlberg et al., 2018. Automatic evaluation on English-to-German and English-to-Turkish tasks showed that integration of such auxiliary prediction tasks into NMT is at least as good as the standard NMT approach. For the OSNMT, we observed a significant improvement in BLEU over the baseline OSNMT implementation due to a reduced output sequence length that resulted from the introduction of the target factors.

| Subjects: | **Computation and Language (cs.CL)**; Neural and Evolutionary Computing (cs.NE) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1910.03912 [cs.CL]**                                 |
|           | (or **arXiv:1910.03912v1 [cs.CL]** for this version)         |



# 2019-10-09

[Return to Index](#Index)



<h2 id="2019-10-09-1">1. Improving Neural Machine Translation Robustness via Data Augmentation: Beyond Back Translation</h2>
Title: [Improving Neural Machine Translation Robustness via Data Augmentation: Beyond Back Translation](https://arxiv.org/abs/1910.03009)

Authors: [Zhenhao Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Lucia Specia](https://arxiv.org/search/cs?searchtype=author&query=Specia%2C+L)

*(Submitted on 7 Oct 2019)*

> Neural Machine Translation (NMT) models have been proved strong when translating clean texts, but they are very sensitive to noise in the input. Improving NMT models robustness can be seen as a form of "domain'' adaption to noise. The recently created Machine Translation on Noisy Text task corpus provides noisy-clean parallel data for a few language pairs, but this data is very limited in size and diversity. The state-of-the-art approaches are heavily dependent on large volumes of back-translated data. This paper has two main contributions: Firstly, we propose new data augmentation methods to extend limited noisy data and further improve NMT robustness to noise while keeping the models small. Secondly, we explore the effect of utilizing noise from external data in the form of speech transcripts and show that it could help robustness.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1910.03009 [cs.CL]**                         |
|           | (or **arXiv:1910.03009v1 [cs.CL]** for this version) |





<h2 id="2019-10-09-2">2. Make Up Your Mind! Adversarial Generation of Inconsistent Natural Language Explanations</h2>
Title: [Make Up Your Mind! Adversarial Generation of Inconsistent Natural Language Explanations](https://arxiv.org/abs/1910.03065)

Authors: [Camburu Oana-Maria](https://arxiv.org/search/cs?searchtype=author&query=Oana-Maria%2C+C), [Shillingford Brendan](https://arxiv.org/search/cs?searchtype=author&query=Brendan%2C+S), [Minervini Pasquale](https://arxiv.org/search/cs?searchtype=author&query=Pasquale%2C+M), [Lukasiewicz Thomas](https://arxiv.org/search/cs?searchtype=author&query=Thomas%2C+L), [Blunsom Phil](https://arxiv.org/search/cs?searchtype=author&query=Phil%2C+B)

*(Submitted on 7 Oct 2019)*

> To increase trust in artificial intelligence systems, a growing amount of works are enhancing these systems with the capability of producing natural language explanations that support their predictions. In this work, we show that such appealing frameworks are nonetheless prone to generating inconsistent explanations, such as "A dog is an animal" and "A dog is not an animal", which are likely to decrease users' trust in these systems. To detect such inconsistencies, we introduce a simple but effective adversarial framework for generating a complete target sequence, a scenario that has not been addressed so far. Finally, we apply our framework to a state-of-the-art neural model that provides natural language explanations on SNLI, and we show that this model is capable of generating a significant amount of inconsistencies.

| Subjects:          | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | NeurIPS 2019 Workshop on Safety and Robustness in Decision Making, Vancouver, Canada |
| Cite as:           | **arXiv:1910.03065 [cs.CL]**                                 |
|                    | (or **arXiv:1910.03065v1 [cs.CL]** for this version)         |





<h2 id="2019-10-09-3">3. One-To-Many Multilingual End-to-end Speech Translation</h2>
Title: [One-To-Many Multilingual End-to-end Speech Translation](https://arxiv.org/abs/1910.03320)

Authors: [Mattia Antonino Di Gangi](https://arxiv.org/search/cs?searchtype=author&query=Di+Gangi%2C+M+A), [Matteo Negri](https://arxiv.org/search/cs?searchtype=author&query=Negri%2C+M), [Marco Turchi](https://arxiv.org/search/cs?searchtype=author&query=Turchi%2C+M)

*(Submitted on 8 Oct 2019)*

> Nowadays, training end-to-end neural models for spoken language translation (SLT) still has to confront with extreme data scarcity conditions. The existing SLT parallel corpora are indeed orders of magnitude smaller than those available for the closely related tasks of automatic speech recognition (ASR) and machine translation (MT), which usually comprise tens of millions of instances. To cope with data paucity, in this paper we explore the effectiveness of transfer learning in end-to-end SLT by presenting a multilingual approach to the task. Multilingual solutions are widely studied in MT and usually rely on ``\textit{target forcing}'', in which multilingual parallel data are combined to train a single model by prepending to the input sequences a language token that specifies the target language. However, when tested in speech translation, our experiments show that MT-like \textit{target forcing}, used as is, is not effective in discriminating among the target languages. Thus, we propose a variant that uses target-language embeddings to shift the input representations in different portions of the space according to the language, so to better support the production of output in the desired target language. Our experiments on end-to-end SLT from English into six languages show important improvements when translating into similar languages, especially when these are supported by scarce data. Further improvements are obtained when using English ASR data as an additional language (up to *[Math Processing Error]* BLEU points).

| Comments: | 8 pages, one figure, version accepted at ASRU 2019           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Audio and Speech Processing (eess.AS) |
| Cite as:  | **arXiv:1910.03320 [cs.CL]**                                 |
|           | (or **arXiv:1910.03320v1 [cs.CL]** for this version)         |





<h2 id="2019-10-09-4">4. An Interactive Machine Translation Framework for Modernizing Historical Documents</h2>
Title: [An Interactive Machine Translation Framework for Modernizing Historical Documents](https://arxiv.org/abs/1910.03355)

Authors: [Miguel Domingo](https://arxiv.org/search/cs?searchtype=author&query=Domingo%2C+M), [Francisco Casacuberta](https://arxiv.org/search/cs?searchtype=author&query=Casacuberta%2C+F)

*(Submitted on 8 Oct 2019)*

> Due to the nature of human language, historical documents are hard to comprehend by contemporary people. This limits their accessibility to scholars specialized in the time period in which the documents were written. Modernization aims at breaking this language barrier by generating a new version of a historical document, written in the modern version of the document's original language. However, while it is able to increase the document's comprehension, modernization is still far from producing an error-free version. In this work, we propose a collaborative framework in which a scholar can work together with the machine to generate the new version. We tested our approach on a simulated environment, achieving significant reductions of the human effort needed to produce the modernized version of the document.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1910.03355 [cs.CL]**                         |
|           | (or **arXiv:1910.03355v1 [cs.CL]** for this version) |





<h2 id="2019-10-09-5">5. Overcoming the Rare Word Problem for Low-Resource Language Pairs in Neural Machine Translation</h2>
Title: [Overcoming the Rare Word Problem for Low-Resource Language Pairs in Neural Machine Translation](https://arxiv.org/abs/1910.03467)

Authors: [Thi-Vinh Ngo](https://arxiv.org/search/cs?searchtype=author&query=Ngo%2C+T), [Thanh-Le Ha](https://arxiv.org/search/cs?searchtype=author&query=Ha%2C+T), [Phuong-Thai Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+P), [Le-Minh Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+L)

*(Submitted on 7 Oct 2019)*

> Among the six challenges of neural machine translation (NMT) coined by (Koehn and Knowles, 2017), rare-word problem is considered the most severe one, especially in translation of low-resource languages. In this paper, we propose three solutions to address the rare words in neural machine translation systems. First, we enhance source context to predict the target words by connecting directly the source embeddings to the output of the attention component in NMT. Second, we propose an algorithm to learn morphology of unknown words for English in supervised way in order to minimize the adverse effect of rare-word problem. Finally, we exploit synonymous relation from the WordNet to overcome out-of-vocabulary (OOV) problem of NMT. We evaluate our approaches on two low-resource language pairs: English-Vietnamese and Japanese-Vietnamese. In our experiments, we have achieved significant improvements of up to roughly +1.0 BLEU points in both language pairs.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1910.03467 [cs.CL]**                                 |
|           | (or **arXiv:1910.03467v1 [cs.CL]** for this version)         |





<h2 id="2019-10-09-6">6. Controlled Text Generation for Data Augmentation in Intelligent Artificial Agents</h2>
Title: [Controlled Text Generation for Data Augmentation in Intelligent Artificial Agents](https://arxiv.org/abs/1910.03487)

Authors: [Nikolaos Malandrakis](https://arxiv.org/search/cs?searchtype=author&query=Malandrakis%2C+N), [Minmin Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+M), [Anuj Goyal](https://arxiv.org/search/cs?searchtype=author&query=Goyal%2C+A), [Shuyang Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+S), [Abhishek Sethi](https://arxiv.org/search/cs?searchtype=author&query=Sethi%2C+A), [Angeliki Metallinou](https://arxiv.org/search/cs?searchtype=author&query=Metallinou%2C+A)

*(Submitted on 4 Oct 2019)*

> Data availability is a bottleneck during early stages of development of new capabilities for intelligent artificial agents. We investigate the use of text generation techniques to augment the training data of a popular commercial artificial agent across categories of functionality, with the goal of faster development of new functionality. We explore a variety of encoder-decoder generative models for synthetic training data generation and propose using conditional variational auto-encoders. Our approach requires only direct optimization, works well with limited data and significantly outperforms the previous controlled text generation techniques. Further, the generated data are used as additional training samples in an extrinsic intent classification task, leading to improved performance by up to 5\% absolute f-score in low-resource cases, validating the usefulness of our approach.

| Comments: | EMNLP WNGT workshop                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Machine Learning (stat.ML) |
| Cite as:  | **arXiv:1910.03487 [cs.CL]**                                 |
|           | (or **arXiv:1910.03487v1 [cs.CL]** for this version)         |



# 2019-10-08

[Return to Index](#Index)



<h2 id="2019-10-08-1">1. How Transformer Revitalizes Character-based Neural Machine Translation: An Investigation on Japanese-Vietnamese Translation Systems</h2>
Title: [How Transformer Revitalizes Character-based Neural Machine Translation: An Investigation on Japanese-Vietnamese Translation Systems](https://arxiv.org/abs/1910.02238)

Authors:[Thi-Vinh Ngo](https://arxiv.org/search/cs?searchtype=author&query=Ngo%2C+T), [Thanh-Le Ha](https://arxiv.org/search/cs?searchtype=author&query=Ha%2C+T), [Phuong-Thai Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+P), [Le-Minh Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+L)

*(Submitted on 5 Oct 2019)*

> While translating between Chinese-centric languages, many works have discovered clear advantages of using characters as the translation unit. Unfortunately, traditional recurrent neural machine translation systems hinder the practical usage of those character-based systems due to their architectural limitations. They are unfavorable in handling extremely long sequences as well as highly restricted in parallelizing the computations. In this paper, we demonstrate that the new transformer architecture can perform character-based translation better than the recurrent one. We conduct experiments on a low-resource language pair: Japanese-Vietnamese. Our models considerably outperform the state-of-the-art systems which employ word-based recurrent architectures.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1910.02238 [cs.CL]**                         |
|           | (or **arXiv:1910.02238v1 [cs.CL]** for this version) |





<h2 id="2019-10-08-2">2. Domain Differential Adaptation for Neural Machine Translation</h2>
Title: [Domain Differential Adaptation for Neural Machine Translation](https://arxiv.org/abs/1910.02555)

Authors: [Zi-Yi Dou](https://arxiv.org/search/cs?searchtype=author&query=Dou%2C+Z), [Xinyi Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Junjie Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+J), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G)

*(Submitted on 7 Oct 2019)*

> Neural networks are known to be data hungry and domain sensitive, but it is nearly impossible to obtain large quantities of labeled data for every domain we are interested in. This necessitates the use of domain adaptation strategies. One common strategy encourages generalization by aligning the global distribution statistics between source and target domains, but one drawback is that the statistics of different domains or tasks are inherently divergent, and smoothing over these differences can lead to sub-optimal performance. In this paper, we propose the framework of {\it Domain Differential Adaptation (DDA)}, where instead of smoothing over these differences we embrace them, directly modeling the difference between domains using models in a related task. We then use these learned domain differentials to adapt models for the target task accordingly. Experimental results on domain adaptation for neural machine translation demonstrate the effectiveness of this strategy, achieving consistent improvements over other alternative adaptation strategies in multiple experimental settings.

| Comments: | Workshop on Neural Generation and Translation (WNGT) at EMNLP 2019 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1910.02555 [cs.CL]**                                 |
|           | (or **arXiv:1910.02555v1 [cs.CL]** for this version)         |





<h2 id="2019-10-08-3">3. On Leveraging the Visual Modality for Neural Machine Translation</h2>
Title: [On Leveraging the Visual Modality for Neural Machine Translation](https://arxiv.org/abs/1910.02754)

Authors:[Vikas Raunak](https://arxiv.org/search/cs?searchtype=author&query=Raunak%2C+V), [Sang Keun Choe](https://arxiv.org/search/cs?searchtype=author&query=Choe%2C+S+K), [Quanyang Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+Q), [Yi Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+Y), [Florian Metze](https://arxiv.org/search/cs?searchtype=author&query=Metze%2C+F)

*(Submitted on 7 Oct 2019)*

> Leveraging the visual modality effectively for Neural Machine Translation (NMT) remains an open problem in computational linguistics. Recently, Caglayan et al. posit that the observed gains are limited mainly due to the very simple, short, repetitive sentences of the Multi30k dataset (the only multimodal MT dataset available at the time), which renders the source text sufficient for context. In this work, we further investigate this hypothesis on a new large scale multimodal Machine Translation (MMT) dataset, How2, which has 1.57 times longer mean sentence length than Multi30k and no repetition. We propose and evaluate three novel fusion techniques, each of which is designed to ensure the utilization of visual context at different stages of the Sequence-to-Sequence transduction pipeline, even under full linguistic context. However, we still obtain only marginal gains under full linguistic context and posit that visual embeddings extracted from deep vision models (ResNet for Multi30k, ResNext for How2) do not lend themselves to increasing the discriminativeness between the vocabulary elements at token level prediction in NMT. We demonstrate this qualitatively by analyzing attention distribution and quantitatively through Principal Component Analysis, arriving at the conclusion that it is the quality of the visual embeddings rather than the length of sentences, which need to be improved in existing MMT datasets.

| Comments: | Accepted to INLG 2019                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1910.02754 [cs.CL]**                                 |
|           | (or **arXiv:1910.02754v1 [cs.CL]** for this version)         |





<h2 id="2019-10-08-4">4. Adversarial reconstruction for Multi-modal Machine Translation</h2>
Title: [Adversarial reconstruction for Multi-modal Machine Translation](https://arxiv.org/abs/1910.02766)

Authors:[Jean-Benoit Delbrouck](https://arxiv.org/search/cs?searchtype=author&query=Delbrouck%2C+J), [Stéphane Dupont](https://arxiv.org/search/cs?searchtype=author&query=Dupont%2C+S)

*(Submitted on 7 Oct 2019)*

> Even with the growing interest in problems at the intersection of Computer Vision and Natural Language, grounding (i.e. identifying) the components of a structured description in an image still remains a challenging task. This contribution aims to propose a model which learns grounding by reconstructing the visual features for the Multi-modal translation task. Previous works have partially investigated standard approaches such as regression methods to approximate the reconstruction of a visual input. In this paper, we propose a different and novel approach which learns grounding by adversarial feedback. To do so, we modulate our network following the recent promising adversarial architectures and evaluate how the adversarial response from a visual reconstruction as an auxiliary task helps the model in its learning. We report the highest scores in term of BLEU and METEOR metrics on the different datasets.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1910.02766 [cs.CL]**                         |
|           | (or **arXiv:1910.02766v1 [cs.CL]** for this version) |







# 2019-10-07

[Return to Index](#Index)



<h2 id="2019-10-07-1">1. Distilling Transformers into Simple Neural Networks with Unlabeled Transfer Data</h2>
Title: [Distilling Transformers into Simple Neural Networks with Unlabeled Transfer Data](https://arxiv.org/abs/1910.01769)

Authors: [Subhabrata Mukherjee](https://arxiv.org/search/cs?searchtype=author&query=Mukherjee%2C+S), [Ahmed Hassan Awadallah](https://arxiv.org/search/cs?searchtype=author&query=Awadallah%2C+A+H)

*(Submitted on 4 Oct 2019)*

> Recent advances in pre-training huge models on large amounts of text through self supervision have obtained state-of-the-art results in various natural language processing tasks. However, these huge and expensive models are difficult to use in practise for downstream tasks. Some recent efforts use knowledge distillation to compress these models. However, we see a gap between the performance of the smaller student models as compared to that of the large teacher. In this work, we leverage large amounts of in-domain unlabeled transfer data in addition to a limited amount of labeled training instances to bridge this gap. We show that simple RNN based student models even with hard distillation can perform at par with the huge teachers given the transfer set. The student performance can be further improved with soft distillation and leveraging teacher intermediate representations. We show that our student models can compress the huge teacher by up to 26x while still matching or even marginally exceeding the teacher performance in low-resource settings with small amount of labeled data.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1910.01769 [cs.CL]**                                 |
|           | (or **arXiv:1910.01769v1 [cs.CL]** for this version)         |





<h2 id="2019-10-07-2">2. Modeling Confidence in Sequence-to-Sequence Models</h2>
Title: [Modeling Confidence in Sequence-to-Sequence Models](https://arxiv.org/abs/1910.01859)

Authors: [Jan Niehues](https://arxiv.org/search/cs?searchtype=author&query=Niehues%2C+J), [Ngoc-Quan Pham](https://arxiv.org/search/cs?searchtype=author&query=Pham%2C+N)

*(Submitted on 4 Oct 2019)*

> Recently, significant improvements have been achieved in various natural language processing tasks using neural sequence-to-sequence models. While aiming for the best generation quality is important, ultimately it is also necessary to develop models that can assess the quality of their output.
> In this work, we propose to use the similarity between training and test conditions as a measure for models' confidence. We investigate methods solely using the similarity as well as methods combining it with the posterior probability. While traditionally only target tokens are annotated with confidence measures, we also investigate methods to annotate source tokens with confidence. By learning an internal alignment model, we can significantly improve confidence projection over using state-of-the-art external alignment tools. We evaluate the proposed methods on downstream confidence estimation for machine translation (MT). We show improvements on segment-level confidence estimation as well as on confidence estimation for source tokens. In addition, we show that the same methods can also be applied to other tasks using sequence-to-sequence models. On the automatic speech recognition (ASR) task, we are able to find 60% of the errors by looking at 20% of the data.

| Comments: | 8 pages; INLG 2019                                   |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1910.01859 [cs.CL]**                         |
|           | (or **arXiv:1910.01859v1 [cs.CL]** for this version) |





<h2 id="2019-10-07-3">3. Can I Trust the Explainer? Verifying Post-hoc Explanatory Methods</h2>
Title: [Can I Trust the Explainer? Verifying Post-hoc Explanatory Methods](https://arxiv.org/abs/1910.02065)

Authors: [Camburu Oana-Maria](https://arxiv.org/search/cs?searchtype=author&query=Oana-Maria%2C+C), [Giunchiglia Eleonora](https://arxiv.org/search/cs?searchtype=author&query=Eleonora%2C+G), [Foerster Jakob](https://arxiv.org/search/cs?searchtype=author&query=Jakob%2C+F), [Lukasiewicz Thomas](https://arxiv.org/search/cs?searchtype=author&query=Thomas%2C+L), [Blunsom Phil](https://arxiv.org/search/cs?searchtype=author&query=Phil%2C+B)

*(Submitted on 4 Oct 2019)*

> For AI systems to garner widespread public acceptance, we must develop methods capable of explaining the decisions of black-box models such as neural networks. In this work, we identify two issues of current explanatory methods. First, we show that two prevalent perspectives on explanations---feature-additivity and feature-selection---lead to fundamentally different instance-wise explanations. In the literature, explainers from different perspectives are currently being directly compared, despite their distinct explanation goals. The second issue is that current post-hoc explainers have only been thoroughly validated on simple models, such as linear regression, and, when applied to real-world neural networks, explainers are commonly evaluated under the assumption that the learned models behave reasonably. However, neural networks often rely on unreasonable correlations, even when producing correct decisions. We introduce a verification framework for explanatory methods under the feature-selection perspective. Our framework is based on a non-trivial neural network architecture trained on a real-world task, and for which we are able to provide guarantees on its inner workings. We validate the efficacy of our evaluation by showing the failure modes of current explainers. We aim for this framework to provide a publicly available, off-the-shelf evaluation when the feature-selection perspective on explanations is needed.

| Comments: | NeurIPS 2019 Workshop on Safety and Robustness in Decision Making, Vancouver, Canada |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1910.02065 [cs.CL]**                                 |
|           | (or **arXiv:1910.02065v1 [cs.CL]** for this version)         |





# 2019-10-04

[Return to Index](#Index)



<h2 id="2019-10-04-1">1. Cracking the Contextual Commonsense Code: Understanding Commonsense Reasoning Aptitude of Deep Contextual Representations</h2>
Title: [Cracking the Contextual Commonsense Code: Understanding Commonsense Reasoning Aptitude of Deep Contextual Representations](https://arxiv.org/abs/1910.01157)

Authors: [Jeff Da](https://arxiv.org/search/cs?searchtype=author&query=Da%2C+J), [Jungo Kusai](https://arxiv.org/search/cs?searchtype=author&query=Kusai%2C+J)

*(Submitted on 2 Oct 2019)*

> Pretrained deep contextual representations have advanced the state-of-the-art on various commonsense NLP tasks, but we lack a concrete understanding of the capability of these models. Thus, we investigate and challenge several aspects of BERT's commonsense representation abilities. First, we probe BERT's ability to classify various object attributes, demonstrating that BERT shows a strong ability in encoding various commonsense features in its embedding space, but is still deficient in many areas. Next, we show that, by augmenting BERT's pretraining data with additional data related to the deficient attributes, we are able to improve performance on a downstream commonsense reasoning task while using a minimal amount of data. Finally, we develop a method of fine-tuning knowledge graphs embeddings alongside BERT and show the continued importance of explicit knowledge graphs.

| Comments: | Accepted to EMNLP Commonsense (COIN)                 |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1910.01157 [cs.CL]**                         |
|           | (or **arXiv:1910.01157v1 [cs.CL]** for this version) |





<h2 id="2019-10-04-2">2. Linking artificial and human neural representations of language</h2>
Title: [Linking artificial and human neural representations of language](https://arxiv.org/abs/1910.01244)

Authors: [Jon Gauthier](https://arxiv.org/search/cs?searchtype=author&query=Gauthier%2C+J), [Roger Levy](https://arxiv.org/search/cs?searchtype=author&query=Levy%2C+R)

*(Submitted on 2 Oct 2019)*

> What information from an act of sentence understanding is robustly represented in the human brain? We investigate this question by comparing sentence encoding models on a brain decoding task, where the sentence that an experimental participant has seen must be predicted from the fMRI signal evoked by the sentence. We take a pre-trained BERT architecture as a baseline sentence encoding model and fine-tune it on a variety of natural language understanding (NLU) tasks, asking which lead to improvements in brain-decoding performance.
> We find that none of the sentence encoding tasks tested yield significant increases in brain decoding performance. Through further task ablations and representational analyses, we find that tasks which produce syntax-light representations yield significant improvements in brain decoding performance. Our results constrain the space of NLU models that could best account for human neural representations of language, but also suggest limits on the possibility of decoding fine-grained syntactic information from fMRI human neuroimaging.

| Comments: | EMNLP 2019                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Neurons and Cognition (q-bio.NC) |
| Cite as:  | **arXiv:1910.01244 [cs.CL]**                                 |
|           | (or **arXiv:1910.01244v1 [cs.CL]** for this version)         |





# 2019-10-03

[Return to Index](#Index)



<h2 id="2019-10-03-1">1. Speech-to-speech Translation between Untranscribed Unknown Languages</h2>
Title: [Speech-to-speech Translation between Untranscribed Unknown Languages](https://arxiv.org/abs/1910.00795)

Authors: [Andros Tjandra](https://arxiv.org/search/cs?searchtype=author&query=Tjandra%2C+A), [Sakriani Sakti](https://arxiv.org/search/cs?searchtype=author&query=Sakti%2C+S), [Satoshi Nakamura](https://arxiv.org/search/cs?searchtype=author&query=Nakamura%2C+S)

*(Submitted on 2 Oct 2019)*

> In this paper, we explore a method for training speech-to-speech translation tasks without any transcription or linguistic supervision. Our proposed method consists of two steps: First, we train and generate discrete representation with unsupervised term discovery with a discrete quantized autoencoder. Second, we train a sequence-to-sequence model that directly maps the source language speech to the target language's discrete representation. Our proposed method can directly generate target speech without any auxiliary or pre-training steps with a source or target transcription. To the best of our knowledge, this is the first work that performed pure speech-to-speech translation between untranscribed unknown languages.

| Comments: | Accepted in IEEE ASRU 2019. Web-page for more samples & details: [this https URL](https://sp2code-translation-v1.netlify.com/) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **arXiv:1910.00795 [cs.CL]**                                 |
|           | (or **arXiv:1910.00795v1 [cs.CL]** for this version)         |





<h2 id="2019-10-03-2">2. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter</h2>
Title: [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)

Authors: [Victor Sanh](https://arxiv.org/search/cs?searchtype=author&query=Sanh%2C+V), [Lysandre Debut](https://arxiv.org/search/cs?searchtype=author&query=Debut%2C+L), [Julien Chaumond](https://arxiv.org/search/cs?searchtype=author&query=Chaumond%2C+J), [Thomas Wolf](https://arxiv.org/search/cs?searchtype=author&query=Wolf%2C+T)

*(Submitted on 2 Oct 2019)*

> As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural Language Processing (NLP), operating these large models in on-the-edge and/or under constrained computational training or inference budgets remain challenging. In this work, we propose a method to pre-train a smaller general-purpose language representation model, called DistilBERT, which can then be fine-tuned with good performances on a wide range of tasks like its larger counterparts. While most prior work investigated the use of distillation for building task-specific models, we leverage knowledge distillation during the pre-training phase and show that it is possible to reduce the size of a BERT model by 40%, while retaining 97% of its language understanding capabilities and being 60% faster. To leverage the inductive biases learned by larger models during pre-training, we introduce a triple loss combining language modeling, distillation and cosine-distance losses. Our smaller, faster and lighter model is cheaper to pre-train and we demonstrate its capabilities for on-device computations in a proof-of-concept experiment and a comparative on-device study.

| Comments: | 5 pages, 1 figure, 4 tables. Accepted at the 5th Workshop on Energy Efficient Machine Learning and Cognitive Computing - NeurIPS 2019 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1910.01108 [cs.CL]**                                 |
|           | (or **arXiv:1910.01108v1 [cs.CL]** for this version)         |



# 2019-10-02

[Return to Index](#Index)



<h2 id="2019-10-02-1">1. Interrogating the Explanatory Power of Attention in Neural Machine Translation</h2>
Title: [Interrogating the Explanatory Power of Attention in Neural Machine Translation](https://arxiv.org/abs/1910.00139)

Authors: [Pooya Moradi](https://arxiv.org/search/cs?searchtype=author&query=Moradi%2C+P), [Nishant Kambhatla](https://arxiv.org/search/cs?searchtype=author&query=Kambhatla%2C+N), [Anoop Sarkar](https://arxiv.org/search/cs?searchtype=author&query=Sarkar%2C+A)

*(Submitted on 30 Sep 2019)*

> Attention models have become a crucial component in neural machine translation (NMT). They are often implicitly or explicitly used to justify the model's decision in generating a specific token but it has not yet been rigorously established to what extent attention is a reliable source of information in NMT. To evaluate the explanatory power of attention for NMT, we examine the possibility of yielding the same prediction but with counterfactual attention models that modify crucial aspects of the trained attention model. Using these counterfactual attention mechanisms we assess the extent to which they still preserve the generation of function and content words in the translation process. Compared to a state of the art attention model, our counterfactual attention models produce 68% of function words and 21% of content words in our German-English dataset. Our experiments demonstrate that attention models by themselves cannot reliably explain the decisions made by a NMT model.

| Comments: | Accepted at the 3rd Workshop on Neural Generation and Translation (WNGT 2019) held at EMNLP-IJCNLP 2019 (Camera ready) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1910.00139 [cs.CL]**                                 |
|           | (or **arXiv:1910.00139v1 [cs.CL]** for this version)         |





<h2 id="2019-10-02-2">2. Improved Word Sense Disambiguation Using Pre-Trained Contextualized Word Representations</h2>
Title: [Improved Word Sense Disambiguation Using Pre-Trained Contextualized Word Representations](https://arxiv.org/abs/1910.00194)

Authors: [Christian Hadiwinoto](https://arxiv.org/search/cs?searchtype=author&query=Hadiwinoto%2C+C), [Hwee Tou Ng](https://arxiv.org/search/cs?searchtype=author&query=Ng%2C+H+T), [Wee Chung Gan](https://arxiv.org/search/cs?searchtype=author&query=Gan%2C+W+C)

*(Submitted on 1 Oct 2019)*

> Contextualized word representations are able to give different representations for the same word in different contexts, and they have been shown to be effective in downstream natural language processing tasks, such as question answering, named entity recognition, and sentiment analysis. However, evaluation on word sense disambiguation (WSD) in prior work shows that using contextualized word representations does not outperform the state-of-the-art approach that makes use of non-contextualized word embeddings. In this paper, we explore different strategies of integrating pre-trained contextualized word representations and our best strategy achieves accuracies exceeding the best prior published accuracies by significant margins on multiple benchmark WSD datasets.

| Comments: | 10 pages, 2 figures, EMNLP 2019                      |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1910.00194 [cs.CL]**                         |
|           | (or **arXiv:1910.00194v1 [cs.CL]** for this version) |





<h2 id="2019-10-02-3">3. Multilingual End-to-End Speech Translation</h2>
Title: [Multilingual End-to-End Speech Translation](https://arxiv.org/abs/1910.00254)

Authors: [Hirofumi Inaguma](https://arxiv.org/search/cs?searchtype=author&query=Inaguma%2C+H), [Kevin Duh](https://arxiv.org/search/cs?searchtype=author&query=Duh%2C+K), [Tatsuya Kawahara](https://arxiv.org/search/cs?searchtype=author&query=Kawahara%2C+T), [Shinji Watanabe](https://arxiv.org/search/cs?searchtype=author&query=Watanabe%2C+S)

*(Submitted on 1 Oct 2019)*

> In this paper, we propose a simple yet effective framework for multilingual end-to-end speech translation (ST), in which speech utterances in source languages are directly translated to the desired target languages with a universal sequence-to-sequence architecture. While multilingual models have shown to be useful for automatic speech recognition (ASR) and machine translation (MT), this is the first time they are applied to the end-to-end ST problem. We show the effectiveness of multilingual end-to-end ST in two scenarios: one-to-many and many-to-many translations with publicly available data. We experimentally confirm that multilingual end-to-end ST models significantly outperform bilingual ones in both scenarios. The generalization of multilingual training is also evaluated in a transfer learning scenario to a very low-resource language pair. All of our codes and the database are publicly available to encourage further research in this emergent multilingual ST topic.

| Comments: | Accepted to ASRU 2019                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Audio and Speech Processing (eess.AS) |
| Cite as:  | **arXiv:1910.00254 [cs.CL]**                                 |
|           | (or **arXiv:1910.00254v1 [cs.CL]** for this version)         |





<h2 id="2019-10-02-4">4. When and Why is Document-level Context Useful in Neural Machine Translation?</h2>
Title: [When and Why is Document-level Context Useful in Neural Machine Translation?](https://arxiv.org/abs/1910.00294)

Authors: [Yunsu Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+Y), [Duc Thanh Tran](https://arxiv.org/search/cs?searchtype=author&query=Tran%2C+D+T), [Hermann Ney](https://arxiv.org/search/cs?searchtype=author&query=Ney%2C+H)

*(Submitted on 1 Oct 2019)*

> Document-level context has received lots of attention for compensating neural machine translation (NMT) of isolated sentences. However, recent advances in document-level NMT focus on sophisticated integration of the context, explaining its improvement with only a few selected examples or targeted test sets. We extensively quantify the causes of improvements by a document-level model in general test sets, clarifying the limit of the usefulness of document-level context in NMT. We show that most of the improvements are not interpretable as utilizing the context. We also show that a minimal encoding is sufficient for the context modeling and very long context is not helpful for NMT.

| Comments: | DiscoMT 2019 camera-ready                            |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1910.00294 [cs.CL]**                         |
|           | (or **arXiv:1910.00294v1 [cs.CL]** for this version) |





<h2 id="2019-10-02-5">5. Grammatical Error Correction in Low-Resource Scenarios</h2>
Title: [Grammatical Error Correction in Low-Resource Scenarios](https://arxiv.org/abs/1910.00353)

Authors: [Jakub Náplava](https://arxiv.org/search/cs?searchtype=author&query=Náplava%2C+J), [Milan Straka](https://arxiv.org/search/cs?searchtype=author&query=Straka%2C+M)

*(Submitted on 1 Oct 2019 ([v1](https://arxiv.org/abs/1910.00353v1)), last revised 2 Oct 2019 (this version, v2))*

> Grammatical error correction in English is a long studied problem with many existing systems and datasets. However, there has been only a limited research on error correction of other languages. In this paper, we present a new dataset AKCES-GEC on grammatical error correction for Czech. We then make experiments on Czech, German and Russian and show that when utilizing synthetic parallel corpus, Transformer neural machine translation model can reach new state-of-the-art results on these datasets. AKCES-GEC is published under CC BY-NC-SA 4.0 license at [this https URL](https://hdl.handle.net/11234/1-3057) and the source code of the GEC model is available at [this https URL](https://github.com/ufal/low-resource-gec-wnut2019).

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1910.00353 [cs.CL]**                         |
|           | (or **arXiv:1910.00353v2 [cs.CL]** for this version) |





<h2 id="2019-10-02-6">6. Application of Low-resource Machine Translation Techniques to Russian-Tatar Language Pair</h2>
Title: [Application of Low-resource Machine Translation Techniques to Russian-Tatar Language Pair](https://arxiv.org/abs/1910.00368)

Authors: [Aidar Valeev](https://arxiv.org/search/cs?searchtype=author&query=Valeev%2C+A), [Ilshat Gibadullin](https://arxiv.org/search/cs?searchtype=author&query=Gibadullin%2C+I), [Albina Khusainova](https://arxiv.org/search/cs?searchtype=author&query=Khusainova%2C+A), [Adil Khan](https://arxiv.org/search/cs?searchtype=author&query=Khan%2C+A)

*(Submitted on 1 Oct 2019)*

> Neural machine translation is the current state-of-the-art in machine translation. Although it is successful in a resource-rich setting, its applicability for low-resource language pairs is still debatable. In this paper, we explore the effect of different techniques to improve machine translation quality when a parallel corpus is as small as 324 000 sentences, taking as an example previously unexplored Russian-Tatar language pair. We apply such techniques as transfer learning and semi-supervised learning to the base Transformer model, and empirically show that the resulting models improve Russian to Tatar and Tatar to Russian translation quality by +2.57 and +3.66 BLEU, respectively.

| Comments: | Presented on ICATHS'19                               |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1910.00368 [cs.CL]**                         |
|           | (or **arXiv:1910.00368v1 [cs.CL]** for this version) |





<h2 id="2019-10-02-7">7. A Survey of Methods to Leverage Monolingual Data in Low-resource Neural Machine Translation</h2>
Title: [A Survey of Methods to Leverage Monolingual Data in Low-resource Neural Machine Translation](https://arxiv.org/abs/1910.00373)

Authors: [Ilshat Gibadullin](https://arxiv.org/search/cs?searchtype=author&query=Gibadullin%2C+I), [Aidar Valeev](https://arxiv.org/search/cs?searchtype=author&query=Valeev%2C+A), [Albina Khusainova](https://arxiv.org/search/cs?searchtype=author&query=Khusainova%2C+A), [Adil Khan](https://arxiv.org/search/cs?searchtype=author&query=Khan%2C+A)

*(Submitted on 1 Oct 2019)*

> Neural machine translation has become the state-of-the-art for language pairs with large parallel corpora. However, the quality of machine translation for low-resource languages leaves much to be desired. There are several approaches to mitigate this problem, such as transfer learning, semi-supervised and unsupervised learning techniques. In this paper, we review the existing methods, where the main idea is to exploit the power of monolingual data, which, compared to parallel, is usually easier to obtain and significantly greater in amount.

| Comments: | Presented in ICATHS'19                               |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1910.00373 [cs.CL]**                         |
|           | (or **arXiv:1910.00373v1 [cs.CL]** for this version) |





<h2 id="2019-10-02-8">8. Machine Translation for Machines: the Sentiment Classification Use Case</h2>
Title: [Machine Translation for Machines: the Sentiment Classification Use Case](https://arxiv.org/abs/1910.00478)

Authors: [Amirhossein Tebbifakhr](https://arxiv.org/search/cs?searchtype=author&query=Tebbifakhr%2C+A), [Luisa Bentivogli](https://arxiv.org/search/cs?searchtype=author&query=Bentivogli%2C+L), [Matteo Negri](https://arxiv.org/search/cs?searchtype=author&query=Negri%2C+M), [Marco Turchi](https://arxiv.org/search/cs?searchtype=author&query=Turchi%2C+M)

*(Submitted on 1 Oct 2019)*

> We propose a neural machine translation (NMT) approach that, instead of pursuing adequacy and fluency ("human-oriented" quality criteria), aims to generate translations that are best suited as input to a natural language processing component designed for a specific downstream task (a "machine-oriented" criterion). Towards this objective, we present a reinforcement learning technique based on a new candidate sampling strategy, which exploits the results obtained on the downstream task as weak feedback. Experiments in sentiment classification of Twitter data in German and Italian show that feeding an English classifier with machine-oriented translations significantly improves its performance. Classification results outperform those obtained with translations produced by general-purpose NMT models as well as by an approach based on reinforcement learning. Moreover, our results on both languages approximate the classification accuracy computed on gold standard English tweets.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1910.00478 [cs.CL]**                         |
|           | (or **arXiv:1910.00478v1 [cs.CL]** for this version) |





<h2 id="2019-10-02-9">9. Putting Machine Translation in Context with the Noisy Channel Model</h2>
Title: [Putting Machine Translation in Context with the Noisy Channel Model](https://arxiv.org/abs/1910.00553)

Authors: [Lei Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+L), [Laurent Sartran](https://arxiv.org/search/cs?searchtype=author&query=Sartran%2C+L), [Wojciech Stokowiec](https://arxiv.org/search/cs?searchtype=author&query=Stokowiec%2C+W), [Wang Ling](https://arxiv.org/search/cs?searchtype=author&query=Ling%2C+W), [Lingpeng Kong](https://arxiv.org/search/cs?searchtype=author&query=Kong%2C+L), [Phil Blunsom](https://arxiv.org/search/cs?searchtype=author&query=Blunsom%2C+P), [Chris Dyer](https://arxiv.org/search/cs?searchtype=author&query=Dyer%2C+C)

*(Submitted on 1 Oct 2019)*

> We show that Bayes' rule provides a compelling mechanism for controlling unconditional document language models, using the long-standing challenge of effectively leveraging document context in machine translation. In our formulation, we estimate the probability of a candidate translation as the product of the unconditional probability of the candidate output document and the ``reverse translation probability'' of translating the candidate output back into the input source language document---the so-called ``noisy channel'' decomposition. A particular advantage of our model is that it requires only parallel sentences to train, rather than parallel documents, which are not always available. Using a new beam search reranking approximation to solve the decoding problem, we find that document language models outperform language models that assume independence between sentences, and that using either a document or sentence language model outperforms comparable models that directly estimate the translation probability. We obtain the best-published results on the NIST Chinese--English translation task, a standard task for evaluating document translation. Our model also outperforms the benchmark Transformer model by approximately 2.5 BLEU on the WMT19 Chinese--English translation task.

| Comments: | 14 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1910.00553 [cs.CL]**                                 |
|           | (or **arXiv:1910.00553v1 [cs.CL]** for this version)         |





# 2019-10-01

[Return to Index](#Index)



<h2 id="2019-10-01-1">1. Revisiting Self-Training for Neural Sequence Generation</h2> 
Title: [Revisiting Self-Training for Neural Sequence Generation](https://arxiv.org/abs/1909.13788)

Authors:[Junxian He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+J), [Jiatao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+J), [Jiajun Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+J), [Marc'Aurelio Ranzato](https://arxiv.org/search/cs?searchtype=author&query=Ranzato%2C+M)

*(Submitted on 30 Sep 2019)*

> Self-training is one of the earliest and simplest semi-supervised methods. The key idea is to augment the original labeled dataset with unlabeled data paired with the model's prediction (i.e. pseudo-parallel data). While self-training has been extensively studied on classification problems, in complex sequence generation tasks (e.g. machine translation) it is still unclear how self-training works due to the compositionality of the target space. In this work, we first empirically show that self-training is able to decently improve the supervised baseline on neural sequence generation tasks. Through careful examination of the performance gains, we find that the perturbation on the hidden states (i.e. dropout) is critical for self-training to benefit from the pseudo-parallel data, which acts as a regularizer and forces the model to yield close predictions for similar unlabeled inputs. Such effect helps the model correct some incorrect predictions on unlabeled data. To further encourage this mechanism, we propose to inject noise to the input space, resulting in a "noisy" version of self-training. Empirical study on standard machine translation and text summarization benchmarks shows that noisy self-training is able to effectively utilize unlabeled data and improve the performance of the supervised baseline by a large margin.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1909.13788 [cs.LG]**                                 |
|           | (or **arXiv:1909.13788v1 [cs.LG]** for this version)         |





<h2 id="2019-10-01-2">2. The Source-Target Domain Mismatch Problem in Machine Translation</h2> 
Title: [The Source-Target Domain Mismatch Problem in Machine Translation](https://arxiv.org/abs/1909.13151)

Authors:[Jiajun Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+J), [Peng-Jen Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+P), [Matt Le](https://arxiv.org/search/cs?searchtype=author&query=Le%2C+M), [Junxian He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+J), [Jiatao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+J), [Myle Ott](https://arxiv.org/search/cs?searchtype=author&query=Ott%2C+M), [Michael Auli](https://arxiv.org/search/cs?searchtype=author&query=Auli%2C+M), [Marc'Aurelio Ranzato](https://arxiv.org/search/cs?searchtype=author&query=Ranzato%2C+M)

*(Submitted on 28 Sep 2019)*

> While we live in an increasingly interconnected world, different places still exhibit strikingly different cultures and many events we experience in our every day life pertain only to the specific place we live in. As a result, people often talk about different things in different parts of the world. In this work we study the effect of local context in machine translation and postulate that particularly in low resource settings this causes the domains of the source and target language to greatly mismatch, as the two languages are often spoken in further apart regions of the world with more distinctive cultural traits and unrelated local events. In this work we first propose a controlled setting to carefully analyze the source-target domain mismatch, and its dependence on the amount of parallel and monolingual data. Second, we test both a model trained with back-translation and one trained with self-training. The latter leverages in-domain source monolingual data but uses potentially incorrect target references. We found that these two approaches are often complementary to each other. For instance, on a low-resource Nepali-English dataset the combined approach improves upon the baseline using just parallel data by 2.5 BLEU points, and by 0.6 BLEU point when compared to back-translation.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1909.13151 [cs.CL]**                         |
|           | (or **arXiv:1909.13151v1 [cs.CL]** for this version) |





<h2 id="2019-10-01-3">3. Controllable Data Synthesis Method for Grammatical Error Correction</h2> 
Title: [Controllable Data Synthesis Method for Grammatical Error Correction](https://arxiv.org/abs/1909.13302)

Authors:[Chencheng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Liner Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+L), [Yun Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Yongping Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+Y), [Erhong Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+E)

*(Submitted on 29 Sep 2019 ([v1](https://arxiv.org/abs/1909.13302v1)), last revised 2 Oct 2019 (this version, v2))*

> Due to the lack of parallel data in current Grammatical Error Correction (GEC) task, models based on Sequence to Sequence framework cannot be adequately trained to obtain higher performance. We propose two data synthesis methods which can control the error rate and the ratio of error types on synthetic data. The first approach is to corrupt each word in the monolingual corpus with a fixed probability, including replacement, insertion and deletion. Another approach is to train error generation models and further filtering the decoding results of the models. The experiments on different synthetic data show that the error rate is 40% and the ratio of error types is the same can improve the model performance better. Finally, we synthesize about 100 million data and achieve comparable performance as the state of the art, which uses twice as much data as we use.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1909.13302 [cs.CL]**                         |
|           | (or **arXiv:1909.13302v2 [cs.CL]** for this version) |





<h2 id="2019-10-01-4">4. Regressing Word and Sentence Embeddings for Regularization of Neural Machine Translation</h2> 
Title: [Regressing Word and Sentence Embeddings for Regularization of Neural Machine Translation](https://arxiv.org/abs/1909.13466)

Authors: [Inigo Jauregi Unanue](https://arxiv.org/search/cs?searchtype=author&query=Unanue%2C+I+J), [Ehsan Zare Borzeshi](https://arxiv.org/search/cs?searchtype=author&query=Borzeshi%2C+E+Z), [Massimo Piccardi](https://arxiv.org/search/cs?searchtype=author&query=Piccardi%2C+M)

*(Submitted on 30 Sep 2019)*

> In recent years, neural machine translation (NMT) has become the dominant approach in automated translation. However, like many other deep learning approaches, NMT suffers from overfitting when the amount of training data is limited. This is a serious issue for low-resource language pairs and many specialized translation domains that are inherently limited in the amount of available supervised data. For this reason, in this paper we propose regressing word (ReWE) and sentence (ReSE) embeddings at training time as a way to regularize NMT models and improve their generalization. During training, our models are trained to jointly predict categorical (words in the vocabulary) and continuous (word and sentence embeddings) outputs. An extensive set of experiments over four language pairs of variable training set size has showed that ReWE and ReSE can outperform strong state-of-the-art baseline models, with an improvement that is larger for smaller training sets (e.g., up to +5:15 BLEU points in Basque-English translation). Visualizations of the decoder's output space show that the proposed regularizers improve the clustering of unique words, facilitating correct predictions. In a final experiment on unsupervised NMT, we show that ReWE and ReSE are also able to improve the quality of machine translation when no parallel data are available.

| Comments: | \c{opyright} 2019 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1909.13466 [cs.CL]**                                 |
|           | (or **arXiv:1909.13466v1 [cs.CL]** for this version)         |





<h2 id="2019-10-01-5">5. Simple and Effective Paraphrastic Similarity from Parallel Translations</h2> 
Title: [Simple and Effective Paraphrastic Similarity from Parallel Translations](https://arxiv.org/abs/1909.13872)

Authors: [John Wieting](https://arxiv.org/search/cs?searchtype=author&query=Wieting%2C+J), [Kevin Gimpel](https://arxiv.org/search/cs?searchtype=author&query=Gimpel%2C+K), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G), [Taylor Berg-Kirkpatrick](https://arxiv.org/search/cs?searchtype=author&query=Berg-Kirkpatrick%2C+T)

*(Submitted on 30 Sep 2019)*

> We present a model and methodology for learning paraphrastic sentence embeddings directly from bitext, removing the time-consuming intermediate step of creating paraphrase corpora. Further, we show that the resulting model can be applied to cross-lingual tasks where it both outperforms and is orders of magnitude faster than more complex state-of-the-art baselines.

| Comments: | Published as a short paper at ACL 2019               |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.13872 [cs.CL]**                         |
|           | (or **arXiv:1909.13872v1 [cs.CL]** for this version) |

