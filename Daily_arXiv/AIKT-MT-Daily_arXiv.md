# Daily arXiv: Machine Translation - Apr., 2019

### Index

- [2019-04-08](#2019-04-08)
  - [1. Modeling Recurrence for Transformer](#2019-04-08-1)
  - [2. Convolutional Self-Attention Networks](#2019-04-08-2)
- [2019-04-05](#2019-04-05)
  - [1. Consistency by Agreement in Zero-shot Neural Machine Translation](#2019-04-05-1)
  - [2. The Effect of Downstream Classification Tasks for Evaluating Sentence Embeddings](#2019-04-05-2)
  - [3. Density Matching for Bilingual Word Embedding](#2019-04-05-3)
  - [4. ReWE: Regressing Word Embeddings for Regularization of Neural Machine Translation Systems](#2019-04-05-4)
- [2019-04-04](#2019-04-04)
  - [1. Attentive Mimicking: Better Word Embeddings by Attending to Informative Contexts](#2019-04-04-1)
  - [2. Identification, Interpretability, and Bayesian Word Embeddings](#2019-04-04-2)
  - [3. Cross-lingual transfer learning for spoken language understanding](#2019-04-04-3)
  - [4. Modeling Vocabulary for Big Code Machine Learning](#2019-04-04-4)
- [2019-04-03](#2019-04-03)
  - [1. Learning to Stop in Structured Prediction for Neural Machine Translation](#2019-04-03-1)
  - [2. A Multi-Task Approach for Disentangling Syntax and Semantics in Sentence Representations](#2019-04-03-2)
  - [3. Pragmatically Informative Text Generation](#2019-04-03-3)
  - [4. Using Multi-Sense Vector Embeddings for Reverse Dictionaries](#2019-04-03-4)
  - [5. Neural Vector Conceptualization for Word Vector Space Interpretation](#2019-04-03-5)
- [2019-04-02](#2019-04-02)
  - [1. Machine translation considering context information using Encoder-Decoder model](#2019-04-01-1)
  - [2. Multimodal Machine Translation with Embedding Prediction](#2019-04-02-2)
  - [3. Lost in Interpretation: Predicting Untranslated Terminology in Simultaneous Interpretation](#2019-04-02-3)

* [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)
* [2019-02](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-02.md)



# 2019-04-08

[Return to Index](#Index)

<h2 id="2019-04-08-1">1. Modeling Recurrence for Transformer</h2> 

Title： [Modeling Recurrence for Transformer](<https://arxiv.org/abs/1904.03092>)

Authors: [Jie Hao](https://arxiv.org/search/cs?searchtype=author&query=Hao%2C+J), [Xing Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Baosong Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+B), [Longyue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Jinfeng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+J), [Zhaopeng Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+Z)

*(Submitted on 5 Apr 2019)*

> Recently, the Transformer model that is based solely on attention mechanisms, has advanced the state-of-the-art on various machine translation tasks. However, recent studies reveal that the lack of recurrence hinders its further improvement of translation capacity. In response to this problem, we propose to directly model recurrence for Transformer with an additional recurrence encoder. In addition to the standard recurrent neural network, we introduce a novel attentive recurrent network to leverage the strengths of both attention and recurrent networks. Experimental results on the widely-used WMT14 English-German and WMT17 Chinese-English translation tasks demonstrate the effectiveness of the proposed approach. Our studies also reveal that the proposed model benefits from a short-cut that bridges the source and target sequences with a single recurrent layer, which outperforms its deep counterpart.

| Comments: | NAACL 2019                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | [arXiv:1904.03092](https://arxiv.org/abs/1904.03092) [cs.CL] |
|           | (or **arXiv:1904.03092v1 [cs.CL]** for this version)         |



<h2 id="2019-04-08-2">2. Convolutional Self-Attention Networks</h2> 

Title: [Convolutional Self-Attention Networks](<https://arxiv.org/abs/1904.03107>)

Authors: [Baosong Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+B), [Longyue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Derek Wong](https://arxiv.org/search/cs?searchtype=author&query=Wong%2C+D), [Lidia S. Chao](https://arxiv.org/search/cs?searchtype=author&query=Chao%2C+L+S), [Zhaopeng Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+Z)

*(Submitted on 5 Apr 2019)*

> Self-attention networks (SANs) have drawn increasing interest due to their high parallelization in computation and flexibility in modeling dependencies. SANs can be further enhanced with multi-head attention by allowing the model to attend to information from different representation subspaces. In this work, we propose novel convolutional self-attention networks, which offer SANs the abilities to 1) strengthen dependencies among neighboring elements, and 2) model the interaction between features extracted by multiple attention heads. Experimental results of machine translation on different language pairs and model settings show that our approach outperforms both the strong Transformer baseline and other existing models on enhancing the locality of SANs. Comparing with prior studies, the proposed model is parameter free in terms of introducing no more parameters.

| Comments: | NAACL 2019                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | [arXiv:1904.03107](https://arxiv.org/abs/1904.03107) [cs.CL] |
|           | (or **arXiv:1904.03107v1 [cs.CL]** for this version)         |



# 2019-04-05 

[Return to Index](#Index)

<h2 id="2019-04-05-1">1. Consistency by Agreement in Zero-shot Neural Machine Translation</h2> 

Title: [Consistency by Agreement in Zero-shot Neural Machine Translation](<https://arxiv.org/abs/1904.02338>)

Authors: [Maruan Al-Shedivat](https://arxiv.org/search/cs?searchtype=author&query=Al-Shedivat%2C+M), [Ankur P. Parikh](https://arxiv.org/search/cs?searchtype=author&query=Parikh%2C+A+P)

*(Submitted on 4 Apr 2019)*

> Generalization and reliability of multilingual translation often highly depend on the amount of available parallel data for each language pair of interest. In this paper, we focus on zero-shot generalization---a challenging setup that tests models on translation directions they have not been optimized for at training time. To solve the problem, we (i) reformulate multilingual translation as probabilistic inference, (ii) define the notion of zero-shot consistency and show why standard training often results in models unsuitable for zero-shot tasks, and (iii) introduce a consistent agreement-based training method that encourages the model to produce equivalent translations of parallel sentences in auxiliary languages. We test our multilingual NMT models on multiple public zero-shot translation benchmarks (IWSLT17, UN corpus, Europarl) and show that agreement-based learning often results in 2-3 BLEU zero-shot improvement over strong baselines without any loss in performance on supervised translation directions.

| Comments: | NAACL 2019                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Neural and Evolutionary Computing (cs.NE); Machine Learning (stat.ML) |
| Cite as:  | [arXiv:1904.02338](https://arxiv.org/abs/1904.02338) [cs.LG] |
|           | (or **arXiv:1904.02338v1 [cs.LG]** for this version)         |



<h2 id="2019-04-05-2">2. The Effect of Downstream Classification Tasks for Evaluating Sentence Embeddings</h2> 

Title: [The Effect of Downstream Classification Tasks for Evaluating Sentence Embeddings](<https://arxiv.org/abs/1904.02228>)

Authors: [Peter Potash](https://arxiv.org/search/cs?searchtype=author&query=Potash%2C+P)

*(Submitted on 3 Apr 2019)*

> One popular method for quantitatively evaluating the performance of sentence embeddings involves their usage on downstream language processing tasks that require sentence representations as input. One simple such task is classification, where the sentence representations are used to train and test models on several classification datasets. We argue that by evaluating sentence representations in such a manner, the goal of the representations becomes learning a low-dimensional factorization of a sentence-task label matrix. We show how characteristics of this matrix can affect the ability for a low-dimensional factorization to perform as sentence representations in a suite of classification tasks. Primarily, sentences that have more labels across all possible classification tasks have a higher reconstruction loss, though this effect can be drastically negated if the amount of such sentences is small.

| Comments: | 5 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.02228](https://arxiv.org/abs/1904.02228) [cs.CL] |
|           | (or **arXiv:1904.02228v1 [cs.CL]** for this version)         |



<h2 id="2019-04-05-3">3. Density Matching for Bilingual Word Embedding</h2> 

Title： [Density Matching for Bilingual Word Embedding](<https://arxiv.org/abs/1904.02343>)

Authors: [Chunting Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+C), [Xuezhe Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+X), [Di Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+D), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G)

*(Submitted on 4 Apr 2019)*

> Recent approaches to cross-lingual word embedding have generally been based on linear transformations between the sets of embedding vectors in the two languages. In this paper, we propose an approach that instead expresses the two monolingual embedding spaces as probability densities defined by a Gaussian mixture model, and matches the two densities using a method called normalizing flow. The method requires no explicit supervision, and can be learned with only a seed dictionary of words that have identical strings. We argue that this formulation has several intuitively attractive properties, particularly with the respect to improving robustness and generalization to mappings between difficult language pairs or word pairs. On a benchmark data set of bilingual lexicon induction and cross-lingual word similarity, our approach can achieve competitive or superior performance compared to state-of-the-art published results, with particularly strong results being found on etymologically distant and/or morphologically rich languages.

| Comments: | Accepted by NAACL-HLT 2019                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1904.02343](https://arxiv.org/abs/1904.02343) [cs.CL] |
|           | (or **arXiv:1904.02343v1 [cs.CL]** for this version)         |



<h2 id="2019-04-05-4">4. ReWE: Regressing Word Embeddings for Regularization of Neural Machine Translation Systems</h2> 

Title: [ReWE: Regressing Word Embeddings for Regularization of Neural Machine Translation Systems](<https://arxiv.org/abs/1904.02461>)

Authors: [Inigo Jauregi Unanue](https://arxiv.org/search/cs?searchtype=author&query=Unanue%2C+I+J), [Ehsan Zare Borzeshi](https://arxiv.org/search/cs?searchtype=author&query=Borzeshi%2C+E+Z), [Nazanin Esmaili](https://arxiv.org/search/cs?searchtype=author&query=Esmaili%2C+N), [Massimo Piccardi](https://arxiv.org/search/cs?searchtype=author&query=Piccardi%2C+M)

*(Submitted on 4 Apr 2019)*

> Regularization of neural machine translation is still a significant problem, especially in low-resource settings. To mollify this problem, we propose regressing word embeddings (ReWE) as a new regularization technique in a system that is jointly trained to predict the next word in the translation (categorical value) and its word embedding (continuous value). Such a joint training allows the proposed system to learn the distributional properties represented by the word embeddings, empirically improving the generalization to unseen sentences. Experiments over three translation datasets have showed a consistent improvement over a strong baseline, ranging between 0.91 and 2.54 BLEU points, and also a marked improvement over a state-of-the-art system.

| Comments: | Accepted at NAACL-HLT 2019                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.02461](https://arxiv.org/abs/1904.02461) [cs.CL] |
|           | (or **arXiv:1904.02461v1 [cs.CL]** for this version)         |



# 2019-04-04

[Return to Index](#Index)

<h2 id="2019-04-04-1">1. Attentive Mimicking: Better Word Embeddings by Attending to Informative Contexts</h2> 

Title: [Attentive Mimicking: Better Word Embeddings by Attending to Informative Contexts](<https://arxiv.org/abs/1904.01617>)

Authors: [Timo Schick](https://arxiv.org/search/cs?searchtype=author&query=Schick%2C+T), [Hinrich Schütze](https://arxiv.org/search/cs?searchtype=author&query=Sch%C3%BCtze%2C+H)

*(Submitted on 2 Apr 2019)*

> Learning high-quality embeddings for rare words is a hard problem because of sparse context information. Mimicking (Pinter et al., 2017) has been proposed as a solution: given embeddings learned by a standard algorithm, a model is first trained to reproduce embeddings of frequent words from their surface form and then used to compute embeddings for rare words. In this paper, we introduce attentive mimicking: the mimicking model is given access not only to a word's surface form, but also to all available contexts and learns to attend to the most informative and reliable contexts for computing an embedding. In an evaluation on four tasks, we show that attentive mimicking outperforms previous work for both rare and medium-frequency words. Thus, compared to previous work, attentive mimicking improves embeddings for a much larger part of the vocabulary, including the medium-frequency range.

| Comments: | Accepted at NAACL2019                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.01617](https://arxiv.org/abs/1904.01617) [cs.CL] |
|           | (or **arXiv:1904.01617v1 [cs.CL]** for this version)         |



<h2 id="2019-04-04-2">2. Identification, Interpretability, and Bayesian Word Embeddings</h2> 

Title: [Identification, Interpretability, and Bayesian Word Embeddings](<https://arxiv.org/abs/1904.01628>)

Authors: [Adam M. Lauretig](https://arxiv.org/search/cs?searchtype=author&query=Lauretig%2C+A+M)

*(Submitted on 2 Apr 2019)*

> Social scientists have recently turned to analyzing text using tools from natural language processing like word embeddings to measure concepts like ideology, bias, and affinity. However, word embeddings are difficult to use in the regression framework familiar to social scientists: embeddings are are neither identified, nor directly interpretable. I offer two advances on standard embedding models to remedy these problems. First, I develop Bayesian Word Embeddings with Automatic Relevance Determination priors, relaxing the assumption that all embedding dimensions have equal weight. Second, I apply work identifying latent variable models to anchor the dimensions of the resulting embeddings, identifying them, and making them interpretable and usable in a regression. I then apply this model and anchoring approach to two cases, the shift in internationalist rhetoric in the American presidents' inaugural addresses, and the relationship between bellicosity in American foreign policy decision-makers' deliberations. I find that inaugural addresses became less internationalist after 1945, which goes against the conventional wisdom, and that an increase in bellicosity is associated with an increase in hostile actions by the United States, showing that elite deliberations are not cheap talk, and helping confirm the validity of the model.

| Comments: | Accepted to the Third Workshop on Natural Language Processing and Computational Social Science at NAACL-HLT 2019 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Applications (stat.AP) |
| Cite as:  | [arXiv:1904.01628](https://arxiv.org/abs/1904.01628) [cs.CL] |
|           | (or **arXiv:1904.01628v1 [cs.CL]** for this version)         |

<h2 id="2019-04-04-3">3. Cross-lingual transfer learning for spoken language understanding</h2> 

Title: [Cross-lingual transfer learning for spoken language understanding](<https://arxiv.org/abs/1904.01825>)

Authors: [Quynh Ngoc Thi Do](https://arxiv.org/search/cs?searchtype=author&query=Do%2C+Q+N+T), [Judith Gaspers](https://arxiv.org/search/cs?searchtype=author&query=Gaspers%2C+J)

*(Submitted on 3 Apr 2019)*

> Typically, spoken language understanding (SLU) models are trained on annotated data which are costly to gather. Aiming to reduce data needs for bootstrapping a SLU system for a new language, we present a simple but effective weight transfer approach using data from another language. The approach is evaluated with our promising multi-task SLU framework developed towards different languages. We evaluate our approach on the ATIS and a real-world SLU dataset, showing that i) our monolingual models outperform the state-of-the-art, ii) we can reduce data amounts needed for bootstrapping a SLU system for a new language greatly, and iii) while multitask training improves over separate training, different weight transfer settings may work best for different SLU modules.

| Comments: | accepted at ICASSP, 2019                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.01825](https://arxiv.org/abs/1904.01825) [cs.CL] |
|           | (or **arXiv:1904.01825v1 [cs.CL]** for this version)         |



<h2 id="2019-04-04-4">4. Modeling Vocabulary for Big Code Machine Learning</h2> 

Title: [Modeling Vocabulary for Big Code Machine Learning](<https://arxiv.org/abs/1904.01873>)

Authors: [Hlib Babii](https://arxiv.org/search/cs?searchtype=author&query=Babii%2C+H), [Andrea Janes](https://arxiv.org/search/cs?searchtype=author&query=Janes%2C+A), [Romain Robbes](https://arxiv.org/search/cs?searchtype=author&query=Robbes%2C+R)

*(Submitted on 3 Apr 2019)*

> When building machine learning models that operate on source code, several decisions have to be made to model source-code vocabulary. These decisions can have a large impact: some can lead to not being able to train models at all, others significantly affect performance, particularly for Neural Language Models. Yet, these decisions are not often fully described. This paper lists important modeling choices for source code vocabulary, and explores their impact on the resulting vocabulary on a large-scale corpus of 14,436 projects. We show that a subset of decisions have decisive characteristics, allowing to train accurate Neural Language Models quickly on a large corpus of 10,106 projects.

| Comments: | 12 pages, 1 figure                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Software Engineering (cs.SE) |
| Cite as:  | [arXiv:1904.01873](https://arxiv.org/abs/1904.01873) [cs.CL] |
|           | (or **arXiv:1904.01873v1 [cs.CL]** for this version)         |



# 2019-04-03

[Return to Index](#Index)

<h2 id="2019-04-03-1">1. Learning to Stop in Structured Prediction for Neural Machine Translation</h2> 

Title: [Learning to Stop in Structured Prediction for Neural Machine Translation](<https://arxiv.org/abs/1904.01032>)

Authors: [Mingbo Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+M), [Renjie Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+R), [Liang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+L)

*(Submitted on 1 Apr 2019)*

> Beam search optimization resolves many issues in neural machine translation. However, this method lacks principled stopping criteria and does not learn how to stop during training, and the model naturally prefers the longer hypotheses during the testing time in practice since they use the raw score instead of the probability-based score. We propose a novel ranking method which enables an optimal beam search stopping criteria. We further introduce a structured prediction loss function which penalizes suboptimal finished candidates produced by beam search during training. Experiments of neural machine translation on both synthetic data and real languages (German-to-English and Chinese-to-English) demonstrate our proposed methods lead to better length and BLEU score.

| Comments:          | 5 pages                                                      |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**                         |
| Journal reference: | NAACL 2019                                                   |
| Cite as:           | [arXiv:1904.01032](https://arxiv.org/abs/1904.01032) [cs.CL] |
|                    | (or **arXiv:1904.01032v1 [cs.CL]** for this version)         |



<h2 id="2019-04-03-2">2. A Multi-Task Approach for Disentangling Syntax and Semantics in Sentence Representations</h2> 

Title: [A Multi-Task Approach for Disentangling Syntax and Semantics in Sentence Representations](<https://arxiv.org/abs/1904.01173>)

Authors: [Mingda Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+M), [Qingming Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+Q), [Sam Wiseman](https://arxiv.org/search/cs?searchtype=author&query=Wiseman%2C+S), [Kevin Gimpel](https://arxiv.org/search/cs?searchtype=author&query=Gimpel%2C+K)

*(Submitted on 2 Apr 2019)*

> We propose a generative model for a sentence that uses two latent variables, with one intended to represent the syntax of the sentence and the other to represent its semantics. We show we can achieve better disentanglement between semantic and syntactic representations by training with multiple losses, including losses that exploit aligned paraphrastic sentences and word-order information. We also investigate the effect of moving from bag-of-words to recurrent neural network modules. We evaluate our models as well as several popular pretrained embeddings on standard semantic similarity tasks and novel syntactic similarity tasks. Empirically, we find that the model with the best performing syntactic and semantic representations also gives rise to the most disentangled representations.

| Comments:          | NAACL 2019 Long paper                                        |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**                         |
| Journal reference: | NAACL 2019                                                   |
| Cite as:           | [arXiv:1904.01173](https://arxiv.org/abs/1904.01173) [cs.CL] |
|                    | (or **arXiv:1904.01173v1 [cs.CL]** for this version)         |



<h2 id="2019-04-03-3">3. Pragmatically Informative Text Generation</h2> 

Title: [Pragmatically Informative Text Generation](<https://arxiv.org/abs/1904.01301>)

Authors: [Sheng Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+S), [Daniel Fried](https://arxiv.org/search/cs?searchtype=author&query=Fried%2C+D), [Jacob Andreas](https://arxiv.org/search/cs?searchtype=author&query=Andreas%2C+J), [Dan Klein](https://arxiv.org/search/cs?searchtype=author&query=Klein%2C+D)

*(Submitted on 2 Apr 2019)*

> We improve the informativeness of models for conditional text generation using techniques from computational pragmatics. These techniques formulate language production as a game between speakers and listeners, in which a speaker should generate output text that a listener can use to correctly identify the original input that the text describes. While such approaches are widely used in cognitive science and grounded language learning, they have received less attention for more standard language generation tasks. We consider two pragmatic modeling methods for text generation: one where pragmatics is imposed by information preservation, and another where pragmatics is imposed by explicit modeling of distractors. We find that these methods improve the performance of strong existing systems for abstractive summarization and generation from structured meaning representations.

| Comments: | 8 pages. accepted as a conference paper at NAACL2019 (short paper) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.01301](https://arxiv.org/abs/1904.01301) [cs.CL] |
|           | (or **arXiv:1904.01301v1 [cs.CL]** for this version)         |



<h2 id="2019-04-03-4">4. Using Multi-Sense Vector Embeddings for Reverse Dictionaries</h2> 

Title: [Using Multi-Sense Vector Embeddings for Reverse Dictionaries](<https://arxiv.org/abs/1904.01451>)

Authors: [Michael A. Hedderich](https://arxiv.org/search/cs?searchtype=author&query=Hedderich%2C+M+A), [Andrew Yates](https://arxiv.org/search/cs?searchtype=author&query=Yates%2C+A), [Dietrich Klakow](https://arxiv.org/search/cs?searchtype=author&query=Klakow%2C+D), [Gerard de Melo](https://arxiv.org/search/cs?searchtype=author&query=de+Melo%2C+G)

*(Submitted on 2 Apr 2019)*

> Popular word embedding methods such as word2vec and GloVe assign a single vector representation to each word, even if a word has multiple distinct meanings. Multi-sense embeddings instead provide different vectors for each sense of a word. However, they typically cannot serve as a drop-in replacement for conventional single-sense embeddings, because the correct sense vector needs to be selected for each word. In this work, we study the effect of multi-sense embeddings on the task of reverse dictionaries. We propose a technique to easily integrate them into an existing neural network architecture using an attention mechanism. Our experiments demonstrate that large improvements can be obtained when employing multi-sense embeddings both in the input sequence as well as for the target representation. An analysis of the sense distributions and of the learned attention is provided as well.

| Comments: | Accepted as long paper at the 13th International Conference on Computational Semantics (IWCS 2019) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1904.01451](https://arxiv.org/abs/1904.01451) [cs.CL] |
|           | (or **arXiv:1904.01451v1 [cs.CL]** for this version)         |



<h2 id="2019-04-03-5">5. Neural Vector Conceptualization for Word Vector Space Interpretation</h2> 

Title: [Neural Vector Conceptualization for Word Vector Space Interpretation](<https://arxiv.org/abs/1904.01500>)

Authors: [Robert Schwarzenberg](https://arxiv.org/search/cs?searchtype=author&query=Schwarzenberg%2C+R), [Lisa Raithel](https://arxiv.org/search/cs?searchtype=author&query=Raithel%2C+L), [David Harbecke](https://arxiv.org/search/cs?searchtype=author&query=Harbecke%2C+D)

*(Submitted on 2 Apr 2019)*

> Distributed word vector spaces are considered hard to interpret which hinders the understanding of natural language processing (NLP) models. In this work, we introduce a new method to interpret arbitrary samples from a word vector space. To this end, we train a neural model to conceptualize word vectors, which means that it activates higher order concepts it recognizes in a given vector. Contrary to prior approaches, our model operates in the original vector space and is capable of learning non-linear relations between word vectors and concepts. Furthermore, we show that it produces considerably less entropic concept activation profiles than the popular cosine similarity.

| Comments: | NAACL-HLT 2019 Workshop on Evaluating Vector Space Representations for NLP (RepEval) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1904.01500](https://arxiv.org/abs/1904.01500) [cs.CL] |
|           | (or **arXiv:1904.01500v1 [cs.CL]** for this version)         |



# 2019-04-02

[Return to Index](#Index)

<h2 id="2019-04-02-1">1. Machine translation considering context information using Encoder-Decoder model</h2> 

Title: [Machine translation considering context information using Encoder-Decoder model](<https://arxiv.org/abs/1904.00160>)

Authors:[Tetsuto Takano](https://arxiv.org/search/cs?searchtype=author&query=Takano%2C+T), [Satoshi Yamane](https://arxiv.org/search/cs?searchtype=author&query=Yamane%2C+S)

(Submitted on 30 Mar 2019)

> In the task of machine translation, context information is one of the important factor. But considering the context information model dose not proposed. The paper propose a new model which can integrate context information and make translation. In this paper, we create a new model based Encoder Decoder model. When translating current sentence, the model integrates output from preceding encoder with current encoder. The model can consider context information and the result score is higher than existing model.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1904.00160](https://arxiv.org/abs/1904.00160) [cs.CL] |
|           | (or **arXiv:1904.00160v1 [cs.CL]** for this version)         |



<h2 id="2019-04-02-2">2. Multimodal Machine Translation with Embedding Prediction</h2> 

Title: [Multimodal Machine Translation with Embedding Prediction](<https://arxiv.org/abs/1904.00639>)

Authors: [Tosho Hirasawa](https://arxiv.org/search/cs?searchtype=author&query=Hirasawa%2C+T), [Hayahide Yamagishi](https://arxiv.org/search/cs?searchtype=author&query=Yamagishi%2C+H), [Yukio Matsumura](https://arxiv.org/search/cs?searchtype=author&query=Matsumura%2C+Y), [Mamoru Komachi](https://arxiv.org/search/cs?searchtype=author&query=Komachi%2C+M)

(Submitted on 1 Apr 2019)

> Multimodal machine translation is an attractive application of neural machine translation (NMT). It helps computers to deeply understand visual objects and their relations with natural languages. However, multimodal NMT systems suffer from a shortage of available training data, resulting in poor performance for translating rare words. In NMT, pretrained word embeddings have been shown to improve NMT of low-resource domains, and a search-based approach is proposed to address the rare word problem. In this study, we effectively combine these two approaches in the context of multimodal NMT and explore how we can take full advantage of pretrained word embeddings to better translate rare words. We report overall performance improvements of 1.24 METEOR and 2.49 BLEU and achieve an improvement of 7.67 F-score for rare word translation.

| Comments: | 6 pages; NAACL 2019 Student Research Workshop                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.00639](https://arxiv.org/abs/1904.00639) [cs.CL] |
|           | (or **arXiv:1904.00639v1 [cs.CL]** for this version)         |



<h2 id="2019-04-02-3">3. Lost in Interpretation: Predicting Untranslated Terminology in Simultaneous Interpretation</h2> 

Title: [Lost in Interpretation: Predicting Untranslated Terminology in Simultaneous Interpretation](<https://arxiv.org/abs/1904.00930>)

Authors: [Nikolai Vogler](https://arxiv.org/search/cs?searchtype=author&query=Vogler%2C+N), [Craig Stewart](https://arxiv.org/search/cs?searchtype=author&query=Stewart%2C+C), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G)

(Submitted on 1 Apr 2019)

> Simultaneous interpretation, the translation of speech from one language to another in real-time, is an inherently difficult and strenuous task. One of the greatest challenges faced by interpreters is the accurate translation of difficult terminology like proper names, numbers, or other entities. Intelligent computer-assisted interpreting (CAI) tools that could analyze the spoken word and detect terms likely to be untranslated by an interpreter could reduce translation error and improve interpreter performance. In this paper, we propose a task of predicting which terminology simultaneous interpreters will leave untranslated, and examine methods that perform this task using supervised sequence taggers. We describe a number of task-specific features explicitly designed to indicate when an interpreter may struggle with translating a word. Experimental results on a newly-annotated version of the NAIST Simultaneous Translation Corpus (Shimizu et al., 2014) indicate the promise of our proposed method.

| Comments: | NAACL 2019                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.00930](https://arxiv.org/abs/1904.00930) [cs.CL] |
|           | (or **arXiv:1904.00930v1 [cs.CL]** for this version)         |