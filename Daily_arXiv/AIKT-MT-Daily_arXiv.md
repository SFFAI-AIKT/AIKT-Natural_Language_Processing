# Daily arXiv: Machine Translation - Apr., 2019

### Index

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