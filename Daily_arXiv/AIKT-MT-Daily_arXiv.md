# Daily arXiv: Machine Translation - Nov., 2019

# Index

- [2019-11-07](#2019-11-07)
  - [1. Fast Transformer Decoding: One Write-Head is All You Need](#2019-11-07-1)
  - [2. Unsupervised Cross-lingual Representation Learning at Scale](#2019-11-07-2)
  - [3. Guiding Non-Autoregressive Neural Machine Translation Decoding with Reordering Information](#2019-11-07-3)
- [2019-11-06](#2019-11-06)
  - [1. Training Neural Machine Translation (NMT) Models using Tensor Train Decomposition on TensorFlow (T3F)](#2019-11-06-1)
  - [2. Emerging Cross-lingual Structure in Pretrained Language Models](#2019-11-06-2)
  - [3. On Compositionality in Neural Machine Translation](#2019-11-06-3)
  - [4. Improving Bidirectional Decoding with Dynamic Target Semantics in Neural Machine Translation](#2019-11-06-4)
  - [5. Adversarial Language Games for Advanced Natural Language Intelligence](#2019-11-06-5)
  - [6. Data Diversification: An Elegant Strategy For Neural Machine Translation](#2019-11-06-6)
- [2019-11-05](#2019-11-05)
  - [1. Attributed Sequence Embedding](#2019-11-05-1)
  - [2. Machine Translation Evaluation using Bi-directional Entailment](#2019-11-05-2)
  - [3. Controlling Text Complexity in Neural Machine Translation](#2019-11-05-3)
  - [4. Machine Translation in Pronunciation Space](#2019-11-05-4)
  - [5. Analysing Coreference in Transformer Outputs](#2019-11-05-5)
  - [6. Ordering Matters: Word Ordering Aware Unsupervised NMT](#2019-11-05-6)
- [2019-11-04](#2019-11-04)
  - [1. Neural Cross-Lingual Relation Extraction Based on Bilingual Word Embedding Mapping](#2019-11-04-1)
  - [2. Sequence Modeling with Unconstrained Generation Order](#2019-11-04-2)
  - [3. On the Linguistic Representational Power of Neural Machine Translation Models](#2019-11-04-3)
- [2019-11-01](#2019-11-01)
  - [1. Fill in the Blanks: Imputing Missing Sentences for Larger-Context Neural Machine Translation](#2019-11-01-1)
  - [2. Document-level Neural Machine Translation with Inter-Sentence Attention](#2019-11-01-2)
  - [3. Naver Labs Europe's Systems for the Document-Level Generation and Translation Task at WNGT 2019](#2019-11-01-3)
  - [4. Machine Translation of Restaurant Reviews: New Corpus for Domain Adaptation and Robustness](#2019-11-01-4)
  - [5. Adversarial NLI: A New Benchmark for Natural Language Understanding](#2019-11-01-5)
- [2019-10](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-10.md)
- [2019-09](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-09.md)
- [2019-08](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-08.md)
- [2019-07](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-07.md)
- [2019-06](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-06.md)
- [2019-05](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-05.md)
- [2019-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-04.md)
- [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)



# 2019-11-07

[Return to Index](#Index)



<h2 id="2019-11-07-1">1. Fast Transformer Decoding: One Write-Head is All You Need</h2>

Title: [Fast Transformer Decoding: One Write-Head is All You Need]( https://arxiv.org/abs/1911.02150 )

Authors:[Noam Shazeer](https://arxiv.org/search/cs?searchtype=author&query=Shazeer%2C+N)

*(Submitted on 6 Nov 2019)*

> Multi-head attention layers, as used in the Transformer neural sequence model, are a powerful alternative to RNNs for moving information across and between sequences. While training these layers is generally fast and simple, due to parallelizability across the length of the sequence, incremental inference (where such paralleization is impossible) is often slow, due to the memory-bandwidth cost of repeatedly loading the large "keys" and "values" tensors. We propose a variant called multi-query attention, where the keys and values are shared across all of the different attention "heads", greatly reducing the size of these tensors and hence the memory bandwidth requirements of incremental decoding. We verify experimentally that the resulting models can indeed be much faster to decode, and incur only minor quality degradation from the baseline.

| Subjects: | **Neural and Evolutionary Computing (cs.NE)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) [cs.NE] |
|           | (or [arXiv:1911.02150v1](https://arxiv.org/abs/1911.02150v1) [cs.NE] for this version) |



<h2 id="2019-11-07-2">2. Unsupervised Cross-lingual Representation Learning at Scale</h2>

Title: [Unsupervised Cross-lingual Representation Learning at Scale]( https://arxiv.org/abs/1911.02116 )

Authors:[Alexis Conneau](https://arxiv.org/search/cs?searchtype=author&query=Conneau%2C+A), [Kartikay Khandelwal](https://arxiv.org/search/cs?searchtype=author&query=Khandelwal%2C+K), [Naman Goyal](https://arxiv.org/search/cs?searchtype=author&query=Goyal%2C+N), [Vishrav Chaudhary](https://arxiv.org/search/cs?searchtype=author&query=Chaudhary%2C+V), [Guillaume Wenzek](https://arxiv.org/search/cs?searchtype=author&query=Wenzek%2C+G), [Francisco Guzmán](https://arxiv.org/search/cs?searchtype=author&query=Guzmán%2C+F), [Edouard Grave](https://arxiv.org/search/cs?searchtype=author&query=Grave%2C+E), [Myle Ott](https://arxiv.org/search/cs?searchtype=author&query=Ott%2C+M), [Luke Zettlemoyer](https://arxiv.org/search/cs?searchtype=author&query=Zettlemoyer%2C+L), [Veselin Stoyanov](https://arxiv.org/search/cs?searchtype=author&query=Stoyanov%2C+V)

*(Submitted on 5 Nov 2019)*

> This paper shows that pretraining multilingual language models at scale leads to significant performance gains for a wide range of cross-lingual transfer tasks. We train a Transformer-based masked language model on one hundred languages, using more than two terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +13.8% average accuracy on XNLI, +12.3% average F1 score on MLQA, and +2.1% average F1 score on NER. XLM-R performs particularly well on low-resource languages, improving 11.8% in XNLI accuracy for Swahili and 9.2% for Urdu over the previous XLM model. We also present a detailed empirical evaluation of the key factors that are required to achieve these gains, including the trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high and low resource languages at scale. Finally, we show, for the first time, the possibility of multilingual modeling without sacrificing per-language performance; XLM-Ris very competitive with strong monolingual models on the GLUE and XNLI benchmarks. We will make XLM-R code, data, and models publicly available.

| Comments: | 12 pages, 7 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1911.02116](https://arxiv.org/abs/1911.02116) [cs.CL] |
|           | (or [arXiv:1911.02116v1](https://arxiv.org/abs/1911.02116v1) [cs.CL] for this version) |





<h2 id="2019-11-07-3">3. Guiding Non-Autoregressive Neural Machine Translation Decoding with Reordering Information</h2>

Title: [Guiding Non-Autoregressive Neural Machine Translation Decoding with Reordering Information]( https://arxiv.org/abs/1911.02215 )

Authors:[Qiu Ran](https://arxiv.org/search/cs?searchtype=author&query=Ran%2C+Q), [Yankai Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+Y), [Peng Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+P), [Jie Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J)

*(Submitted on 6 Nov 2019)*

> Non-autoregressive neural machine translation (NAT) generates each target word in parallel and has achieved promising inference acceleration. However, existing NAT models still have a big gap in translation quality compared to autoregressive neural machine translation models due to the enormous decoding space. To address this problem, we propose a novel NAT framework named ReorderNAT which explicitly models the reordering information in the decoding procedure. We further introduce deterministic and non-deterministic decoding strategies that utilize reordering information to narrow the decoding search space in our proposed ReorderNAT. Experimental results on various widely-used datasets show that our proposed model achieves better performance compared to existing NAT models, and even achieves comparable translation quality as autoregressive translation models with a significant speedup.

| Comments: | 12 pages, 5 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1911.02215](https://arxiv.org/abs/1911.02215) [cs.CL] |
|           | (or [arXiv:1911.02215v1](https://arxiv.org/abs/1911.02215v1) [cs.CL] for this version) |





# 2019-11-06

[Return to Index](#Index)



<h2 id="2019-11-06-1">1. Training Neural Machine Translation (NMT) Models using Tensor Train Decomposition on TensorFlow (T3F)</h2>
Title: [Training Neural Machine Translation (NMT) Models using Tensor Train Decomposition on TensorFlow (T3F)]( https://arxiv.org/abs/1911.01933 )

Authors: [Amelia Drew](https://arxiv.org/search/cs?searchtype=author&query=Drew%2C+A), [Alexander Heinecke](https://arxiv.org/search/cs?searchtype=author&query=Heinecke%2C+A)

*(Submitted on 5 Nov 2019)*

> We implement a Tensor Train layer in the TensorFlow Neural Machine Translation (NMT) model using the t3f library. We perform training runs on the IWSLT English-Vietnamese '15 and WMT German-English '16 datasets with learning rates ∈{0.0004,0.0008,0.0012}, maximum ranks ∈{2,4,8,16} and a range of core dimensions. We compare against a target BLEU test score of 24.0, obtained by our benchmark run. For the IWSLT English-Vietnamese training, we obtain BLEU test/dev scores of 24.0/21.9 and 24.2/21.9 using core dimensions (2,2,256)×(2,2,512) with learning rate 0.0012 and rank distributions (1,4,4,1) and (1,4,16,1) respectively. These runs use 113\% and 397\% of the flops of the benchmark run respectively. We find that, of the parameters surveyed, a higher learning rate and more `rectangular' core dimensions generally produce higher BLEU scores. For the WMT German-English dataset, we obtain BLEU scores of 24.0/23.8 using core dimensions (4,4,128)×(4,4,256) with learning rate 0.0012 and rank distribution (1,2,2,1). We discuss the potential for future optimization and application of Tensor Train decomposition to other NMT models.

| Comments: | 10 pages, 2 tables                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Machine Learning (stat.ML) |
| Cite as:  | [arXiv:1911.01933](https://arxiv.org/abs/1911.01933) [cs.LG] |
|           | (or [arXiv:1911.01933v1](https://arxiv.org/abs/1911.01933v1) [cs.LG] for this version) |





<h2 id="2019-11-06-2">2. Emerging Cross-lingual Structure in Pretrained Language Models</h2>
Title: [Emerging Cross-lingual Structure in Pretrained Language Models]( https://arxiv.org/abs/1911.01464 )

Authors: [Shijie Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+S), [Alexis Conneau](https://arxiv.org/search/cs?searchtype=author&query=Conneau%2C+A), [Haoran Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+H), [Luke Zettlemoyer](https://arxiv.org/search/cs?searchtype=author&query=Zettlemoyer%2C+L), [Veselin Stoyanov](https://arxiv.org/search/cs?searchtype=author&query=Stoyanov%2C+V)

*(Submitted on 4 Nov 2019)*

> We study the problem of multilingual masked language modeling, i.e. the training of a single model on concatenated text from multiple languages, and present a detailed study of several factors that influence why these models are so effective for cross-lingual transfer. We show, contrary to what was previously hypothesized, that transfer is possible even when there is no shared vocabulary across the monolingual corpora and also when the text comes from very different domains. The only requirement is that there are some shared parameters in the top layers of the multi-lingual encoder. To better understand this result, we also show that representations from independently trained models in different languages can be aligned post-hoc quite effectively, strongly suggesting that, much like for non-contextual word embeddings, there are universal latent symmetries in the learned embedding spaces. For multilingual masked language modeling, these symmetries seem to be automatically discovered and aligned during the joint training process.

| Comments: | 10 pages, 6 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1911.01464](https://arxiv.org/abs/1911.01464) [cs.CL] |
|           | (or [arXiv:1911.01464v1](https://arxiv.org/abs/1911.01464v1) [cs.CL] for this version) |





<h2 id="2019-11-06-3">3. On Compositionality in Neural Machine Translation</h2>
Title: [On Compositionality in Neural Machine Translation]( https://arxiv.org/abs/1911.01497 )

Authors: [Vikas Raunak](https://arxiv.org/search/cs?searchtype=author&query=Raunak%2C+V), [Vaibhav Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+V), [Florian Metze](https://arxiv.org/search/cs?searchtype=author&query=Metze%2C+F), [Jaimie Callan](https://arxiv.org/search/cs?searchtype=author&query=Callan%2C+J)

*(Submitted on 4 Nov 2019)*

> We investigate two specific manifestations of compositionality in Neural Machine Translation (NMT) : (1) Productivity - the ability of the model to extend its predictions beyond the observed length in training data and (2) Systematicity - the ability of the model to systematically recombine known parts and rules. We evaluate a standard Sequence to Sequence model on tests designed to assess these two properties in NMT. We quantitatively demonstrate that inadequate temporal processing, in the form of poor encoder representations is a bottleneck for both Productivity and Systematicity. We propose a simple pre-training mechanism which alleviates model performance on the two properties and leads to a significant improvement in BLEU scores.

| Comments: | Accepted at Context and Compositionality Workshop, NeurIPS 2019 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1911.01497](https://arxiv.org/abs/1911.01497) [cs.CL] |
|           | (or [arXiv:1911.01497v1](https://arxiv.org/abs/1911.01497v1) [cs.CL] for this version) |





<h2 id="2019-11-06-4">4. Improving Bidirectional Decoding with Dynamic Target Semantics in Neural Machine Translation</h2>
Title: [Improving Bidirectional Decoding with Dynamic Target Semantics in Neural Machine Translation]( https://arxiv.org/abs/1911.01597 )

Authors: [Yong Shan](https://arxiv.org/search/cs?searchtype=author&query=Shan%2C+Y), [Yang Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+Y), [Jinchao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+J), [Fandong Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+F), [Wen Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+W)

*(Submitted on 5 Nov 2019)*

> Generally, Neural Machine Translation models generate target words in a left-to-right (L2R) manner and fail to exploit any future (right) semantics information, which usually produces an unbalanced translation. Recent works attempt to utilize the right-to-left (R2L) decoder in bidirectional decoding to alleviate this problem. In this paper, we propose a novel \textbf{D}ynamic \textbf{I}nteraction \textbf{M}odule (\textbf{DIM}) to dynamically exploit target semantics from R2L translation for enhancing the L2R translation quality. Different from other bidirectional decoding approaches, DIM firstly extracts helpful target information through addressing and reading operations, then updates target semantics for tracking the interactive history. Additionally, we further introduce an \textbf{agreement regularization} term into the training objective to narrow the gap between L2R and R2L translations. Experimental results on NIST Chinese⇒English and WMT'16 English⇒Romanian translation tasks show that our system achieves significant improvements over baseline systems, which also reaches comparable results compared to the state-of-the-art Transformer model with much fewer parameters of it.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1911.01597](https://arxiv.org/abs/1911.01597) [cs.CL] |
|           | (or [arXiv:1911.01597v1](https://arxiv.org/abs/1911.01597v1) [cs.CL] for this version) |





<h2 id="2019-11-06-5">5. Adversarial Language Games for Advanced Natural Language Intelligence</h2>
Title: [Adversarial Language Games for Advanced Natural Language Intelligence]( https://arxiv.org/abs/1911.01622 )

Authors: [Yuan Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao%2C+Y), [Haoxi Zhong](https://arxiv.org/search/cs?searchtype=author&query=Zhong%2C+H), [Zhengyan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Xu Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+X), [Xiaozhi Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Chaojun Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+C), [Guoyang Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+G), [Zhiyuan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Maosong Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+M)

*(Submitted on 5 Nov 2019)*

> While adversarial games have been well studied in various board games and electronic sports games, etc., such adversarial games remain a nearly blank field in natural language processing. As natural language is inherently an interactive game, we propose a challenging pragmatics game called Adversarial Taboo, in which an attacker and a defender compete with each other through sequential natural language interactions. The attacker is tasked with inducing the defender to speak a target word invisible to the defender, while the defender is tasked with detecting the target word before being induced by the attacker. In Adversarial Taboo, a successful attacker must hide its intention and subtly induce the defender, while a competitive defender must be cautious with its utterances and infer the intention of the attacker. To instantiate the game, we create a game environment and a competition platform. Sufficient pilot experiments and empirical studies on several baseline attack and defense strategies show promising and interesting results. Based on the analysis on the game and experiments, we discuss multiple promising directions for future research.

| Comments: | Work in progress                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1911.01622](https://arxiv.org/abs/1911.01622) [cs.CL] |
|           | (or [arXiv:1911.01622v1](https://arxiv.org/abs/1911.01622v1) [cs.CL] for this version) |





<h2 id="2019-11-06-6">6. Data Diversification: An Elegant Strategy For Neural Machine Translation</h2>
Title: [Data Diversification: An Elegant Strategy For Neural Machine Translation]( https://arxiv.org/abs/1911.01986 )

Authors: [Xuan-Phi Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+X), [Shafiq Joty](https://arxiv.org/search/cs?searchtype=author&query=Joty%2C+S), [Wu Kui](https://arxiv.org/search/cs?searchtype=author&query=Kui%2C+W), [Ai Ti Aw](https://arxiv.org/search/cs?searchtype=author&query=Aw%2C+A+T)

*(Submitted on 5 Nov 2019)*

> A common approach to improve neural machine translation is to invent new architectures. However, the research process of designing and refining such new models is often exhausting. Another approach is to resort to huge extra monolingual data to conduct semi-supervised training, like back-translation. But extra monolingual data is not always available, especially for low resource languages. In this paper, we propose to diversify the available training data by using multiple forward and backward peer models to augment the original training dataset. Our method does not require extra data like back-translation, nor additional computations and parameters like using pretrained models. Our data diversification method achieves state-of-the-art BLEU score of 30.7 in the WMT'14 English-German task. It also consistently and substantially improves translation quality in 8 other translation tasks: 4 IWSLT tasks (English-German and English-French) and 4 low-resource translation tasks (English-Nepali and English-Sinhala).

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1911.01986](https://arxiv.org/abs/1911.01986) [cs.CL] |
|           | (or [arXiv:1911.01986v1](https://arxiv.org/abs/1911.01986v1) [cs.CL] for this version) |



# 2019-11-05

[Return to Index](#Index)



<h2 id="2019-11-05-1">1. Attributed Sequence Embedding</h2>
Title: [Attributed Sequence Embedding]( https://arxiv.org/abs/1911.00949 )

Authors: [Zhongfang Zhuang](https://arxiv.org/search/cs?searchtype=author&query=Zhuang%2C+Z), [Xiangnan Kong](https://arxiv.org/search/cs?searchtype=author&query=Kong%2C+X), [Elke Rundensteiner](https://arxiv.org/search/cs?searchtype=author&query=Rundensteiner%2C+E), [Jihane Zouaoui](https://arxiv.org/search/cs?searchtype=author&query=Zouaoui%2C+J), [Aditya Arora](https://arxiv.org/search/cs?searchtype=author&query=Arora%2C+A)

*(Submitted on 3 Nov 2019)*

> Mining tasks over sequential data, such as clickstreams and gene sequences, require a careful design of embeddings usable by learning algorithms. Recent research in feature learning has been extended to sequential data, where each instance consists of a sequence of heterogeneous items with a variable length. However, many real-world applications often involve attributed sequences, where each instance is composed of both a sequence of categorical items and a set of attributes. In this paper, we study this new problem of attributed sequence embedding, where the goal is to learn the representations of attributed sequences in an unsupervised fashion. This problem is core to many important data mining tasks ranging from user behavior analysis to the clustering of gene sequences. This problem is challenging due to the dependencies between sequences and their associated attributes. We propose a deep multimodal learning framework, called NAS, to produce embeddings of attributed sequences. The embeddings are task independent and can be used on various mining tasks of attributed sequences. We demonstrate the effectiveness of our embeddings of attributed sequences in various unsupervised learning tasks on real-world datasets.

| Comments: | Accepted by IEEE Big Data 2019                               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Databases (cs.DB); Machine Learning (stat.ML) |
| Cite as:  | [arXiv:1911.00949](https://arxiv.org/abs/1911.00949) [cs.LG] |
|           | (or [arXiv:1911.00949v1](https://arxiv.org/abs/1911.00949v1) [cs.LG] for this version) |





<h2 id="2019-11-05-2">2. Machine Translation Evaluation using Bi-directional Entailment</h2>
Title: [Machine Translation Evaluation using Bi-directional Entailment]( https://arxiv.org/abs/1911.00681 )

Authors: [Rakesh Khobragade](https://arxiv.org/search/cs?searchtype=author&query=Khobragade%2C+R), [Heaven Patel](https://arxiv.org/search/cs?searchtype=author&query=Patel%2C+H), [Anand Namdev](https://arxiv.org/search/cs?searchtype=author&query=Namdev%2C+A), [Anish Mishra](https://arxiv.org/search/cs?searchtype=author&query=Mishra%2C+A), [Pushpak Bhattacharyya](https://arxiv.org/search/cs?searchtype=author&query=Bhattacharyya%2C+P)

*(Submitted on 2 Nov 2019)*

> In this paper, we propose a new metric for Machine Translation (MT) evaluation, based on bi-directional entailment. We show that machine generated translation can be evaluated by determining paraphrasing with a reference translation provided by a human translator. We hypothesize, and show through experiments, that paraphrasing can be detected by evaluating entailment relationship in the forward and backward direction. Unlike conventional metrics, like BLEU or METEOR, our approach uses deep learning to determine the semantic similarity between candidate and reference translation for generating scores rather than relying upon simple n-gram overlap. We use BERT's pre-trained implementation of transformer networks, fine-tuned on MNLI corpus, for natural language inferencing. We apply our evaluation metric on WMT'14 and WMT'17 dataset to evaluate systems participating in the translation task and find that our metric has a better correlation with the human annotated score compared to the other traditional metrics at system level.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1911.00681](https://arxiv.org/abs/1911.00681) [cs.CL] |
|           | (or [arXiv:1911.00681v1](https://arxiv.org/abs/1911.00681v1) [cs.CL] for this version) |





<h2 id="2019-11-05-3">3. Controlling Text Complexity in Neural Machine Translation</h2>
Title: [Controlling Text Complexity in Neural Machine Translation]( https://arxiv.org/abs/1911.00835 )

Authors: [Sweta Agrawal](https://arxiv.org/search/cs?searchtype=author&query=Agrawal%2C+S), [Marine Carpuat](https://arxiv.org/search/cs?searchtype=author&query=Carpuat%2C+M)

*(Submitted on 3 Nov 2019)*

> This work introduces a machine translation task where the output is aimed at audiences of different levels of target language proficiency. We collect a high quality dataset of news articles available in English and Spanish, written for diverse grade levels and propose a method to align segments across comparable bilingual articles. The resulting dataset makes it possible to train multi-task sequence-to-sequence models that translate Spanish into English targeted at an easier reading grade level than the original Spanish. We show that these multi-task models outperform pipeline approaches that translate and simplify text independently.

| Comments: | Accepted to EMNLP-IJCNLP 2019                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1911.00835](https://arxiv.org/abs/1911.00835) [cs.CL] |
|           | (or [arXiv:1911.00835v1](https://arxiv.org/abs/1911.00835v1) [cs.CL] for this version) |





<h2 id="2019-11-05-4">4. Machine Translation in Pronunciation Space</h2>
Title: [Machine Translation in Pronunciation Space]( https://arxiv.org/abs/1911.00932 )

Authors: [Hairong Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+H), [Mingbo Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+M), [Liang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+L)

*(Submitted on 3 Nov 2019)*

> The research in machine translation community focus on translation in text space. However, humans are in fact also good at direct translation in pronunciation space. Some existing translation systems, such as simultaneous machine translation, are inherently more natural and thus potentially more robust by directly translating in pronunciation space. In this paper, we conduct large scale experiments on a self-built dataset with about 20M En-Zh pairs of text sentences and corresponding pronunciation sentences. We proposed three new categories of translations: 1) translating a pronunciation sentence in source language into a pronunciation sentence in target language (P2P-Tran), 2) translating a text sentence in source language into a pronunciation sentence in target language (T2P-Tran), and 3) translating a pronunciation sentence in source language into a text sentence in target language (P2T-Tran), and compare them with traditional text translation (T2T-Tran). Our experiments clearly show that all 4 categories of translations have comparable performances, with small and sometimes ignorable differences.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1911.00932](https://arxiv.org/abs/1911.00932) [cs.CL] |
|           | (or [arXiv:1911.00932v1](https://arxiv.org/abs/1911.00932v1) [cs.CL] for this version) |





<h2 id="2019-11-05-5">5. Analysing Coreference in Transformer Outputs</h2>
Title: [Analysing Coreference in Transformer Outputs]( https://arxiv.org/abs/1911.01188 )

Authors: [Ekaterina Lapshinova-Koltunski](https://arxiv.org/search/cs?searchtype=author&query=Lapshinova-Koltunski%2C+E), [Cristina España-Bonet](https://arxiv.org/search/cs?searchtype=author&query=España-Bonet%2C+C), [Josef van Genabith](https://arxiv.org/search/cs?searchtype=author&query=van+Genabith%2C+J)

*(Submitted on 4 Nov 2019)*

> We analyse coreference phenomena in three neural machine translation systems trained with different data settings with or without access to explicit intra- and cross-sentential anaphoric information. We compare system performance on two different genres: news and TED talks. To do this, we manually annotate (the possibly incorrect) coreference chains in the MT outputs and evaluate the coreference chain translations. We define an error typology that aims to go further than pronoun translation adequacy and includes types such as incorrect word selection or missing words. The features of coreference chains in automatic translations are also compared to those of the source texts and human translations. The analysis shows stronger potential translationese effects in machine translated outputs than in human translations.

| Comments:          | 12 pages, 1 figure                                           |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**                         |
| Journal reference: | Fourth Workshop on Discourse in Machine Translation (DiscoMT 2019) |
| Cite as:           | [arXiv:1911.01188](https://arxiv.org/abs/1911.01188) [cs.CL] |
|                    | (or [arXiv:1911.01188v1](https://arxiv.org/abs/1911.01188v1) [cs.CL] for this version) |





<h2 id="2019-11-05-6">6. Ordering Matters: Word Ordering Aware Unsupervised NMT</h2>
Title: [Ordering Matters: Word Ordering Aware Unsupervised NMT]( https://arxiv.org/abs/1911.01212 )

Authors: [Tamali Banerjee](https://arxiv.org/search/cs?searchtype=author&query=Banerjee%2C+T), [Rudra Murthy V](https://arxiv.org/search/cs?searchtype=author&query=V%2C+R+M), [Pushpak Bhattacharyya](https://arxiv.org/search/cs?searchtype=author&query=Bhattacharyya%2C+P)

*(Submitted on 30 Oct 2019)*

> Denoising-based Unsupervised Neural Machine Translation (U-NMT) models typically employ denoising strategy at the encoder module to prevent the model from memorizing the input source sentence. Specifically, given an input sentence of length n, the model applies n/2 random swaps between consecutive words and trains the denoising-based U-NMT model. Though effective, applying denoising strategy on every sentence in the training data leads to uncertainty in the model thereby, limiting the benefits from the denoising-based U-NMT model. In this paper, we propose a simple fine-tuning strategy where we fine-tune the trained denoising-based U-NMT system without the denoising strategy. The input sentences are presented as is i.e., without any shuffling noise added. We observe significant improvements in translation performance on many language pairs from our fine-tuning strategy. Our analysis reveals that our proposed models lead to increase in higher n-gram BLEU score compared to the denoising U-NMT models.

| Comments: | 8 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Machine Learning (stat.ML) |
| Cite as:  | [arXiv:1911.01212](https://arxiv.org/abs/1911.01212) [cs.CL] |
|           | (or [arXiv:1911.01212v1](https://arxiv.org/abs/1911.01212v1) [cs.CL] for this version) |





# 2019-11-04

[Return to Index](#Index)



<h2 id="2019-11-04-1">1. Neural Cross-Lingual Relation Extraction Based on Bilingual Word Embedding Mapping</h2>
Title: [Neural Cross-Lingual Relation Extraction Based on Bilingual Word Embedding Mapping]( https://arxiv.org/abs/1911.00069 )

Authors: [Jian Ni](https://arxiv.org/search/cs?searchtype=author&query=Ni%2C+J), [Radu Florian](https://arxiv.org/search/cs?searchtype=author&query=Florian%2C+R)

*(Submitted on 31 Oct 2019)*

> Relation extraction (RE) seeks to detect and classify semantic relationships between entities, which provides useful information for many NLP applications. Since the state-of-the-art RE models require large amounts of manually annotated data and language-specific resources to achieve high accuracy, it is very challenging to transfer an RE model of a resource-rich language to a resource-poor language. In this paper, we propose a new approach for cross-lingual RE model transfer based on bilingual word embedding mapping. It projects word embeddings from a target language to a source language, so that a well-trained source-language neural network RE model can be directly applied to the target language. Experiment results show that the proposed approach achieves very good performance for a number of target languages on both in-house and open datasets, using a small bilingual dictionary with only 1K word pairs.

| Comments: | 11 pages, Conference on Empirical Methods in Natural Language Processing (EMNLP), 2019 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Information Retrieval (cs.IR); Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1911.00069](https://arxiv.org/abs/1911.00069) [cs.CL] |
|           | (or [arXiv:1911.00069v1](https://arxiv.org/abs/1911.00069v1) [cs.CL] for this version) |





<h2 id="2019-11-04-2">2. Sequence Modeling with Unconstrained Generation Order</h2>
Title: [Sequence Modeling with Unconstrained Generation Order]( https://arxiv.org/abs/1911.00176 )

Authors: [Dmitrii Emelianenko](https://arxiv.org/search/cs?searchtype=author&query=Emelianenko%2C+D), [Elena Voita](https://arxiv.org/search/cs?searchtype=author&query=Voita%2C+E), [Pavel Serdyukov](https://arxiv.org/search/cs?searchtype=author&query=Serdyukov%2C+P)

*(Submitted on 1 Nov 2019)*

> The dominant approach to sequence generation is to produce a sequence in some predefined order, e.g. left to right. In contrast, we propose a more general model that can generate the output sequence by inserting tokens in any arbitrary order. Our model learns decoding order as a result of its training procedure. Our experiments show that this model is superior to fixed order models on a number of sequence generation tasks, such as Machine Translation, Image-to-LaTeX and Image Captioning.

| Comments: | Camera-ready version for NeurIPS2019                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1911.00176](https://arxiv.org/abs/1911.00176) [cs.CL] |
|           | (or [arXiv:1911.00176v1](https://arxiv.org/abs/1911.00176v1) [cs.CL] for this version) |





<h2 id="2019-11-04-3">3. On the Linguistic Representational Power of Neural Machine Translation Models</h2>
Title: [On the Linguistic Representational Power of Neural Machine Translation Models]( https://arxiv.org/abs/1911.00317 )

Authors: [Yonatan Belinkov](https://arxiv.org/search/cs?searchtype=author&query=Belinkov%2C+Y), [Nadir Durrani](https://arxiv.org/search/cs?searchtype=author&query=Durrani%2C+N), [Fahim Dalvi](https://arxiv.org/search/cs?searchtype=author&query=Dalvi%2C+F), [Hassan Sajjad](https://arxiv.org/search/cs?searchtype=author&query=Sajjad%2C+H), [James Glass](https://arxiv.org/search/cs?searchtype=author&query=Glass%2C+J)

*(Submitted on 1 Nov 2019)*

> Despite the recent success of deep neural networks in natural language processing (NLP), their interpretability remains a challenge. We analyze the representations learned by neural machine translation models at various levels of granularity and evaluate their quality through relevant extrinsic properties. In particular, we seek answers to the following questions: (i) How accurately is word-structure captured within the learned representations, an important aspect in translating morphologically-rich languages? (ii) Do the representations capture long-range dependencies, and effectively handle syntactically divergent languages? (iii) Do the representations capture lexical semantics? We conduct a thorough investigation along several parameters: (i) Which layers in the architecture capture each of these linguistic phenomena; (ii) How does the choice of translation unit (word, character, or subword unit) impact the linguistic properties captured by the underlying representations? (iii) Do the encoder and decoder learn differently and independently? (iv) Do the representations learned by multilingual NMT models capture the same amount of linguistic information as their bilingual counterparts? Our data-driven, quantitative evaluation illuminates important aspects in NMT models and their ability to capture various linguistic phenomena. We show that deep NMT models learn a non-trivial amount of linguistic information. Notable findings include: i) Word morphology and part-of-speech information are captured at the lower layers of the model; (ii) In contrast, lexical semantics or non-local syntactic and semantic dependencies are better represented at the higher layers; (iii) Representations learned using characters are more informed about wordmorphology compared to those learned using subword units; and (iv) Representations learned by multilingual models are richer compared to bilingual models.

| Comments: | Accepted to appear in the Journal of Computational Linguistics |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1911.00317](https://arxiv.org/abs/1911.00317) [cs.CL] |
|           | (or [arXiv:1911.00317v1](https://arxiv.org/abs/1911.00317v1) [cs.CL] for this version) |





# 2019-11-01

[Return to Index](#Index)



<h2 id="2019-11-01-1">1. Fill in the Blanks: Imputing Missing Sentences for Larger-Context Neural Machine Translation</h2>
Title: [Fill in the Blanks: Imputing Missing Sentences for Larger-Context Neural Machine Translation]( https://arxiv.org/abs/1910.14075 )

Authors: [Sébastien Jean](https://arxiv.org/search/cs?searchtype=author&query=Jean%2C+S), [Ankur Bapna](https://arxiv.org/search/cs?searchtype=author&query=Bapna%2C+A), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O)

*(Submitted on 30 Oct 2019)*

> Most neural machine translation systems still translate sentences in isolation. To make further progress, a promising line of research additionally considers the surrounding context in order to provide the model potentially missing source-side information, as well as to maintain a coherent output. One difficulty in training such larger-context (i.e. document-level) machine translation systems is that context may be missing from many parallel examples. To circumvent this issue, two-stage approaches, in which sentence-level translations are post-edited in context, have recently been proposed. In this paper, we instead consider the viability of filling in the missing context. In particular, we consider three distinct approaches to generate the missing context: using random contexts, applying a copy heuristic or generating it with a language model. In particular, the copy heuristic significantly helps with lexical coherence, while using completely random contexts hurts performance on many long-distance linguistic phenomena. We also validate the usefulness of tagged back-translation. In addition to improving BLEU scores as expected, using back-translated data helps larger-context machine translation systems to better capture long-range phenomena.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1910.14075](https://arxiv.org/abs/1910.14075) [cs.CL] |
|           | (or [arXiv:1910.14075v1](https://arxiv.org/abs/1910.14075v1) [cs.CL] for this version) |





<h2 id="2019-11-01-2">2. Document-level Neural Machine Translation with Inter-Sentence Attention</h2>
Title: [Document-level Neural Machine Translation with Inter-Sentence Attention]( https://arxiv.org/abs/1910.14528 )

Authors: [Shu Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+S), [Rui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+R), [Zuchao Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Masao Utiyama](https://arxiv.org/search/cs?searchtype=author&query=Utiyama%2C+M), [Kehai Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+K), [Eiichiro Sumita](https://arxiv.org/search/cs?searchtype=author&query=Sumita%2C+E), [Hai Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+H), [Bao-liang Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+B)

*(Submitted on 31 Oct 2019)*

> Standard neural machine translation (NMT) is on the assumption of document-level context independent. Most existing document-level NMT methods only focus on briefly introducing document-level information but fail to concern about selecting the most related part inside document context. The capacity of memory network for detecting the most relevant part of the current sentence from the memory provides a natural solution for the requirement of modeling document-level context by document-level NMT. In this work, we propose a Transformer NMT system with associated memory network (AMN) to both capture the document-level context and select the most salient part related to the concerned translation from the memory. Experiments on several tasks show that the proposed method significantly improves the NMT performance over strong Transformer baselines and other related studies.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1910.14528](https://arxiv.org/abs/1910.14528) [cs.CL] |
|           | (or [arXiv:1910.14528v1](https://arxiv.org/abs/1910.14528v1) [cs.CL] for this version) |





<h2 id="2019-11-01-3">3. Naver Labs Europe's Systems for the Document-Level Generation and Translation Task at WNGT 2019</h2>
Title: [Naver Labs Europe's Systems for the Document-Level Generation and Translation Task at WNGT 2019]( https://arxiv.org/abs/1910.14539 )

Authors: [Fahimeh Saleh](https://arxiv.org/search/cs?searchtype=author&query=Saleh%2C+F), [Alexandre Bérard](https://arxiv.org/search/cs?searchtype=author&query=Bérard%2C+A), [Ioan Calapodescu](https://arxiv.org/search/cs?searchtype=author&query=Calapodescu%2C+I), [Laurent Besacier](https://arxiv.org/search/cs?searchtype=author&query=Besacier%2C+L)

*(Submitted on 31 Oct 2019)*

> Recently, neural models led to significant improvements in both machine translation (MT) and natural language generation tasks (NLG). However, generation of long descriptive summaries conditioned on structured data remains an open challenge. Likewise, MT that goes beyond sentence-level context is still an open issue (e.g., document-level MT or MT with metadata). To address these challenges, we propose to leverage data from both tasks and do transfer learning between MT, NLG, and MT with source-side metadata (MT+NLG). First, we train document-based MT systems with large amounts of parallel data. Then, we adapt these models to pure NLG and MT+NLG tasks by fine-tuning with smaller amounts of domain-specific data. This end-to-end NLG approach, without data selection and planning, outperforms the previous state of the art on the Rotowire NLG task. We participated to the "Document Generation and Translation" task at WNGT 2019, and ranked first in all tracks.

| Comments: | WNGT 2019 - System Description Paper                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1910.14539](https://arxiv.org/abs/1910.14539) [cs.CL] |
|           | (or [arXiv:1910.14539v1](https://arxiv.org/abs/1910.14539v1) [cs.CL] for this version) |





<h2 id="2019-11-01-4">4. Machine Translation of Restaurant Reviews: New Corpus for Domain Adaptation and Robustness</h2>
Title: [Machine Translation of Restaurant Reviews: New Corpus for Domain Adaptation and Robustness]( https://arxiv.org/abs/1910.14589 )

Authors: [Alexandre Bérard](https://arxiv.org/search/cs?searchtype=author&query=Bérard%2C+A), [Ioan Calapodescu](https://arxiv.org/search/cs?searchtype=author&query=Calapodescu%2C+I), [Marc Dymetman](https://arxiv.org/search/cs?searchtype=author&query=Dymetman%2C+M), [Claude Roux](https://arxiv.org/search/cs?searchtype=author&query=Roux%2C+C), [Jean-Luc Meunier](https://arxiv.org/search/cs?searchtype=author&query=Meunier%2C+J), [Vassilina Nikoulina](https://arxiv.org/search/cs?searchtype=author&query=Nikoulina%2C+V)

*(Submitted on 31 Oct 2019)*

> We share a French-English parallel corpus of Foursquare restaurant reviews ([this https URL](https://europe.naverlabs.com/research/natural-language-processing/machine-translation-of-restaurant-reviews)), and define a new task to encourage research on Neural Machine Translation robustness and domain adaptation, in a real-world scenario where better-quality MT would be greatly beneficial. We discuss the challenges of such user-generated content, and train good baseline models that build upon the latest techniques for MT robustness. We also perform an extensive evaluation (automatic and human) that shows significant improvements over existing online systems. Finally, we propose task-specific metrics based on sentiment analysis or translation accuracy of domain-specific polysemous words.

| Comments: | WNGT 2019 Paper                                              |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1910.14589](https://arxiv.org/abs/1910.14589) [cs.CL] |
|           | (or [arXiv:1910.14589v1](https://arxiv.org/abs/1910.14589v1) [cs.CL] for this version) |







<h2 id="2019-11-01-5">5. Adversarial NLI: A New Benchmark for Natural Language Understanding</h2>
Title: [Adversarial NLI: A New Benchmark for Natural Language Understanding]( https://arxiv.org/abs/1910.14599 )

Authors: [Yixin Nie](https://arxiv.org/search/cs?searchtype=author&query=Nie%2C+Y), [Adina Williams](https://arxiv.org/search/cs?searchtype=author&query=Williams%2C+A), [Emily Dinan](https://arxiv.org/search/cs?searchtype=author&query=Dinan%2C+E), [Mohit Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+M), [Jason Weston](https://arxiv.org/search/cs?searchtype=author&query=Weston%2C+J), [Douwe Kiela](https://arxiv.org/search/cs?searchtype=author&query=Kiela%2C+D)

*(Submitted on 31 Oct 2019)*

> We introduce a new large-scale NLI benchmark dataset, collected via an iterative, adversarial human-and-model-in-the-loop procedure. We show that training models on this new dataset leads to state-of-the-art performance on a variety of popular NLI benchmarks, while posing a more difficult challenge with its new test set. Our analysis sheds light on the shortcomings of current state-of-the-art models, and shows that non-expert annotators are successful at finding their weaknesses. The data collection method can be applied in a never-ending learning scenario, becoming a moving target for NLU, rather than a static benchmark that will quickly saturate.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1910.14599](https://arxiv.org/abs/1910.14599) [cs.CL] |
|           | (or [arXiv:1910.14599v1](https://arxiv.org/abs/1910.14599v1) [cs.CL] for this version) |