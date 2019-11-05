# Daily arXiv: Machine Translation - Nov., 2019

# Index

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