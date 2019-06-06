# Daily arXiv: Machine Translation - Jun., 2019

### Index

- [2019-06-06](#2019-06-06)
  - [1. Imitation Learning for Non-Autoregressive Neural Machine Translation](#2019-06-06-1)
  - [2. The Unreasonable Effectiveness of Transformer Language Models in Grammatical Error Correction](#2019-06-06-2)
  - [3. Learning Deep Transformer Models for Machine Translation](#2019-06-06-3)
  - [4. Learning Bilingual Sentence Embeddings via Autoencoding and Computing Similarities with a Multilayer Perceptron](#2019-06-06-4)
- [2019-06-05](#2019-06-05)
  - [1. Improved Zero-shot Neural Machine Translation via Ignoring Spurious Correlations](#2019-06-05-1)
  - [2. Exploring Phoneme-Level Speech Representations for End-to-End Speech Translation](#2019-06-05-2)
  - [3. Exploiting Sentential Context for Neural Machine Translation](#2019-06-05-3)
  - [4. Lattice-Based Transformer Encoder for Neural Machine Translation](#2019-06-05-4)
- [2019-06-04](#2019-06-04)
  - [1. Thinking Slow about Latency Evaluation for Simultaneous Machine Translation](#2019-06-04-1)
  - [2. Domain Adaptation of Neural Machine Translation by Lexicon Induction](#2019-06-04-2)
  - [3. Domain Adaptive Inference for Neural Machine Translation](#2019-06-04-3)
  - [4. Fluent Translations from Disfluent Speech in End-to-End Speech Translation](#2019-06-04-4)
  - [5. Evaluating Gender Bias in Machine Translation](#2019-06-04-5)
  - [6. From Words to Sentences: A Progressive Learning Approach for Zero-resource Machine Translation with Visual Pivots](#2019-06-04-6)
- [2019-06-03](#2019-06-03)
  - [1. DiaBLa: A Corpus of Bilingual Spontaneous Written Dialogues for Machine Translation](#2019-06-03)

* [2019-05](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-05.md)
* [2019-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-04.md)
* [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)
* [2019-02](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-02.md)



# 2019-06-06

[Return to Index](#Index)
<h2 id="2019-06-06-1">1. Imitation Learning for Non-Autoregressive Neural Machine Translation</h2>

Title: [Imitation Learning for Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/1906.02041)
Authors: [Bingzhen Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+B), [Mingxuan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+M), [Hao Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+H), [Junyang Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+J), [Xu Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+X)

*(Submitted on 5 Jun 2019)*

> Non-autoregressive translation models (NAT) have achieved impressive inference speedup. A potential issue of the existing NAT algorithms, however, is that the decoding is conducted in parallel, without directly considering previous context. In this paper, we propose an imitation learning framework for non-autoregressive machine translation, which still enjoys the fast translation speed but gives comparable translation performance compared to its auto-regressive counterpart. We conduct experiments on the IWSLT16, WMT14 and WMT16 datasets. Our proposed model achieves a significant speedup over the autoregressive models, while keeping the translation quality comparable to the autoregressive models. By sampling sentence length in parallel at inference time, we achieve the performance of 31.85 BLEU on WMT16 Ro→En and 30.68 BLEU on IWSLT16 En→De.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1906.02041 [cs.CL]**                         |
|           | (or **arXiv:1906.02041v1 [cs.CL]** for this version) |

<h2 id="2019-06-06-2">2. The Unreasonable Effectiveness of Transformer Language Models in Grammatical Error Correction</h2>

Title: [The Unreasonable Effectiveness of Transformer Language Models in Grammatical Error Correction](https://arxiv.org/abs/1906.01733)
Authors: [Dimitrios Alikaniotis](https://arxiv.org/search/cs?searchtype=author&query=Alikaniotis%2C+D), [Vipul Raheja](https://arxiv.org/search/cs?searchtype=author&query=Raheja%2C+V)

*(Submitted on 4 Jun 2019)*

> Recent work on Grammatical Error Correction (GEC) has highlighted the importance of language modeling in that it is certainly possible to achieve good performance by comparing the probabilities of the proposed edits. At the same time, advancements in language modeling have managed to generate linguistic output, which is almost indistinguishable from that of human-generated text. In this paper, we up the ante by exploring the potential of more sophisticated language models in GEC and offer some key insights on their strengths and weaknesses. We show that, in line with recent results in other NLP tasks, Transformer architectures achieve consistently high performance and provide a competitive baseline for future machine learning models.

| Comments: | 7 pages, 3 tables, accepted at the 14th Workshop on Innovative Use of NLP for Building Educational Applications |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Neural and Evolutionary Computing (cs.NE) |
| Cite as:  | **arXiv:1906.01733 [cs.CL]**                                 |
|           | (or **arXiv:1906.01733v1 [cs.CL]** for this version)         |

<h2 id="2019-06-06-3">3. Learning Deep Transformer Models for Machine Translation</h2>

Title: [Learning Deep Transformer Models for Machine Translation](https://arxiv.org/abs/1906.01787)
Authors: [Qiang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Q), [Bei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+B), [Tong Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+T), [Jingbo Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+J), [Changliang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+C), [Derek F. Wong](https://arxiv.org/search/cs?searchtype=author&query=Wong%2C+D+F), [Lidia S. Chao](https://arxiv.org/search/cs?searchtype=author&query=Chao%2C+L+S)

*(Submitted on 5 Jun 2019)*

> Transformer is the state-of-the-art model in recent machine translation evaluations. Two strands of research are promising to improve models of this kind: the first uses wide networks (a.k.a. Transformer-Big) and has been the de facto standard for the development of the Transformer system, and the other uses deeper language representation but faces the difficulty arising from learning deep networks. Here, we continue the line of research on the latter. We claim that a truly deep Transformer model can surpass the Transformer-Big counterpart by 1) proper use of layer normalization and 2) a novel way of passing the combination of previous layers to the next. On WMT'16 English- German, NIST OpenMT'12 Chinese-English and larger WMT'18 Chinese-English tasks, our deep system (30/25-layer encoder) outperforms the shallow Transformer-Big/Base baseline (6-layer encoder) by 0.4-2.4 BLEU points. As another bonus, the deep model is 1.6X smaller in size and 3X faster in training than Transformer-Big.

| Comments: | Accepted by ACL 2019                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1906.01787 [cs.CL]**                                 |
|           | (or **arXiv:1906.01787v1 [cs.CL]** for this version)         |

<h2 id="2019-06-06-4">4. Learning Bilingual Sentence Embeddings via Autoencoding and Computing Similarities with a Multilayer Perceptron</h2>

Title: [Learning Bilingual Sentence Embeddings via Autoencoding and Computing Similarities with a Multilayer Perceptron](https://arxiv.org/abs/1906.01942)
Authors: [Yunsu Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+Y), [Hendrik Rosendahl](https://arxiv.org/search/cs?searchtype=author&query=Rosendahl%2C+H), [Nick Rossenbach](https://arxiv.org/search/cs?searchtype=author&query=Rossenbach%2C+N), [Jan Rosendahl](https://arxiv.org/search/cs?searchtype=author&query=Rosendahl%2C+J), [Shahram Khadivi](https://arxiv.org/search/cs?searchtype=author&query=Khadivi%2C+S), [Hermann Ney](https://arxiv.org/search/cs?searchtype=author&query=Ney%2C+H)

*(Submitted on 5 Jun 2019)*

> We propose a novel model architecture and training algorithm to learn bilingual sentence embeddings from a combination of parallel and monolingual data. Our method connects autoencoding and neural machine translation to force the source and target sentence embeddings to share the same space without the help of a pivot language or an additional transformation. We train a multilayer perceptron on top of the sentence embeddings to extract good bilingual sentence pairs from nonparallel or noisy parallel data. Our approach shows promising performance on sentence alignment recovery and the WMT 2018 parallel corpus filtering tasks with only a single model.

| Comments: | ACL 2019 Repl4NLP camera-ready                               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1906.01942 [cs.CL]**                                 |
|           | (or **arXiv:1906.01942v1 [cs.CL]** for this version)         |


# 2019-06-05

[Return to Index](#Index)
<h2 id="2019-06-05-1">1. Improved Zero-shot Neural Machine Translation via Ignoring Spurious Correlations</h2>

Title: [Improved Zero-shot Neural Machine Translation via Ignoring Spurious Correlations](https://arxiv.org/abs/1906.01181)
Authors: [Jiatao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+J), [Yong Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Kyunghyun Cho](https://arxiv.org/search/cs?searchtype=author&query=Cho%2C+K), [Victor O.K. Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+V+O)

*(Submitted on 4 Jun 2019)*

> Zero-shot translation, translating between language pairs on which a Neural Machine Translation (NMT) system has never been trained, is an emergent property when training the system in multilingual settings. However, naive training for zero-shot NMT easily fails, and is sensitive to hyper-parameter setting. The performance typically lags far behind the more conventional pivot-based approach which translates twice using a third language as a pivot. In this work, we address the degeneracy problem due to capturing spurious correlations by quantitatively analyzing the mutual information between language IDs of the source and decoded sentences. Inspired by this analysis, we propose to use two simple but effective approaches: (1) decoder pre-training; (2) back-translation. These methods show significant improvement (4~22 BLEU points) over the vanilla zero-shot translation on three challenging multilingual datasets, and achieve similar or better results than the pivot-based approach.

| Comments: | Accepted by ACL 2019                                 |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1906.01181 [cs.CL]**                         |
|           | (or **arXiv:1906.01181v1 [cs.CL]** for this version) |

<h2 id="2019-06-05-2">2. Exploring Phoneme-Level Speech Representations for End-to-End Speech Translation</h2>

Title: [Exploring Phoneme-Level Speech Representations for End-to-End Speech Translation](https://arxiv.org/abs/1906.01199)
Authors: [Elizabeth Salesky](https://arxiv.org/search/cs?searchtype=author&query=Salesky%2C+E), [Matthias Sperber](https://arxiv.org/search/cs?searchtype=author&query=Sperber%2C+M), [Alan W Black](https://arxiv.org/search/cs?searchtype=author&query=Black%2C+A+W)

*(Submitted on 4 Jun 2019)*

> Previous work on end-to-end translation from speech has primarily used frame-level features as speech representations, which creates longer, sparser sequences than text. We show that a naive method to create compressed phoneme-like speech representations is far more effective and efficient for translation than traditional frame-level speech features. Specifically, we generate phoneme labels for speech frames and average consecutive frames with the same label to create shorter, higher-level source sequences for translation. We see improvements of up to 5 BLEU on both our high and low resource language pairs, with a reduction in training time of 60%. Our improvements hold across multiple data sizes and two language pairs.

| Comments: | Accepted to ACL 2019                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **arXiv:1906.01199 [cs.CL]**                                 |
|           | (or **arXiv:1906.01199v1 [cs.CL]** for this version)         |

<h2 id="2019-06-05-3">3. Exploiting Sentential Context for Neural Machine Translation</h2>

Title: [Exploiting Sentential Context for Neural Machine Translation](https://arxiv.org/abs/1906.01268)
Authors: [Xing Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Zhaopeng Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+Z), [Longyue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Shuming Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+S)

*(Submitted on 4 Jun 2019)*

> In this work, we present novel approaches to exploit sentential context for neural machine translation (NMT). Specifically, we first show that a shallow sentential context extracted from the top encoder layer only, can improve translation performance via contextualizing the encoding representations of individual words. Next, we introduce a deep sentential context, which aggregates the sentential context representations from all the internal layers of the encoder to form a more comprehensive context representation. Experimental results on the WMT14 English-to-German and English-to-French benchmarks show that our model consistently improves performance over the strong TRANSFORMER model (Vaswani et al., 2017), demonstrating the necessity and effectiveness of exploiting sentential context for NMT.

| Comments: | Accepted by ACL 2019                                 |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1906.01268 [cs.CL]**                         |
|           | (or **arXiv:1906.01268v1 [cs.CL]** for this version) |

<h2 id="2019-06-05-4">4. Lattice-Based Transformer Encoder for Neural Machine Translation</h2>

Title: [Lattice-Based Transformer Encoder for Neural Machine Translation](https://arxiv.org/abs/1906.01282)
Authors: [Fengshun Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+F), [Jiangtong Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J), [Hai Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+H), [Rui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+R), [Kehai Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+K)

*(Submitted on 4 Jun 2019)*

> Neural machine translation (NMT) takes deterministic sequences for source representations. However, either word-level or subword-level segmentations have multiple choices to split a source sequence with different word segmentors or different subword vocabulary sizes. We hypothesize that the diversity in segmentations may affect the NMT performance. To integrate different segmentations with the state-of-the-art NMT model, Transformer, we propose lattice-based encoders to explore effective word or subword representation in an automatic way during training. We propose two methods: 1) lattice positional encoding and 2) lattice-aware self-attention. These two methods can be used together and show complementary to each other to further improve translation performance. Experiment results show superiorities of lattice-based encoders in word-level and subword-level representations over conventional Transformer encoder.

| Comments: | Accepted by ACL 2019                                 |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1906.01282 [cs.CL]**                         |
|           | (or **arXiv:1906.01282v1 [cs.CL]** for this version) |



# 2019-06-04

[Return to Index](#Index)
<h2 id="2019-06-04-1">1. Thinking Slow about Latency Evaluation for Simultaneous Machine Translation</h2>

Title: [Thinking Slow about Latency Evaluation for Simultaneous Machine Translation](https://arxiv.org/abs/1906.00048)

Authors: [Colin Cherry](https://arxiv.org/search/cs?searchtype=author&query=Cherry%2C+C), [George Foster](https://arxiv.org/search/cs?searchtype=author&query=Foster%2C+G)

*(Submitted on 31 May 2019)*

> Simultaneous machine translation attempts to translate a source sentence before it is finished being spoken, with applications to translation of spoken language for live streaming and conversation. Since simultaneous systems trade quality to reduce latency, having an effective and interpretable latency metric is crucial. We introduce a variant of the recently proposed Average Lagging (AL) metric, which we call Differentiable Average Lagging (DAL). It distinguishes itself by being differentiable and internally consistent to its underlying mathematical model.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1906.00048 [cs.CL]**                         |
|           | (or **arXiv:1906.00048v1 [cs.CL]** for this version) |



<h2 id="2019-06-04-2">2. Domain Adaptation of Neural Machine Translation by Lexicon Induction</h2>

Title: [Domain Adaptation of Neural Machine Translation by Lexicon Induction](https://arxiv.org/abs/1906.00376)

Authors:[Junjie Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+J), [Mengzhou Xia](https://arxiv.org/search/cs?searchtype=author&query=Xia%2C+M), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G), [Jaime Carbonell](https://arxiv.org/search/cs?searchtype=author&query=Carbonell%2C+J)

*(Submitted on 2 Jun 2019)*

> It has been previously noted that neural machine translation (NMT) is very sensitive to domain shift. In this paper, we argue that this is a dual effect of the highly lexicalized nature of NMT, resulting in failure for sentences with large numbers of unknown words, and lack of supervision for domain-specific words. To remedy this problem, we propose an unsupervised adaptation method which fine-tunes a pre-trained out-of-domain NMT model using a pseudo-in-domain corpus. Specifically, we perform lexicon induction to extract an in-domain lexicon, and construct a pseudo-parallel in-domain corpus by performing word-for-word back-translation of monolingual in-domain target sentences. In five domains over twenty pairwise adaptation settings and two model architectures, our method achieves consistent improvements without using any in-domain parallel sentences, improving up to 14 BLEU over unadapted models, and up to 2 BLEU over strong back-translation baselines.

| Subjects:          | **Computation and Language (cs.CL)**                         |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | published at the 57th Annual Meeting of the Association for Computational Linguistics (ACL). July 2019 |
| Cite as:           | **arXiv:1906.00376 [cs.CL]**                                 |
|                    | (or **arXiv:1906.00376v1 [cs.CL]** for this version)         |

 





<h2 id="2019-06-04-3">3. Domain Adaptive Inference for Neural Machine Translation</h2>

Title: [Domain Adaptive Inference for Neural Machine Translation](https://arxiv.org/abs/1906.00408)

Authors: [Danielle Saunders](https://arxiv.org/search/cs?searchtype=author&query=Saunders%2C+D), [Felix Stahlberg](https://arxiv.org/search/cs?searchtype=author&query=Stahlberg%2C+F), [Adria de Gispert](https://arxiv.org/search/cs?searchtype=author&query=de+Gispert%2C+A), [Bill Byrne](https://arxiv.org/search/cs?searchtype=author&query=Byrne%2C+B)

*(Submitted on 2 Jun 2019)*

> We investigate adaptive ensemble weighting for Neural Machine Translation, addressing the case of improving performance on a new and potentially unknown domain without sacrificing performance on the original domain. We adapt sequentially across two Spanish-English and three English-German tasks, comparing unregularized fine-tuning, L2 and Elastic Weight Consolidation. We then report a novel scheme for adaptive NMT ensemble decoding by extending Bayesian Interpolation with source information, and show strong improvements across test domains without access to the domain label.

| Comments: | To appear at ACL 2019                                |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1906.00408 [cs.CL]**                         |
|           | (or **arXiv:1906.00408v1 [cs.CL]** for this version) |



<h2 id="2019-06-04-4">4. Fluent Translations from Disfluent Speech in End-to-End Speech Translation</h2>

Title: [Fluent Translations from Disfluent Speech in End-to-End Speech Translation](https://arxiv.org/abs/1906.00556)

Authors: [Elizabeth Salesky](https://arxiv.org/search/cs?searchtype=author&query=Salesky%2C+E), [Matthias Sperber](https://arxiv.org/search/cs?searchtype=author&query=Sperber%2C+M), [Alex Waibel](https://arxiv.org/search/cs?searchtype=author&query=Waibel%2C+A)

*(Submitted on 3 Jun 2019)*

> Spoken language translation applications for speech suffer due to conversational speech phenomena, particularly the presence of disfluencies. With the rise of end-to-end speech translation models, processing steps such as disfluency removal that were previously an intermediate step between speech recognition and machine translation need to be incorporated into model architectures. We use a sequence-to-sequence model to translate from noisy, disfluent speech to fluent text with disfluencies removed using the recently collected `copy-edited' references for the Fisher Spanish-English dataset. We are able to directly generate fluent translations and introduce considerations about how to evaluate success on this task. This work provides a baseline for a new task, the translation of conversational speech with joint removal of disfluencies.

| Comments: | Accepted at NAACL 2019                               |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1906.00556 [cs.CL]**                         |
|           | (or **arXiv:1906.00556v1 [cs.CL]** for this version) |



<h2 id="2019-06-04-5">5. Evaluating Gender Bias in Machine Translation</h2>

Title: [Evaluating Gender Bias in Machine Translation](https://arxiv.org/abs/1906.00591)

Authors: [Gabriel Stanovsky](https://arxiv.org/search/cs?searchtype=author&query=Stanovsky%2C+G), [Noah A. Smith](https://arxiv.org/search/cs?searchtype=author&query=Smith%2C+N+A), [Luke Zettlemoyer](https://arxiv.org/search/cs?searchtype=author&query=Zettlemoyer%2C+L)

*(Submitted on 3 Jun 2019)*

> We present the first challenge set and evaluation protocol for the analysis of gender bias in machine translation (MT). Our approach uses two recent coreference resolution datasets composed of English sentences which cast participants into non-stereotypical gender roles (e.g., "The doctor asked the nurse to help her in the operation"). We devise an automatic gender bias evaluation method for eight target languages with grammatical gender, based on morphological analysis (e.g., the use of female inflection for the word "doctor"). Our analyses show that four popular industrial MT systems and two recent state-of-the-art academic MT models are significantly prone to gender-biased translation errors for all tested target languages. Our data and code are made publicly available.

| Comments: | Accepted to ACL 2019                                 |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1906.00591 [cs.CL]**                         |
|           | (or **arXiv:1906.00591v1 [cs.CL]** for this version) |



<h2 id="2019-06-04-6">6. From Words to Sentences: A Progressive Learning Approach for Zero-resource Machine Translation with Visual Pivots</h2>

Title: [From Words to Sentences: A Progressive Learning Approach for Zero-resource Machine Translation with Visual Pivots](https://arxiv.org/abs/1906.00872)

Authors: [Shizhe Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+S), [Qin Jin](https://arxiv.org/search/cs?searchtype=author&query=Jin%2C+Q), [Jianlong Fu](https://arxiv.org/search/cs?searchtype=author&query=Fu%2C+J)

*(Submitted on 3 Jun 2019)*

> The neural machine translation model has suffered from the lack of large-scale parallel corpora. In contrast, we humans can learn multi-lingual translations even without parallel texts by referring our languages to the external world. To mimic such human learning behavior, we employ images as pivots to enable zero-resource translation learning. However, a picture tells a thousand words, which makes multi-lingual sentences pivoted by the same image noisy as mutual translations and thus hinders the translation model learning. In this work, we propose a progressive learning approach for image-pivoted zero-resource machine translation. Since words are less diverse when grounded in the image, we first learn word-level translation with image pivots, and then progress to learn the sentence-level translation by utilizing the learned word translation to suppress noises in image-pivoted multi-lingual sentences. Experimental results on two widely used image-pivot translation datasets, IAPR-TC12 and Multi30k, show that the proposed approach significantly outperforms other state-of-the-art methods.

| Comments: | Accepted by IJCAI 2019                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **arXiv:1906.00872 [cs.CL]**                                 |
|           | (or **arXiv:1906.00872v1 [cs.CL]** for this version)         |



# 2019-06-03

[Return to Index](#Index)
<h2 id="2019-06-03-1">1. DiaBLa: A Corpus of Bilingual Spontaneous Written Dialogues for Machine Translation</h2>

Title: [DiaBLa: A Corpus of Bilingual Spontaneous Written Dialogues for Machine Translation](https://arxiv.org/abs/1905.13354)

Authors: [Rachel Bawden](https://arxiv.org/search/cs?searchtype=author&query=Bawden%2C+R), [Sophie Rosset](https://arxiv.org/search/cs?searchtype=author&query=Rosset%2C+S), [Thomas Lavergne](https://arxiv.org/search/cs?searchtype=author&query=Lavergne%2C+T), [Eric Bilinski](https://arxiv.org/search/cs?searchtype=author&query=Bilinski%2C+E)

*(Submitted on 30 May 2019)*

> We present a new English-French test set for the evaluation of Machine Translation (MT) for informal, written bilingual dialogue. The test set contains 144 spontaneous dialogues (5,700+ sentences) between native English and French speakers, mediated by one of two neural MT systems in a range of role-play settings. The dialogues are accompanied by fine-grained sentence-level judgments of MT quality, produced by the dialogue participants themselves, as well as by manually normalised versions and reference translations produced a posteriori. The motivation for the corpus is two-fold: to provide (i) a unique resource for evaluating MT models, and (ii) a corpus for the analysis of MT-mediated communication. We provide a preliminary analysis of the corpus to confirm that the participants' judgments reveal perceptible differences in MT quality between the two MT systems used.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1905.13354 [cs.CL]**                         |
|           | (or **arXiv:1905.13354v1 [cs.CL]** for this version) |