# Daily arXiv: Machine Translation - Jul., 2019

### Index

- [2019-07-16](#2019-07-16)
  - [1. Microsoft Translator at WMT 2019: Towards Large-Scale Document-Level Neural Machine Translation](#2019-07-16-1)
  - [2. Simple Automatic Post-editing for Arabic-Japanese Machine Translation](#2019-07-16-2)
  - [3. GLOSS: Generative Latent Optimization of Sentence Representations](#2019-07-16-3)
  - [4. Naver Labs Europe's Systems for the WMT19 Machine Translation Robustness Task](#2019-07-16-4)
  - [5. Facebook FAIR's WMT19 News Translation Task Submission](#2019-07-16-5)
- [2019-07-15](#2019-07-15)
  - [1. R-Transformer: Recurrent Neural Network Enhanced Transformer](#2019-07-15-1)
  - [2. The University of Edinburgh's Submissions to the WMT19 News Translation Task](#2019-07-15-2)
- [2019-07-12](#2019-07-12)
  - [1. Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges](#2019-07-12-1)
  - [2. Self-Regulated Interactive Sequence-to-Sequence Learning](#2019-07-12-2)
  - [3. To Tune or Not To Tune? How About the Best of Both Worlds?](#2019-07-12-3)
- [2019-07-11](#2019-07-11)
  - [1.  Learning to Speak Fluently in a Foreign Language: Multilingual Speech Synthesis and Cross-Language Voice Cloning](#2019-07-11-1)
  - [2.  Lingua Custodia at WMT'19: Attempts to Control Terminology](#2019-07-11-2)
- [2019-07-10](#2019-07-10)
  - [1. Learning Neural Sequence-to-Sequence Models from Weak Feedback with Bipolar Ramp Loss](#2019-07-10-1)
  - [2. An Intrinsic Nearest Neighbor Analysis of Neural Machine Translation Architectures](#2019-07-10-2)
  - [3. NTT's Machine Translation Systems for WMT19 Robustness Task](#2019-07-10-3)
  - [4. Multilingual Universal Sentence Encoder for Semantic Retrieval](#2019-07-10-4)
- [2019-07-09](#2019-07-09)
  - [1. Exploiting Out-of-Domain Parallel Data through Multilingual Transfer Learning for Low-Resource Neural Machine Translation](#2019-07-09-1)
  - [2. Best Practices for Learning Domain-Specific Cross-Lingual Embeddings](#2019-07-09-2)
  - [3. Evolutionary Algorithm for Sinhala to English Translation](#2019-07-09-3)
  - [4. Correct-and-Memorize: Learning to Translate from Interactive Revisions](#2019-07-09-4)
- [2019-07-08](#2019-07-08)
  - [1. Multi-lingual Intent Detection and Slot Filling in a Joint BERT-based Model](#2019-07-08-1)
- [2019-07-03](#2019-07-03)
  - [1. A Neural Grammatical Error Correction System Built On Better Pre-training and Sequential Transfer Learning](#2019-07-03-1)
  - [2. Improving Robustness in Real-World Neural Machine Translation Engines](#2019-07-03-2)
- [2019-07-02](#2019-07-02)
  - [1. The University of Sydney's Machine Translation System for WMT19](#2019-07-02-1)
  - [2. Few-Shot Representation Learning for Out-Of-Vocabulary Words](#2019-07-02-2)
  - [3. From Bilingual to Multilingual Neural Machine Translation by Incremental Training](#2019-07-02-3)
  - [4. Post-editese: an Exacerbated Translationese](#2019-07-02-4)
- [2019-07-01](#2019-07-01)
  - [1. Findings of the First Shared Task on Machine Translation Robustness](#2019-07-01-1)
  - [2. Lost in Translation: Loss and Decay of Linguistic Richness in Machine Translation](#2019-07-01-2)
  - [3. Widening the Representation Bottleneck in Neural Machine Translation with Lexical Shortcuts](#2019-07-01-3)

* [2019-06](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-06.md)
* [2019-05](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-05.md)
* [2019-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-04.md)
* [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)
* [2019-02](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-02.md)



# 2019-07-16

[Return to Index](#Index)

<h2 id="2019-07-16-1">1. Microsoft Translator at WMT 2019: Towards Large-Scale Document-Level Neural Machine Translation</h2>
Title: [Microsoft Translator at WMT 2019: Towards Large-Scale Document-Level Neural Machine Translation](https://arxiv.org/abs/1907.06170)

Authors: [Marcin Junczys-Dowmunt](https://arxiv.org/search/cs?searchtype=author&query=Junczys-Dowmunt%2C+M)

*(Submitted on 14 Jul 2019)*

> This paper describes the Microsoft Translator submissions to the WMT19 news translation shared task for English-German. Our main focus is document-level neural machine translation with deep transformer models. We start with strong sentence-level baselines, trained on large-scale data created via data-filtering and noisy back-translation and find that back-translation seems to mainly help with translationese input. We explore fine-tuning techniques, deeper models and different ensembling strategies to counter these effects. Using document boundaries present in the authentic and synthetic parallel data, we create sequences of up to 1000 subword segments and train transformer translation models. We experiment with data augmentation techniques for the smaller authentic data with document-boundaries and for larger authentic data without boundaries. We further explore multi-task training for the incorporation of document-level source language monolingual data via the BERT-objective on the encoder and two-pass decoding for combinations of sentence-level and document-level systems. Based on preliminary human evaluation results, evaluators strongly prefer the document-level systems over our comparable sentence-level system. The document-level systems also seem to score higher than the human references in source-based direct assessment.

| Comments: | WMT 2019 Shared Task submission                      |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1907.06170 [cs.CL]**                         |
|           | (or **arXiv:1907.06170v1 [cs.CL]** for this version) |



<h2 id="2019-07-16-2">2. Simple Automatic Post-editing for Arabic-Japanese Machine Translation</h2>
Title: [Simple Automatic Post-editing for Arabic-Japanese Machine Translation](https://arxiv.org/abs/1907.06210)

Authors: [Ella Noll](https://arxiv.org/search/cs?searchtype=author&query=Noll%2C+E), [Mai Oudah](https://arxiv.org/search/cs?searchtype=author&query=Oudah%2C+M), [Nizar Habash](https://arxiv.org/search/cs?searchtype=author&query=Habash%2C+N)

*(Submitted on 14 Jul 2019)*

> A common bottleneck for developing machine translation (MT) systems for some language pairs is the lack of direct parallel translation data sets, in general and in certain domains. Alternative solutions such as zero-shot models or pivoting techniques are successful in getting a strong baseline, but are often below the more supported language-pair systems. In this paper, we focus on Arabic-Japanese machine translation, a less studied language pair; and we work with a unique parallel corpus of Arabic news articles that were manually translated to Japanese. We use this parallel corpus to adapt a state-of-the-art domain/genre agnostic neural MT system via a simple automatic post-editing technique. Our results and detailed analysis suggest that this approach is quite viable for less supported language pairs in specific domains.

| Comments: | Machine translation, Automatic Post editing, Arabic, Japanese |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1907.06210 [cs.CL]**                                 |
|           | (or **arXiv:1907.06210v1 [cs.CL]** for this version)         |



<h2 id="2019-07-16-3">3. GLOSS: Generative Latent Optimization of Sentence Representations</h2>
Title: [GLOSS: Generative Latent Optimization of Sentence Representations](https://arxiv.org/abs/1907.06385)

Authors: [Sidak Pal Singh](https://arxiv.org/search/cs?searchtype=author&query=Singh%2C+S+P), [Angela Fan](https://arxiv.org/search/cs?searchtype=author&query=Fan%2C+A), [Michael Auli](https://arxiv.org/search/cs?searchtype=author&query=Auli%2C+M)

*(Submitted on 15 Jul 2019)*

> We propose a method to learn unsupervised sentence representations in a non-compositional manner based on Generative Latent Optimization. Our approach does not impose any assumptions on how words are to be combined into a sentence representation. We discuss a simple Bag of Words model as well as a variant that models word positions. Both are trained to reconstruct the sentence based on a latent code and our model can be used to generate text. Experiments show large improvements over the related Paragraph Vectors. Compared to uSIF, we achieve a relative improvement of 5% when trained on the same data and our method performs competitively to Sent2vec while trained on 30 times less data.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1907.06385 [cs.CL]**                         |
|           | (or **arXiv:1907.06385v1 [cs.CL]** for this version) |



<h2 id="2019-07-16-4">4. Naver Labs Europe's Systems for the WMT19 Machine Translation Robustness Task</h2>
Title: [Naver Labs Europe's Systems for the WMT19 Machine Translation Robustness Task](https://arxiv.org/abs/1907.06488)

Authors: [Alexandre Bérard](https://arxiv.org/search/cs?searchtype=author&query=Bérard%2C+A), [Ioan Calapodescu](https://arxiv.org/search/cs?searchtype=author&query=Calapodescu%2C+I), [Claude Roux](https://arxiv.org/search/cs?searchtype=author&query=Roux%2C+C)

*(Submitted on 15 Jul 2019)*

> This paper describes the systems that we submitted to the WMT19 Machine Translation robustness task. This task aims to improve MT's robustness to noise found on social media, like informal language, spelling mistakes and other orthographic variations. The organizers provide parallel data extracted from a social media website in two language pairs: French-English and Japanese-English (in both translation directions). The goal is to obtain the best scores on unseen test sets from the same source, according to automatic metrics (BLEU) and human evaluation. We proposed one single and one ensemble system for each translation direction. Our ensemble models ranked first in all language pairs, according to BLEU evaluation. We discuss the pre-processing choices that we made, and present our solutions for robustness to noise and domain adaptation.

| Comments: | WMT 2019 - Shared Task Paper                         |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1907.06488 [cs.CL]**                         |
|           | (or **arXiv:1907.06488v1 [cs.CL]** for this version) |



<h2 id="2019-07-16-5">5. Facebook FAIR's WMT19 News Translation Task Submission</h2>
Title: [Facebook FAIR's WMT19 News Translation Task Submission](https://arxiv.org/abs/1907.06616)

Authors: [Nathan Ng](https://arxiv.org/search/cs?searchtype=author&query=Ng%2C+N), [Kyra Yee](https://arxiv.org/search/cs?searchtype=author&query=Yee%2C+K), [Alexei Baevski](https://arxiv.org/search/cs?searchtype=author&query=Baevski%2C+A), [Myle Ott](https://arxiv.org/search/cs?searchtype=author&query=Ott%2C+M), [Michael Auli](https://arxiv.org/search/cs?searchtype=author&query=Auli%2C+M), [Sergey Edunov](https://arxiv.org/search/cs?searchtype=author&query=Edunov%2C+S)

*(Submitted on 15 Jul 2019)*

> This paper describes Facebook FAIR's submission to the WMT19 shared news translation task. We participate in two language pairs and four language directions, English <-> German and English <-> Russian. Following our submission from last year, our baseline systems are large BPE-based transformer models trained with the Fairseq sequence modeling toolkit which rely on sampled back-translations. This year we experiment with different bitext data filtering schemes, as well as with adding filtered back-translated data. We also ensemble and fine-tune our models on domain-specific data, then decode using noisy channel model reranking. Our submissions are ranked first in all four directions of the human evaluation campaign. On En->De, our system significantly outperforms other systems as well as human translations. This system improves upon our WMT'18 submission by 4.5 BLEU points.

| Comments: | 7 pages; WMT                                         |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1907.06616 [cs.CL]**                         |
|           | (or **arXiv:1907.06616v1 [cs.CL]** for this version) |



# 2019-07-15

[Return to Index](#Index)

<h2 id="2019-07-15-1">1. R-Transformer: Recurrent Neural Network Enhanced Transformer</h2>
Title: [R-Transformer: Recurrent Neural Network Enhanced Transformer](https://arxiv.org/abs/1907.05572)

Authors: [Zhiwei Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Z), [Yao Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+Y), [Zitao Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Jiliang Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+J)

*(Submitted on 12 Jul 2019)*

> Recurrent Neural Networks have long been the dominating choice for sequence modeling. However, it severely suffers from two issues: impotent in capturing very long-term dependencies and unable to parallelize the sequential computation procedure. Therefore, many non-recurrent sequence models that are built on convolution and attention operations have been proposed recently. Notably, models with multi-head attention such as Transformer have demonstrated extreme effectiveness in capturing long-term dependencies in a variety of sequence modeling tasks. Despite their success, however, these models lack necessary components to model local structures in sequences and heavily rely on position embeddings that have limited effects and require a considerable amount of design efforts. In this paper, we propose the R-Transformer which enjoys the advantages of both RNNs and the multi-head attention mechanism while avoids their respective drawbacks. The proposed model can effectively capture both local structures and global long-term dependencies in sequences without any use of position embeddings. We evaluate R-Transformer through extensive experiments with data from a wide range of domains and the empirical results show that R-Transformer outperforms the state-of-the-art methods by a large margin in most of the tasks. We have made the code publicly available at \url{[this https URL](https://github.com/DSE-MSU/R-transformer)}.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1907.05572 [cs.LG]**                                 |
|           | (or **arXiv:1907.05572v1 [cs.LG]** for this version)         |

<h2 id="2019-07-15-2">2. The University of Edinburgh's Submissions to the WMT19 News Translation Task</h2>
Title: [The University of Edinburgh's Submissions to the WMT19 News Translation Task](https://arxiv.org/abs/1907.05854)

Authors: [Rachel Bawden](https://arxiv.org/search/cs?searchtype=author&query=Bawden%2C+R), [Nikolay Bogoychev](https://arxiv.org/search/cs?searchtype=author&query=Bogoychev%2C+N), [Ulrich Germann](https://arxiv.org/search/cs?searchtype=author&query=Germann%2C+U), [Roman Grundkiewicz](https://arxiv.org/search/cs?searchtype=author&query=Grundkiewicz%2C+R), [Faheem Kirefu](https://arxiv.org/search/cs?searchtype=author&query=Kirefu%2C+F), [Antonio Valerio Miceli Barone](https://arxiv.org/search/cs?searchtype=author&query=Barone%2C+A+V+M), [Alexandra Birch](https://arxiv.org/search/cs?searchtype=author&query=Birch%2C+A)

*(Submitted on 12 Jul 2019)*

> The University of Edinburgh participated in the WMT19 Shared Task on News Translation in six language directions: English-to-Gujarati, Gujarati-to-English, English-to-Chinese, Chinese-to-English, German-to-English, and English-to-Czech. For all translation directions, we created or used back-translations of monolingual data in the target language as additional synthetic training data. For English-Gujarati, we also explored semi-supervised MT with cross-lingual language model pre-training, and translation pivoting through Hindi. For translation to and from Chinese, we investigated character-based tokenisation vs. sub-word segmentation of Chinese text. For German-to-English, we studied the impact of vast amounts of back-translated training data on translation quality, gaining a few additional insights over Edunov et al. (2018). For English-to-Czech, we compared different pre-processing and tokenisation regimes.

| Comments: | To appear in the Proceedings of WMT19: Shared Task Papers |
| --------- | --------------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                      |
| Cite as:  | **arXiv:1907.05854 [cs.CL]**                              |
|           | (or **arXiv:1907.05854v1 [cs.CL]** for this version)      |





# 2019-07-12

[Return to Index](#Index)

<h2 id="2019-07-12-1">1. Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges</h2>
Title: [Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges](https://arxiv.org/abs/1907.05019)

Authors:[Naveen Arivazhagan](https://arxiv.org/search/cs?searchtype=author&query=Arivazhagan%2C+N), [Ankur Bapna](https://arxiv.org/search/cs?searchtype=author&query=Bapna%2C+A), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O), [Dmitry Lepikhin](https://arxiv.org/search/cs?searchtype=author&query=Lepikhin%2C+D), [Melvin Johnson](https://arxiv.org/search/cs?searchtype=author&query=Johnson%2C+M), [Maxim Krikun](https://arxiv.org/search/cs?searchtype=author&query=Krikun%2C+M), [Mia Xu Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+M+X), [Yuan Cao](https://arxiv.org/search/cs?searchtype=author&query=Cao%2C+Y), [George Foster](https://arxiv.org/search/cs?searchtype=author&query=Foster%2C+G), [Colin Cherry](https://arxiv.org/search/cs?searchtype=author&query=Cherry%2C+C), [Wolfgang Macherey](https://arxiv.org/search/cs?searchtype=author&query=Macherey%2C+W), [Zhifeng Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Z), [Yonghui Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+Y)

*(Submitted on 11 Jul 2019)*

> We introduce our efforts towards building a universal neural machine translation (NMT) system capable of translating between any language pair. We set a milestone towards this goal by building a single massively multilingual NMT model handling 103 languages trained on over 25 billion examples. Our system demonstrates effective transfer learning ability, significantly improving translation quality of low-resource languages, while keeping high-resource language translation quality on-par with competitive bilingual baselines. We provide in-depth analysis of various aspects of model building that are crucial to achieving quality and practicality in universal NMT. While we prototype a high-quality universal translation system, our extensive empirical analysis exposes issues that need to be further addressed, and we suggest directions for future research.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1907.05019 [cs.CL]**                                 |
|           | (or **arXiv:1907.05019v1 [cs.CL]** for this version)         |

<h2 id="2019-07-12-2">2. Self-Regulated Interactive Sequence-to-Sequence Learning</h2>
Title: [Self-Regulated Interactive Sequence-to-Sequence Learning](https://arxiv.org/abs/1907.05190)

Authors:[Julia Kreutzer](https://arxiv.org/search/cs?searchtype=author&query=Kreutzer%2C+J), [Stefan Riezler](https://arxiv.org/search/cs?searchtype=author&query=Riezler%2C+S)

*(Submitted on 11 Jul 2019)*

> Not all types of supervision signals are created equal: Different types of feedback have different costs and effects on learning. We show how self-regulation strategies that decide when to ask for which kind of feedback from a teacher (or from oneself) can be cast as a learning-to-learn problem leading to improved cost-aware sequence-to-sequence learning. In experiments on interactive neural machine translation, we find that the self-regulator discovers an ϵ-greedy strategy for the optimal cost-quality trade-off by mixing different feedback types including corrections, error markups, and self-supervision. Furthermore, we demonstrate its robustness under domain shift and identify it as a promising alternative to active learning.

| Comments: | ACL 2019                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (stat.ML) |
| Cite as:  | **arXiv:1907.05190 [cs.CL]**                                 |
|           | (or **arXiv:1907.05190v1 [cs.CL]** for this version)         |

<h2 id="2019-07-12-3">3. To Tune or Not To Tune? How About the Best of Both Worlds?</h2>
Title: [To Tune or Not To Tune? How About the Best of Both Worlds?](https://arxiv.org/abs/1907.05338)

Authors: [Ran Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+R), [Haibo Su](https://arxiv.org/search/cs?searchtype=author&query=Su%2C+H), [Chunye Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Kailin Ji](https://arxiv.org/search/cs?searchtype=author&query=Ji%2C+K), [Jupeng Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+J)

*(Submitted on 9 Jul 2019)*

> The introduction of pre-trained language models has revolutionized natural language research communities. However, researchers still know relatively little regarding their theoretical and empirical properties. In this regard, Peters et al. perform several experiments which demonstrate that it is better to adapt BERT with a light-weight task-specific head, rather than building a complex one on top of the pre-trained language model, and freeze the parameters in the said language model. However, there is another option to adopt. In this paper, we propose a new adaptation method which we first train the task model with the BERT parameters frozen and then fine-tune the entire model together. Our experimental results show that our model adaptation method can achieve 4.7% accuracy improvement in semantic similarity task, 0.99% accuracy improvement in sequence labeling task and 0.72% accuracy improvement in the text classification task.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1907.05338 [cs.CL]**                                 |
|           | (or **arXiv:1907.05338v1 [cs.CL]** for this version)         |





# 2019-07-11

[Return to Index](#Index)

<h2 id="2019-07-11-1">1. Learning to Speak Fluently in a Foreign Language: Multilingual Speech Synthesis and Cross-Language Voice Cloning</h2>
Title: [Learning to Speak Fluently in a Foreign Language: Multilingual Speech Synthesis and Cross-Language Voice Cloning](https://arxiv.org/abs/1907.04448)
Authors: [Yu Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Ron J. Weiss](https://arxiv.org/search/cs?searchtype=author&query=Weiss%2C+R+J), [Heiga Zen](https://arxiv.org/search/cs?searchtype=author&query=Zen%2C+H), [Yonghui Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+Y), [Zhifeng Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Z), [RJ Skerry-Ryan](https://arxiv.org/search/cs?searchtype=author&query=Skerry-Ryan%2C+R), [Ye Jia](https://arxiv.org/search/cs?searchtype=author&query=Jia%2C+Y), [Andrew Rosenberg](https://arxiv.org/search/cs?searchtype=author&query=Rosenberg%2C+A), [Bhuvana Ramabhadran](https://arxiv.org/search/cs?searchtype=author&query=Ramabhadran%2C+B)

*(Submitted on 9 Jul 2019)*

> We present a multispeaker, multilingual text-to-speech (TTS) synthesis model based on Tacotron that is able to produce high quality speech in multiple languages. Moreover, the model is able to transfer voices across languages, e.g. synthesize fluent Spanish speech using an English speaker's voice, without training on any bilingual or parallel examples. Such transfer works across distantly related languages, e.g. English and Mandarin. 
> Critical to achieving this result are: 1. using a phonemic input representation to encourage sharing of model capacity across languages, and 2. incorporating an adversarial loss term to encourage the model to disentangle its representation of speaker identity (which is perfectly correlated with language in the training data) from the speech content. Further scaling up the model by training on multiple speakers of each language, and incorporating an autoencoding input to help stabilize attention during training, results in a model which can be used to consistently synthesize intelligible speech for training speakers in all languages seen during training, and in native or foreign accents.

| Comments: | 5 pages, submitted to Interspeech 2019                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **arXiv:1907.04448 [cs.CL]**                                 |
|           | (or **arXiv:1907.04448v1 [cs.CL]** for this version)         |

<h2 id="2019-07-11-2">2. Lingua Custodia at WMT'19: Attempts to Control Terminology</h2>
Title: [Lingua Custodia at WMT'19: Attempts to Control Terminology](https://arxiv.org/abs/1907.04618)
Authors: [Franck Burlot](https://arxiv.org/search/cs?searchtype=author&query=Burlot%2C+F)

*(Submitted on 10 Jul 2019)*

> This paper describes Lingua Custodia's submission to the WMT'19 news shared task for German-to-French on the topic of the EU elections. We report experiments on the adaptation of the terminology of a machine translation system to a specific topic, aimed at providing more accurate translations of specific entities like political parties and person names, given that the shared task provided no in-domain training parallel data dealing with the restricted topic. Our primary submission to the shared task uses backtranslation generated with a type of decoding allowing the insertion of constraints in the output in order to guarantee the correct translation of specific terms that are not necessarily observed in the data.

| Comments: | Proceedings of the Fourth Conference on Machine Translation (WMT), pages72-79, Association for Computational Linguistics |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1907.04618 [cs.CL]**                                 |
|           | (or **arXiv:1907.04618v1 [cs.CL]** for this version)         |



# 2019-07-10

[Return to Index](#Index)

<h2 id="2019-07-10-1">1. Learning Neural Sequence-to-Sequence Models from Weak Feedback with Bipolar Ramp Loss</h2>
Title: [Learning Neural Sequence-to-Sequence Models from Weak Feedback with Bipolar Ramp Loss](https://arxiv.org/abs/1907.03748)

Authors: [Laura Jehl](https://arxiv.org/search/cs?searchtype=author&query=Jehl%2C+L), [Carolin Lawrence](https://arxiv.org/search/cs?searchtype=author&query=Lawrence%2C+C), [Stefan Riezler](https://arxiv.org/search/cs?searchtype=author&query=Riezler%2C+S)

*(Submitted on 6 Jul 2019)*

> In many machine learning scenarios, supervision by gold labels is not available and consequently neural models cannot be trained directly by maximum likelihood estimation (MLE). In a weak supervision scenario, metric-augmented objectives can be employed to assign feedback to model outputs, which can be used to extract a supervision signal for training. We present several objectives for two separate weakly supervised tasks, machine translation and semantic parsing. We show that objectives should actively discourage negative outputs in addition to promoting a surrogate gold structure. This notion of bipolarity is naturally present in ramp loss objectives, which we adapt to neural models. We show that bipolar ramp loss objectives outperform other non-bipolar ramp loss objectives and minimum risk training (MRT) on both weakly supervised tasks, as well as on a supervised machine translation task. Additionally, we introduce a novel token-level ramp loss objective, which is able to outperform even the best sequence-level ramp loss on both weakly supervised tasks.

| Comments: | Transactions of the Association for Computational Linguistics 2019 Vol. 7, 233-248. Presented at ACL, Florence, Italy |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Machine Learning (stat.ML) |
| Cite as:  | **arXiv:1907.03748 [cs.CL]**                                 |
|           | (or **arXiv:1907.03748v1 [cs.CL]** for this version)         |

<h2 id="2019-07-10-2">2. An Intrinsic Nearest Neighbor Analysis of Neural Machine Translation Architectures</h2>
Title: [An Intrinsic Nearest Neighbor Analysis of Neural Machine Translation Architectures](https://arxiv.org/abs/1907.03885)

Authors: [Hamidreza Ghader](https://arxiv.org/search/cs?searchtype=author&query=Ghader%2C+H), [Christof Monz](https://arxiv.org/search/cs?searchtype=author&query=Monz%2C+C)

*(Submitted on 8 Jul 2019)*

> Earlier approaches indirectly studied the information captured by the hidden states of recurrent and non-recurrent neural machine translation models by feeding them into different classifiers. In this paper, we look at the encoder hidden states of both transformer and recurrent machine translation models from the nearest neighbors perspective. We investigate to what extent the nearest neighbors share information with the underlying word embeddings as well as related WordNet entries. Additionally, we study the underlying syntactic structure of the nearest neighbors to shed light on the role of syntactic similarities in bringing the neighbors together. We compare transformer and recurrent models in a more intrinsic way in terms of capturing lexical semantics and syntactic structures, in contrast to extrinsic approaches used by previous works. In agreement with the extrinsic evaluations in the earlier works, our experimental results show that transformers are superior in capturing lexical semantics, but not necessarily better in capturing the underlying syntax. Additionally, we show that the backward recurrent layer in a recurrent model learns more about the semantics of words, whereas the forward recurrent layer encodes more context.

| Comments: | To be presented at Machine Translation Summit 2019 (MTSUMMIT XVII), Dublin, Ireland |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Neural and Evolutionary Computing (cs.NE) |
| Cite as:  | **arXiv:1907.03885 [cs.CL]**                                 |
|           | (or **arXiv:1907.03885v1 [cs.CL]** for this version)         |



<h2 id="2019-07-10-3">3. NTT's Machine Translation Systems for WMT19 Robustness Task</h2>
Title: [NTT's Machine Translation Systems for WMT19 Robustness Task](https://arxiv.org/abs/1907.03927)

Authors: [Soichiro Murakami](https://arxiv.org/search/cs?searchtype=author&query=Murakami%2C+S), [Makoto Morishita](https://arxiv.org/search/cs?searchtype=author&query=Morishita%2C+M), [Tsutomu Hirao](https://arxiv.org/search/cs?searchtype=author&query=Hirao%2C+T), [Masaaki Nagata](https://arxiv.org/search/cs?searchtype=author&query=Nagata%2C+M)

*(Submitted on 9 Jul 2019)*

> This paper describes NTT's submission to the WMT19 robustness task. This task mainly focuses on translating noisy text (e.g., posts on Twitter), which presents different difficulties from typical translation tasks such as news. Our submission combined techniques including utilization of a synthetic corpus, domain adaptation, and a placeholder mechanism, which significantly improved over the previous baseline. Experimental results revealed the placeholder mechanism, which temporarily replaces the non-standard tokens including emojis and emoticons with special placeholder tokens during translation, improves translation accuracy even with noisy texts.

| Comments: | submitted to WMT 2019                                |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1907.03927 [cs.CL]**                         |
|           | (or **arXiv:1907.03927v1 [cs.CL]** for this version) |



<h2 id="2019-07-10-4">4. Multilingual Universal Sentence Encoder for Semantic Retrieval</h2>
Title: [Multilingual Universal Sentence Encoder for Semantic Retrieval](https://arxiv.org/abs/1907.04307)

Authors: [Yinfei Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Y), [Daniel Cer](https://arxiv.org/search/cs?searchtype=author&query=Cer%2C+D), [Amin Ahmad](https://arxiv.org/search/cs?searchtype=author&query=Ahmad%2C+A), [Mandy Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+M), [Jax Law](https://arxiv.org/search/cs?searchtype=author&query=Law%2C+J), [Noah Constant](https://arxiv.org/search/cs?searchtype=author&query=Constant%2C+N), [Gustavo Hernandez Abrego](https://arxiv.org/search/cs?searchtype=author&query=Abrego%2C+G+H), [Steve Yuan](https://arxiv.org/search/cs?searchtype=author&query=Yuan%2C+S), [Chris Tar](https://arxiv.org/search/cs?searchtype=author&query=Tar%2C+C), [Yun-Hsuan Sung](https://arxiv.org/search/cs?searchtype=author&query=Sung%2C+Y), [Brian Strope](https://arxiv.org/search/cs?searchtype=author&query=Strope%2C+B), [Ray Kurzweil](https://arxiv.org/search/cs?searchtype=author&query=Kurzweil%2C+R)

*(Submitted on 9 Jul 2019)*

> We introduce two pre-trained retrieval focused multilingual sentence encoding models, respectively based on the Transformer and CNN model architectures. The models embed text from 16 languages into a single semantic space using a multi-task trained dual-encoder that learns tied representations using translation based bridge tasks (Chidambaram al., 2018). The models provide performance that is competitive with the state-of-the-art on: semantic retrieval (SR), translation pair bitext retrieval (BR) and retrieval question answering (ReQA). On English transfer learning tasks, our sentence-level embeddings approach, and in some cases exceed, the performance of monolingual, English only, sentence embedding models. Our models are made available for download on TensorFlow Hub.

| Comments: | 6 pages, 6 tables, 2 listings, and 1 figure          |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1907.04307 [cs.CL]**                         |
|           | (or **arXiv:1907.04307v1 [cs.CL]** for this version) |




# 2019-07-09

[Return to Index](#Index)

<h2 id="2019-07-09-1">1. Exploiting Out-of-Domain Parallel Data through Multilingual Transfer Learning for Low-Resource Neural Machine Translation</h2>
Title: [Exploiting Out-of-Domain Parallel Data through Multilingual Transfer Learning for Low-Resource Neural Machine Translation](https://arxiv.org/abs/1907.03060)

Authors: [Aizhan Imankulova](https://arxiv.org/search/cs?searchtype=author&query=Imankulova%2C+A), [Raj Dabre](https://arxiv.org/search/cs?searchtype=author&query=Dabre%2C+R), [Atsushi Fujita](https://arxiv.org/search/cs?searchtype=author&query=Fujita%2C+A), [Kenji Imamura](https://arxiv.org/search/cs?searchtype=author&query=Imamura%2C+K)

*(Submitted on 6 Jul 2019)*

> This paper proposes a novel multilingual multistage fine-tuning approach for low-resource neural machine translation (NMT), taking a challenging Japanese--Russian pair for benchmarking. Although there are many solutions for low-resource scenarios, such as multilingual NMT and back-translation, we have empirically confirmed their limited success when restricted to in-domain data. We therefore propose to exploit out-of-domain data through transfer learning, by using it to first train a multilingual NMT model followed by multistage fine-tuning on in-domain parallel and back-translated pseudo-parallel data. Our approach, which combines domain adaptation, multilingualism, and back-translation, helps improve the translation quality by more than 3.7 BLEU points, over a strong baseline, for this extremely low-resource scenario.

| Comments: | Accepted at the 17th Machine Translation Summit      |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1907.03060 [cs.CL]**                         |
|           | (or **arXiv:1907.03060v1 [cs.CL]** for this version) |



<h2 id="2019-07-09-2">2. Best Practices for Learning Domain-Specific Cross-Lingual Embeddings</h2>
Title: [Best Practices for Learning Domain-Specific Cross-Lingual Embeddings](https://arxiv.org/abs/1907.03112)

Authors: [Lena Shakurova](https://arxiv.org/search/cs?searchtype=author&query=Shakurova%2C+L), [Beata Nyari](https://arxiv.org/search/cs?searchtype=author&query=Nyari%2C+B), [Chao Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+C), [Mihai Rotaru](https://arxiv.org/search/cs?searchtype=author&query=Rotaru%2C+M)

*(Submitted on 6 Jul 2019)*

> Cross-lingual embeddings aim to represent words in multiple languages in a shared vector space by capturing semantic similarities across languages. They are a crucial component for scaling tasks to multiple languages by transferring knowledge from languages with rich resources to low-resource languages. A common approach to learning cross-lingual embeddings is to train monolingual embeddings separately for each language and learn a linear projection from the monolingual spaces into a shared space, where the mapping relies on a small seed dictionary. While there are high-quality generic seed dictionaries and pre-trained cross-lingual embeddings available for many language pairs, there is little research on how they perform on specialised tasks. In this paper, we investigate the best practices for constructing the seed dictionary for a specific domain. We evaluate the embeddings on the sequence labelling task of Curriculum Vitae parsing and show that the size of a bilingual dictionary, the frequency of the dictionary words in the domain corpora and the source of data (task-specific vs generic) influence the performance. We also show that the less training data is available in the low-resource language, the more the construction of the bilingual dictionary matters, and demonstrate that some of the choices are crucial in the zero-shot transfer learning case.

| Comments: | Proceedings of the 4th Workshop on Representation Learning for NLP |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1907.03112 [cs.CL]**                                 |
|           | (or **arXiv:1907.03112v1 [cs.CL]** for this version)         |

<h2 id="2019-07-09-3">3. Evolutionary Algorithm for Sinhala to English Translation</h2>
Title: [Evolutionary Algorithm for Sinhala to English Translation](https://arxiv.org/abs/1907.03202)

Authors: [J.K. Joseph](https://arxiv.org/search/cs?searchtype=author&query=Joseph%2C+J), [W.M.T. Chathurika](https://arxiv.org/search/cs?searchtype=author&query=Chathurika%2C+W), [A. Nugaliyadde](https://arxiv.org/search/cs?searchtype=author&query=Nugaliyadde%2C+A), [Y. Mallawarachchi](https://arxiv.org/search/cs?searchtype=author&query=Mallawarachchi%2C+Y)

*(Submitted on 6 Jul 2019)*

> Machine Translation (MT) is an area in natural language processing, which focus on translating from one language to another. Many approaches ranging from statistical methods to deep learning approaches are used in order to achieve MT. However, these methods either require a large number of data or a clear understanding about the language. Sinhala language has less digital text which could be used to train a deep neural network. Furthermore, Sinhala has complex rules therefore, it is harder to create statistical rules in order to apply statistical methods in MT. This research focuses on Sinhala to English translation using an Evolutionary Algorithm (EA). EA is used to identifying the correct meaning of Sinhala text and to translate it to English. The Sinhala text is passed to identify the meaning in order to get the correct meaning of the sentence. With the use of the EA the translation is carried out. The translated text is passed on to grammatically correct the sentence. This has shown to achieve accurate results.

| Comments: | The paper was submitted to National Information Technology Conference (2019) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Neural and Evolutionary Computing (cs.NE) |
| Cite as:  | **arXiv:1907.03202 [cs.CL]**                                 |
|           | (or **arXiv:1907.03202v1 [cs.CL]** for this version)         |

<h2 id="2019-07-09-4">4. Correct-and-Memorize: Learning to Translate from Interactive Revisions</h2>
Title: [Correct-and-Memorize: Learning to Translate from Interactive Revisions](https://arxiv.org/abs/1907.03468)

Authors: [Rongxiang Weng](https://arxiv.org/search/cs?searchtype=author&query=Weng%2C+R), [Hao Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+H), [Shujian Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Lei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Yifan Xia](https://arxiv.org/search/cs?searchtype=author&query=Xia%2C+Y), [Jiajun Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+J)

*(Submitted on 8 Jul 2019)*

> State-of-the-art machine translation models are still not on par with human translators. Previous work takes human interactions into the neural machine translation process to obtain improved results in target languages. However, not all model-translation errors are equal -- some are critical while others are minor. In the meanwhile, the same translation mistakes occur repeatedly in a similar context. To solve both issues, we propose CAMIT, a novel method for translating in an interactive environment. Our proposed method works with critical revision instructions, therefore allows human to correct arbitrary words in model-translated sentences. In addition, CAMIT learns from and softly memorizes revision actions based on the context, alleviating the issue of repeating mistakes. Experiments in both ideal and real interactive translation settings demonstrate that our proposed \method enhances machine translation results significantly while requires fewer revision instructions from human compared to previous methods.

| Comments: | Accepted at IJCAI 2019                               |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1907.03468 [cs.CL]**                         |
|           | (or **arXiv:1907.03468v1 [cs.CL]** for this version) |





# 2019-07-08

[Return to Index](#Index)

<h2 id="2019-07-08-1">1. Multi-lingual Intent Detection and Slot Filling in a Joint BERT-based Model</h2>
Title: [Multi-lingual Intent Detection and Slot Filling in a Joint BERT-based Model](https://arxiv.org/abs/1907.02884)

Authors: [Giuseppe Castellucci](https://arxiv.org/search/cs?searchtype=author&query=Castellucci%2C+G), [Valentina Bellomaria](https://arxiv.org/search/cs?searchtype=author&query=Bellomaria%2C+V), [Andrea Favalli](https://arxiv.org/search/cs?searchtype=author&query=Favalli%2C+A), [Raniero Romagnoli](https://arxiv.org/search/cs?searchtype=author&query=Romagnoli%2C+R)

*(Submitted on 5 Jul 2019)*

> Intent Detection and Slot Filling are two pillar tasks in Spoken Natural Language Understanding. Common approaches adopt joint Deep Learning architectures in attention-based recurrent frameworks. In this work, we aim at exploiting the success of "recurrence-less" models for these tasks. We introduce Bert-Joint, i.e., a multi-lingual joint text classification and sequence labeling framework. The experimental evaluation over two well-known English benchmarks demonstrates the strong performances that can be obtained with this model, even when few annotated data is available. Moreover, we annotated a new dataset for the Italian language, and we observed similar performances without the need for changing the model.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1907.02884 [cs.CL]**                                 |
|           | (or **arXiv:1907.02884v1 [cs.CL]** for this version)         |



# 2019-07-03

[Return to Index](#Index)

<h2 id="2019-07-03-1">1. A Neural Grammatical Error Correction System Built On Better Pre-training and Sequential Transfer Learning</h2>
Title: [A Neural Grammatical Error Correction System Built On Better Pre-training and Sequential Transfer Learning](https://arxiv.org/abs/1907.01256)
Authors:  [Yo Joong Choe](https://arxiv.org/search/cs?searchtype=author&query=Choe%2C+Y+J), [Jiyeon Ham](https://arxiv.org/search/cs?searchtype=author&query=Ham%2C+J), [Kyubyong Park](https://arxiv.org/search/cs?searchtype=author&query=Park%2C+K), [Yeoil Yoon](https://arxiv.org/search/cs?searchtype=author&query=Yoon%2C+Y)

*(Submitted on 2 Jul 2019)*

> Grammatical error correction can be viewed as a low-resource sequence-to-sequence task, because publicly available parallel corpora are limited. To tackle this challenge, we first generate erroneous versions of large unannotated corpora using a realistic noising function. The resulting parallel corpora are subsequently used to pre-train Transformer models. Then, by sequentially applying transfer learning, we adapt these models to the domain and style of the test set. Combined with a context-aware neural spellchecker, our system achieves competitive results in both restricted and low resource tracks in ACL 2019 BEA Shared Task. We release all of our code and materials for reproducibility.

| Comments: | Accepted to ACL 2019 Workshop on Innovative Use of NLP for Building Educational Applications (BEA) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1907.01256 [cs.CL]**                                 |
|           | (or **arXiv:1907.01256v1 [cs.CL]** for this version)         |


<h2 id="2019-07-03-2">2. Improving Robustness in Real-World Neural Machine Translation Engines</h2>
Title: [Improving Robustness in Real-World Neural Machine Translation Engines](https://arxiv.org/abs/1907.01279)
Authors:  [Rohit Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+R), [Patrik Lambert](https://arxiv.org/search/cs?searchtype=author&query=Lambert%2C+P), [Raj Nath Patel](https://arxiv.org/search/cs?searchtype=author&query=Patel%2C+R+N), [John Tinsley](https://arxiv.org/search/cs?searchtype=author&query=Tinsley%2C+J)

*(Submitted on 2 Jul 2019)*

> As a commercial provider of machine translation, we are constantly training engines for a variety of uses, languages, and content types. In each case, there can be many variables, such as the amount of training data available, and the quality requirements of the end user. These variables can have an impact on the robustness of Neural MT engines. On the whole, Neural MT cures many ills of other MT paradigms, but at the same time, it has introduced a new set of challenges to address. In this paper, we describe some of the specific issues with practical NMT and the approaches we take to improve model robustness in real-world scenarios.

| Comments: | 6 Pages, Accepted in Machine Translation Summit 2019 |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1907.01279 [cs.CL]**                         |
|           | (or **arXiv:1907.01279v1 [cs.CL]** for this version) |



# 2019-07-02

[Return to Index](#Index)

<h2 id="2019-07-02-1">1. The University of Sydney's Machine Translation System for WMT19</h2>
Title: [The University of Sydney's Machine Translation System for WMT19](https://arxiv.org/abs/1907.00494)

Authors:[Liang Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+L), [Dacheng Tao](https://arxiv.org/search/cs?searchtype=author&query=Tao%2C+D)

*(Submitted on 30 Jun 2019)*

> This paper describes the University of Sydney's submission of the WMT 2019 shared news translation task. We participated in the Finnish→English direction and got the best BLEU(33.0) score among all the participants. Our system is based on the self-attentional Transformer networks, into which we integrated the most recent effective strategies from academic research (e.g., BPE, back translation, multi-features data selection, data augmentation, greedy model ensemble, reranking, ConMBR system combination, and post-processing). Furthermore, we propose a novel augmentation method CycleTranslation and a data mixture strategy Big/Small parallel construction to entirely exploit the synthetic corpus. Extensive experiments show that adding the above techniques can make continuous improvements of the BLEU scores, and the best result outperforms the baseline (Transformer ensemble model trained with the original parallel corpus) by approximately 5.3 BLEU score, achieving the state-of-the-art performance.

| Comments: | To appear in WMT2019                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1907.00494 [cs.CL]**                                 |
|           | (or **arXiv:1907.00494v1 [cs.CL]** for this version)         |

<h2 id="2019-07-02-2">2. Few-Shot Representation Learning for Out-Of-Vocabulary Words</h2>
Title: [Few-Shot Representation Learning for Out-Of-Vocabulary Words](https://arxiv.org/abs/1907.00505)

Authors: [Ziniu Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+Z), [Ting Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+T), [Kai-Wei Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang%2C+K), [Yizhou Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+Y)

*(Submitted on 1 Jul 2019)*

> Existing approaches for learning word embeddings often assume there are sufficient occurrences for each word in the corpus, such that the representation of words can be accurately estimated from their contexts. However, in real-world scenarios, out-of-vocabulary (a.k.a. OOV) words that do not appear in training corpus emerge frequently. It is challenging to learn accurate representations of these words with only a few observations. In this paper, we formulate the learning of OOV embeddings as a few-shot regression problem, and address it by training a representation function to predict the oracle embedding vector (defined as embedding trained with abundant observations) based on limited observations. Specifically, we propose a novel hierarchical attention-based architecture to serve as the neural regression function, with which the context information of a word is encoded and aggregated from K observations. Furthermore, our approach can leverage Model-Agnostic Meta-Learning (MAML) for adapting the learned model to the new corpus fast and robustly. Experiments show that the proposed approach significantly outperforms existing methods in constructing accurate embeddings for OOV words, and improves downstream tasks where these embeddings are utilized.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1907.00505 [cs.CL]**                         |
|           | (or **arXiv:1907.00505v1 [cs.CL]** for this version) |

<h2 id="2019-07-02-3">3. From Bilingual to Multilingual Neural Machine Translation by Incremental Training</h2>
Title: [From Bilingual to Multilingual Neural Machine Translation by Incremental Training](https://arxiv.org/abs/1907.00735)

Authors: [Carlos Escolano](https://arxiv.org/search/cs?searchtype=author&query=Escolano%2C+C), [Marta R. Costa-Jussà](https://arxiv.org/search/cs?searchtype=author&query=Costa-Jussà%2C+M+R), [José A. R. Fonollosa](https://arxiv.org/search/cs?searchtype=author&query=Fonollosa%2C+J+A+R)

*(Submitted on 28 Jun 2019)*

> Multilingual Neural Machine Translation approaches are based on the use of task-specific models and the addition of one more language can only be done by retraining the whole system. In this work, we propose a new training schedule that allows the system to scale to more languages without modification of the previous components based on joint training and language-independent encoder/decoder modules allowing for zero-shot translation. This work in progress shows close results to the state-of-the-art in the WMT task.

| Comments: | Accepted paper at ACL 2019 Student Research Workshop. arXiv admin note: substantial text overlap with [arXiv:1905.06831](https://arxiv.org/abs/1905.06831) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1907.00735 [cs.CL]**                                 |
|           | (or **arXiv:1907.00735v1 [cs.CL]** for this version)         |

<h2 id="2019-07-02-4">4. Post-editese: an Exacerbated Translationese</h2>
Title: [Post-editese: an Exacerbated Translationese](https://arxiv.org/abs/1907.00900)

Authors: [Antonio Toral](https://arxiv.org/search/cs?searchtype=author&query=Toral%2C+A)

*(Submitted on 1 Jul 2019)*

> Post-editing (PE) machine translation (MT) is widely used for dissemination because it leads to higher productivity than human translation from scratch (HT). In addition, PE translations are found to be of equal or better quality than HTs. However, most such studies measure quality solely as the number of errors. We conduct a set of computational analyses in which we compare PE against HT on three different datasets that cover five translation directions with measures that address different translation universals and laws of translation: simplification, normalisation and interference. We find out that PEs are simpler and more normalised and have a higher degree of interference from the source language than HTs.

| Comments: | Accepted at the 17th Machine Translation Summit      |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1907.00900 [cs.CL]**                         |
|           | (or **arXiv:1907.00900v1 [cs.CL]** for this version) |




# 2019-07-01

[Return to Index](#Index)

<h2 id="2019-07-01-1">1. Findings of the First Shared Task on Machine Translation Robustness</h2>
Title: [Findings of the First Shared Task on Machine Translation Robustness](https://arxiv.org/abs/1906.11943)

Authors: [Xian Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+X), [Paul Michel](https://arxiv.org/search/cs?searchtype=author&query=Michel%2C+P), [Antonios Anastasopoulos](https://arxiv.org/search/cs?searchtype=author&query=Anastasopoulos%2C+A), [Yonatan Belinkov](https://arxiv.org/search/cs?searchtype=author&query=Belinkov%2C+Y), [Nadir Durrani](https://arxiv.org/search/cs?searchtype=author&query=Durrani%2C+N), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O), [Philipp Koehn](https://arxiv.org/search/cs?searchtype=author&query=Koehn%2C+P), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G), [Juan Pino](https://arxiv.org/search/cs?searchtype=author&query=Pino%2C+J), [Hassan Sajjad](https://arxiv.org/search/cs?searchtype=author&query=Sajjad%2C+H)

*(Submitted on 27 Jun 2019)*

> We share the findings of the first shared task on improving robustness of Machine Translation (MT). The task provides a testbed representing challenges facing MT models deployed in the real world, and facilitates new approaches to improve models; robustness to noisy input and domain mismatch. We focus on two language pairs (English-French and English-Japanese), and the submitted systems are evaluated on a blind test set consisting of noisy comments on Reddit and professionally sourced translations. As a new task, we received 23 submissions by 11 participating teams from universities, companies, national labs, etc. All submitted systems achieved large improvements over baselines, with the best improvement having +22.33 BLEU. We evaluated submissions by both human judgment and automatic evaluation (BLEU), which shows high correlations (Pearson's r = 0.94 and 0.95). Furthermore, we conducted a qualitative analysis of the submitted systems using compare-mt, which revealed their salient differences in handling challenges in this task. Such analysis provides additional insights when there is occasional disagreement between human judgment and BLEU, e.g. systems better at producing colloquial expressions received higher score from human judgment.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1906.11943 [cs.CL]**                         |
|           | (or **arXiv:1906.11943v1 [cs.CL]** for this version) |

<h2 id="2019-07-01-2">2. Lost in Translation: Loss and Decay of Linguistic Richness in Machine Translation</h2>
Title: [Lost in Translation: Loss and Decay of Linguistic Richness in Machine Translation](https://arxiv.org/abs/1906.12068)

Authors: [Eva Vanmassenhove](https://arxiv.org/search/cs?searchtype=author&query=Vanmassenhove%2C+E), [Dimitar Shterionov](https://arxiv.org/search/cs?searchtype=author&query=Shterionov%2C+D), [Andy Way](https://arxiv.org/search/cs?searchtype=author&query=Way%2C+A)

*(Submitted on 28 Jun 2019)*

> This work presents an empirical approach to quantifying the loss of lexical richness in Machine Translation (MT) systems compared to Human Translation (HT). Our experiments show how current MT systems indeed fail to render the lexical diversity of human generated or translated text. The inability of MT systems to generate diverse outputs and its tendency to exacerbate already frequent patterns while ignoring less frequent ones, might be the underlying cause for, among others, the currently heavily debated issues related to gender biased output. Can we indeed, aside from biased data, talk about an algorithm that exacerbates seen biases?

| Comments: | Accepted for publication at the 17th Machine Translation Summit (MTSummit2019), Dublin, Ireland, August 2019 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1906.12068 [cs.CL]**                                 |
|           | (or **arXiv:1906.12068v1 [cs.CL]** for this version)         |

<h2 id="2019-07-01-3">3. Widening the Representation Bottleneck in Neural Machine Translation with Lexical Shortcuts</h2>
Title: [Widening the Representation Bottleneck in Neural Machine Translation with Lexical Shortcuts](https://arxiv.org/abs/1906.12284)

Authors: [Denis Emelin](https://arxiv.org/search/cs?searchtype=author&query=Emelin%2C+D), [Ivan Titov](https://arxiv.org/search/cs?searchtype=author&query=Titov%2C+I), [Rico Sennrich](https://arxiv.org/search/cs?searchtype=author&query=Sennrich%2C+R)

*(Submitted on 28 Jun 2019)*

> The transformer is a state-of-the-art neural translation model that uses attention to iteratively refine lexical representations with information drawn from the surrounding context. Lexical features are fed into the first layer and propagated through a deep network of hidden layers. We argue that the need to represent and propagate lexical features in each layer limits the model's capacity for learning and representing other information relevant to the task. To alleviate this bottleneck, we introduce gated shortcut connections between the embedding layer and each subsequent layer within the encoder and decoder. This enables the model to access relevant lexical content dynamically, without expending limited resources on storing it within intermediate states. We show that the proposed modification yields consistent improvements over a baseline transformer on standard WMT translation tasks in 5 translation directions (0.9 BLEU on average) and reduces the amount of lexical information passed along the hidden layers. We furthermore evaluate different ways to integrate lexical connections into the transformer architecture and present ablation experiments exploring the effect of proposed shortcuts on model behavior.

| Comments: | Accepted submission to WMT 2019                              |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1906.12284 [cs.CL]**                                 |
|           | (or **arXiv:1906.12284v1 [cs.CL]** for this version)         |