# Daily arXiv: Machine Translation - Jul., 2019

### Index

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