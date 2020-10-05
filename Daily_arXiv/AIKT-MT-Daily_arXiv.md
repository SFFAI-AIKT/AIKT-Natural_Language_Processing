# Daily arXiv: Machine Translation - October, 2020

# Index


- [2020-10-05](#2020-10-05)

  - [1. Nearest Neighbor Machine Translation](#2020-10-05-1)
  - [2. A Survey of the State of Explainable AI for Natural Language Processing](#2020-10-05-2)
  - [3. An Empirical Investigation Towards Efficient Multi-Domain Language Model Pre-training](#2020-10-05-3)
  - [4. Which *BERT? A Survey Organizing Contextualized Encoders](#2020-10-05-4)


- [2020-10-02](#2020-10-2)

  - [1. WeChat Neural Machine Translation Systems for WMT20](#2020-10-2-1)
- [2020-10-01](#2020-10-01)

  - [1. Rethinking Attention with Performers](#2020-10-01-1)
  - [2. Cross-lingual Alignment Methods for Multilingual BERT: A Comparative Study](#2020-10-01-2)
  - [3. Can Automatic Post-Editing Improve NMT?](#2020-10-01-3)
  - [4. Cross-lingual Spoken Language Understanding with Regularized Representation Alignment](#2020-10-01-4)
  - [5. On Romanization for Model Transfer Between Scripts in Neural Machine Translation](#2020-10-01-5)
- [2020-09](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-09.md)
- [2020-08](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-08.md)
- [2020-07](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-07.md)
- [2020-06](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-06.md)
- [2020-05](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-05.md)
- [2020-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-04.md)
- [2020-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-03.md)
- [2020-02](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-02.md)
- [2020-01](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-01.md)
- [2019-12](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-12.md)
- [2019-11](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-11.md)
- [2019-10](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-10.md)
- [2019-09](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-09.md)
- [2019-08](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-08.md)
- [2019-07](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-07.md)
- [2019-06](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-06.md)
- [2019-05](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-05.md)
- [2019-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-04.md)
- [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)



# 2020-10-05

[Return to Index](#Index)



<h2 id="2020-10-05-1">1. Nearest Neighbor Machine Translation</h2>

Title: [Nearest Neighbor Machine Translation](https://arxiv.org/abs/2010.00710)

Authors: [Urvashi Khandelwal](https://arxiv.org/search/cs?searchtype=author&query=Khandelwal%2C+U), [Angela Fan](https://arxiv.org/search/cs?searchtype=author&query=Fan%2C+A), [Dan Jurafsky](https://arxiv.org/search/cs?searchtype=author&query=Jurafsky%2C+D), [Luke Zettlemoyer](https://arxiv.org/search/cs?searchtype=author&query=Zettlemoyer%2C+L), [Mike Lewis](https://arxiv.org/search/cs?searchtype=author&query=Lewis%2C+M)

> We introduce k-nearest-neighbor machine translation (kNN-MT), which predicts tokens with a nearest neighbor classifier over a large datastore of cached examples, using representations from a neural translation model for similarity search. This approach requires no additional training and scales to give the decoder direct access to billions of examples at test time, resulting in a highly expressive model that consistently improves performance across many settings. Simply adding nearest neighbor search improves a state-of-the-art German-English translation model by 1.5 BLEU. kNN-MT allows a single model to be adapted to diverse domains by using a domain-specific datastore, improving results by an average of 9.2 BLEU over zero-shot transfer, and achieving new state-of-the-art results---without training on these domains. A massively multilingual model can also be specialized for particular language pairs, with improvements of 3 BLEU for translating from English into German and Chinese. Qualitatively, kNN-MT is easily interpretable; it combines source and target context to retrieve highly relevant examples.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2010.00710](https://arxiv.org/abs/2010.00710) [cs.CL]** |
|           | (or **[arXiv:2010.00710v1](https://arxiv.org/abs/2010.00710v1) [cs.CL]** for this version) |





<h2 id="2020-10-05-2">2. A Survey of the State of Explainable AI for Natural Language Processing</h2>

Title: [A Survey of the State of Explainable AI for Natural Language Processing](https://arxiv.org/abs/2010.00711)

Authors: [Marina Danilevsky](https://arxiv.org/search/cs?searchtype=author&query=Danilevsky%2C+M), [Kun Qian](https://arxiv.org/search/cs?searchtype=author&query=Qian%2C+K), [Ranit Aharonov](https://arxiv.org/search/cs?searchtype=author&query=Aharonov%2C+R), [Yannis Katsis](https://arxiv.org/search/cs?searchtype=author&query=Katsis%2C+Y), [Ban Kawas](https://arxiv.org/search/cs?searchtype=author&query=Kawas%2C+B), [Prithviraj Sen](https://arxiv.org/search/cs?searchtype=author&query=Sen%2C+P)

> Recent years have seen important advances in the quality of state-of-the-art models, but this has come at the expense of models becoming less interpretable. This survey presents an overview of the current state of Explainable AI (XAI), considered within the domain of Natural Language Processing (NLP). We discuss the main categorization of explanations, as well as the various ways explanations can be arrived at and visualized. We detail the operations and explainability techniques currently available for generating explanations for NLP model predictions, to serve as a resource for model developers in the community. Finally, we point out the current gaps and encourage directions for future work in this important research area.

| Comments:    | To appear in AACL-IJCNLP 2020                                |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| ACM classes: | I.2.7                                                        |
| Cite as:     | **[arXiv:2010.00711](https://arxiv.org/abs/2010.00711) [cs.CL]** |
|              | (or **[arXiv:2010.00711v1](https://arxiv.org/abs/2010.00711v1) [cs.CL]** for this version) |





<h2 id="2020-10-05-3">3. An Empirical Investigation Towards Efficient Multi-Domain Language Model Pre-training</h2>

Title: [An Empirical Investigation Towards Efficient Multi-Domain Language Model Pre-training](https://arxiv.org/abs/2010.00784)

Authors: [Kristjan Arumae](https://arxiv.org/search/cs?searchtype=author&query=Arumae%2C+K), [Qing Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+Q), [Parminder Bhatia](https://arxiv.org/search/cs?searchtype=author&query=Bhatia%2C+P)

> Pre-training large language models has become a standard in the natural language processing community. Such models are pre-trained on generic data (e.g. BookCorpus and English Wikipedia) and often fine-tuned on tasks in the same domain. However, in order to achieve state-of-the-art performance on out of domain tasks such as clinical named entity recognition and relation extraction, additional in domain pre-training is required. In practice, staged multi-domain pre-training presents performance deterioration in the form of catastrophic forgetting (CF) when evaluated on a generic benchmark such as GLUE. In this paper we conduct an empirical investigation into known methods to mitigate CF. We find that elastic weight consolidation provides best overall scores yielding only a 0.33% drop in performance across seven generic tasks while remaining competitive in bio-medical tasks. Furthermore, we explore gradient and latent clustering based data selection techniques to improve coverage when using elastic weight consolidation and experience replay methods.

| Comments: | arXiv admin note: text overlap with [arXiv:2004.03794](https://arxiv.org/abs/2004.03794) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2010.00784](https://arxiv.org/abs/2010.00784) [cs.CL]** |
|           | (or **[arXiv:2010.00784v1](https://arxiv.org/abs/2010.00784v1) [cs.CL]** for this version) |





<h2 id="2020-10-05-4">4. Which *BERT? A Survey Organizing Contextualized Encoders</h2>

Title: [Which *BERT? A Survey Organizing Contextualized Encoders](https://arxiv.org/abs/2010.00854)

Authors: [Patrick Xia](https://arxiv.org/search/cs?searchtype=author&query=Xia%2C+P), [Shijie Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+S), [Benjamin Van Durme](https://arxiv.org/search/cs?searchtype=author&query=Van+Durme%2C+B)

> Pretrained contextualized text encoders are now a staple of the NLP community. We present a survey on language representation learning with the aim of consolidating a series of shared lessons learned across a variety of recent efforts. While significant advancements continue at a rapid pace, we find that enough has now been discovered, in different directions, that we can begin to organize advances according to common themes. Through this organization, we highlight important considerations when interpreting recent contributions and choosing which model to use.

| Comments: | EMNLP 2020                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2010.00854](https://arxiv.org/abs/2010.00854) [cs.CL]** |
|           | (or **[arXiv:2010.00854v1](https://arxiv.org/abs/2010.00854v1) [cs.CL]** for this version) |



# 2020-10-02

[Return to Index](#Index)



<h2 id="2020-10-02-1">1. WeChat Neural Machine Translation Systems for WMT20</h2>

Title: [WeChat Neural Machine Translation Systems for WMT20](https://arxiv.org/abs/2010.00247)

Authors: [Fandong Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+F), [Jianhao Yan](https://arxiv.org/search/cs?searchtype=author&query=Yan%2C+J), [Yijin Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Yuan Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+Y), [Xianfeng Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+X), [Qinsong Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+Q), [Peng Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+P), [Ming Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+M), [Jie Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J), [Sifan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+S), [Hao Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+H)

> We participate in the WMT 2020 shared news translation task on Chinese to English. Our system is based on the Transformer (Vaswani et al., 2017a) with effective variants and the DTMT (Meng and Zhang, 2019) architecture. In our experiments, we employ data selection, several synthetic data generation approaches (i.e., back-translation, knowledge distillation, and iterative in-domain knowledge transfer), advanced finetuning approaches and self-bleu based model ensemble. Our constrained Chinese to English system achieves 36.9 case-sensitive BLEU score, which is the highest among all submissions.

| Comments: | Accepted at WMT 2020. Our Chinese to English system achieved the highest case-sensitive BLEU score among all submissions |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2010.00247](https://arxiv.org/abs/2010.00247) [cs.CL]** |
|           | (or **[arXiv:2010.00247v1](https://arxiv.org/abs/2010.00247v1) [cs.CL]** for this version) |



# 2020-10-01

[Return to Index](#Index)



<h2 id="2020-10-01-1">1. Rethinking Attention with Performers</h2>

Title: [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)

Authors: [Krzysztof Choromanski](https://arxiv.org/search/cs?searchtype=author&query=Choromanski%2C+K), [Valerii Likhosherstov](https://arxiv.org/search/cs?searchtype=author&query=Likhosherstov%2C+V), [David Dohan](https://arxiv.org/search/cs?searchtype=author&query=Dohan%2C+D), [Xingyou Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+X), [Andreea Gane](https://arxiv.org/search/cs?searchtype=author&query=Gane%2C+A), [Tamas Sarlos](https://arxiv.org/search/cs?searchtype=author&query=Sarlos%2C+T), [Peter Hawkins](https://arxiv.org/search/cs?searchtype=author&query=Hawkins%2C+P), [Jared Davis](https://arxiv.org/search/cs?searchtype=author&query=Davis%2C+J), [Afroz Mohiuddin](https://arxiv.org/search/cs?searchtype=author&query=Mohiuddin%2C+A), [Lukasz Kaiser](https://arxiv.org/search/cs?searchtype=author&query=Kaiser%2C+L), [David Belanger](https://arxiv.org/search/cs?searchtype=author&query=Belanger%2C+D), [Lucy Colwell](https://arxiv.org/search/cs?searchtype=author&query=Colwell%2C+L), [Adrian Weller](https://arxiv.org/search/cs?searchtype=author&query=Weller%2C+A)

> We introduce Performers, Transformer architectures which can estimate regular (softmax) full-rank-attention Transformers with provable accuracy, but using only linear (as opposed to quadratic) space and time complexity, without relying on any priors such as sparsity or low-rankness. To approximate softmax attention-kernels, Performers use a novel Fast Attention Via positive Orthogonal Random features approach (FAVOR+), which may be of independent interest for scalable kernel methods. FAVOR+ can be also used to efficiently model kernelizable attention mechanisms beyond softmax. This representational power is crucial to accurately compare softmax with other kernels for the first time on large-scale tasks, beyond the reach of regular Transformers, and investigate optimal attention-kernels. Performers are linear architectures fully compatible with regular Transformers and with strong theoretical guarantees: unbiased or nearly-unbiased estimation of the attention matrix, uniform convergence and low estimation variance. We tested Performers on a rich set of tasks stretching from pixel-prediction through text models to protein sequence modeling. We demonstrate competitive results with other examined efficient sparse and dense attention methods, showcasing effectiveness of the novel attention-learning paradigm leveraged by Performers.

| Comments: | 36 pages. This is an updated version of a previous submission which can be found at [arXiv:2006.03555](https://arxiv.org/abs/2006.03555). See [this https URL](https://github.com/google-research/google-research/tree/master/protein_lm) for protein language model code, and [this https URL](https://github.com/google-research/google-research/tree/master/performer) for Performer code |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Machine Learning (stat.ML) |
| Cite as:  | **[arXiv:2009.14794](https://arxiv.org/abs/2009.14794) [cs.LG]** |
|           | (or **[arXiv:2009.14794v1](https://arxiv.org/abs/2009.14794v1) [cs.LG]** for this version) |





<h2 id="2020-10-01-2">2. Cross-lingual Alignment Methods for Multilingual BERT: A Comparative Study</h2>

Title: [Cross-lingual Alignment Methods for Multilingual BERT: A Comparative Study](https://arxiv.org/abs/2009.14304)

Authors: [Saurabh Kulshreshtha](https://arxiv.org/search/cs?searchtype=author&query=Kulshreshtha%2C+S), [José Luis Redondo-García](https://arxiv.org/search/cs?searchtype=author&query=Redondo-García%2C+J+L), [Ching-Yun Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang%2C+C)

> Multilingual BERT (mBERT) has shown reasonable capability for zero-shot cross-lingual transfer when fine-tuned on downstream tasks. Since mBERT is not pre-trained with explicit cross-lingual supervision, transfer performance can further be improved by aligning mBERT with cross-lingual signal. Prior work proposes several approaches to align contextualised embeddings. In this paper we analyse how different forms of cross-lingual supervision and various alignment methods influence the transfer capability of mBERT in zero-shot setting. Specifically, we compare parallel corpora vs. dictionary-based supervision and rotational vs. fine-tuning based alignment methods. We evaluate the performance of different alignment methodologies across eight languages on two tasks: Name Entity Recognition and Semantic Slot Filling. In addition, we propose a novel normalisation method which consistently improves the performance of rotation-based alignment including a notable 3% F1 improvement for distant and typologically dissimilar languages. Importantly we identify the biases of the alignment methods to the type of task and proximity to the transfer language. We also find that supervision from parallel corpus is generally superior to dictionary alignments.

| Comments: | Accepted as a long paper in Findings of EMNLP 2020           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2009.14304](https://arxiv.org/abs/2009.14304) [cs.CL]** |
|           | (or **[arXiv:2009.14304v1](https://arxiv.org/abs/2009.14304v1) [cs.CL]** for this version) |





<h2 id="2020-10-01-3">3. Can Automatic Post-Editing Improve NMT?</h2>

Title: [Can Automatic Post-Editing Improve NMT?](https://arxiv.org/abs/2009.14395)

Authors: [Shamil Chollampatt](https://arxiv.org/search/cs?searchtype=author&query=Chollampatt%2C+S), [Raymond Hendy Susanto](https://arxiv.org/search/cs?searchtype=author&query=Susanto%2C+R+H), [Liling Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+L), [Ewa Szymanska](https://arxiv.org/search/cs?searchtype=author&query=Szymanska%2C+E)

> Automatic post-editing (APE) aims to improve machine translations, thereby reducing human post-editing effort. APE has had notable success when used with statistical machine translation (SMT) systems but has not been as successful over neural machine translation (NMT) systems. This has raised questions on the relevance of APE task in the current scenario. However, the training of APE models has been heavily reliant on large-scale artificial corpora combined with only limited human post-edited data. We hypothesize that APE models have been underperforming in improving NMT translations due to the lack of adequate supervision. To ascertain our hypothesis, we compile a larger corpus of human post-edits of English to German NMT. We empirically show that a state-of-art neural APE model trained on this corpus can significantly improve a strong in-domain NMT system, challenging the current understanding in the field. We further investigate the effects of varying training data sizes, using artificial training data, and domain specificity for the APE task. We release this new corpus under CC BY-NC-SA 4.0 license at [this https URL](https://github.com/shamilcm/pedra).

| Comments: | In EMNLP 2020                                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2009.14395](https://arxiv.org/abs/2009.14395) [cs.CL]** |
|           | (or **[arXiv:2009.14395v1](https://arxiv.org/abs/2009.14395v1) [cs.CL]** for this version) |





<h2 id="2020-10-01-4">4. Cross-lingual Spoken Language Understanding with Regularized Representation Alignment</h2>

Title: [Cross-lingual Spoken Language Understanding with Regularized Representation Alignment](https://arxiv.org/abs/2009.14510)

Authors: [Zihan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Genta Indra Winata](https://arxiv.org/search/cs?searchtype=author&query=Winata%2C+G+I), [Peng Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+P), [Zhaojiang Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+Z), [Pascale Fung](https://arxiv.org/search/cs?searchtype=author&query=Fung%2C+P)

> Despite the promising results of current cross-lingual models for spoken language understanding systems, they still suffer from imperfect cross-lingual representation alignments between the source and target languages, which makes the performance sub-optimal. To cope with this issue, we propose a regularization approach to further align word-level and sentence-level representations across languages without any external resource. First, we regularize the representation of user utterances based on their corresponding labels. Second, we regularize the latent variable model (Liu et al., 2019) by leveraging adversarial training to disentangle the latent variables. Experiments on the cross-lingual spoken language understanding task show that our model outperforms current state-of-the-art methods in both few-shot and zero-shot scenarios, and our model, trained on a few-shot setting with only 3\% of the target language training data, achieves comparable performance to the supervised training with all the training data.

| Comments: | EMNLP-2020 Long Paper                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2009.14510](https://arxiv.org/abs/2009.14510) [cs.CL]** |
|           | (or **[arXiv:2009.14510v1](https://arxiv.org/abs/2009.14510v1) [cs.CL]** for this version) |





<h2 id="2020-10-01-5">5. On Romanization for Model Transfer Between Scripts in Neural Machine Translation</h2>

Title: [On Romanization for Model Transfer Between Scripts in Neural Machine Translation](https://arxiv.org/abs/2009.14824)

Authors: [Chantal Amrhein](https://arxiv.org/search/cs?searchtype=author&query=Amrhein%2C+C), [Rico Sennrich](https://arxiv.org/search/cs?searchtype=author&query=Sennrich%2C+R)

> Transfer learning is a popular strategy to improve the quality of low-resource machine translation. For an optimal transfer of the embedding layer, the child and parent model should share a substantial part of the vocabulary. This is not the case when transferring to languages with a different script. We explore the benefit of romanization in this scenario. Our results show that romanization entails information loss and is thus not always superior to simpler vocabulary transfer methods, but can improve the transfer between related languages with different scripts. We compare two romanization tools and find that they exhibit different degrees of information loss, which affects translation quality. Finally, we extend romanization to the target side, showing that this can be a successful strategy when coupled with a simple deromanization model.

| Comments: | accepted at Findings of EMNLP 2020                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2009.14824](https://arxiv.org/abs/2009.14824) [cs.CL]** |
|           | (or **[arXiv:2009.14824v1](https://arxiv.org/abs/2009.14824v1) [cs.CL]** for this version) |

