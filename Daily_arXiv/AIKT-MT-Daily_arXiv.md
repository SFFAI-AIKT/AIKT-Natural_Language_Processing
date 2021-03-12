# Daily arXiv: Machine Translation - February, 2021

# Index

- [2021-03-12](#2021-03-12)
  - [1. FairFil: Contrastive Neural Debiasing Method for Pretrained Text Encoders](#2021-03-12-1)
  - [2. LightMBERT: A Simple Yet Effective Method for Multilingual BERT Distillation](#2021-03-12-2)
  - [3. Towards Multi-Sense Cross-Lingual Alignment of Contextual Embeddings](#2021-03-12-3)
  - [4. Active2 Learning: Actively reducing redundancies in Active Learning methods for Sequence Tagging and Machine Translation](#2021-03-12-4)
  - [5. The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language Models](#2021-03-12-5)
  - [6. Unsupervised Transfer Learning in Multilingual Neural Machine Translation with Cross-Lingual Word Embeddings](#2021-03-12-6)
  - [7. Towards Continual Learning for Multilingual Machine Translation via Vocabulary Substitution](#2021-03-12-7)
  - [8. CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](#2021-03-12-8)
- [2021-03-11](#2021-03-11)
  - [1. Self-Learning for Zero Shot Neural Machine Translation](#2021-03-11-1)
  - [2. CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review](#2021-03-11-2)
- [2021-03-10](#2021-03-10)
  - [1. AfriVEC: Word Embedding Models for African Languages. Case Study of Fon and Nobiin](#2021-03-10-1)
- [2021-03-09](#2021-03-09)
  - [1. Translating the Unseen? Yorùbá → English MT in Low-Resource, Morphologically-Unmarked Settings](#2021-03-09-1)
- [2021-03-08](#2021-03-08)
  - [1. Hierarchical Transformer for Multilingual Machine Translation](#2021-03-08-1)
  - [2. WordBias: An Interactive Visual Tool for Discovering Intersectional Biases Encoded in Word Embeddings](#2021-03-08-2)
  - [3. Overcoming Poor Word Embeddings with Word Definitions](#2021-03-08-3)
- [2021-03-05](#2021-03-05)
  - [1. An empirical analysis of phrase-based and neural machine translation](#2021-03-05-1)
  - [2. An Empirical Study of End-to-end Simultaneous Speech Translation Decoding Strategies](#2021-03-05-2)
- [2021-03-04](#2021-03-04)
  - [1. Random Feature Attention](#2021-03-04-1)
  - [2. Meta-Curriculum Learning for Domain Adaptation in Neural Machine Translation](#2021-03-04-2)
  - [3. Lex2vec: making Explainable Word Embedding via Distant Supervision](#2021-03-04-3)
  - [4. NeurIPS 2020 NLC2CMD Competition: Translating Natural Language to Bash Commands](#2021-03-04-4)
- [2021-03-03](#2021-03-03)
  - [1. WIT: Wikipedia-based Image Text Dataset for Multimodal Multilingual Machine Learning](#2021-03-03-1)
  - [2. On the Effectiveness of Dataset Embeddings in Mono-lingual,Multi-lingual and Zero-shot Conditions](#2021-03-03-2)
  - [3. Contrastive Explanations for Model Interpretability](#2021-03-03-3)
  - [4. MultiSubs: A Large-scale Multimodal and Multilingual Dataset](#2021-03-03-4)
- [2021-03-02](#2021-03-02)
  - [1. Generative Adversarial Transformers](#2021-03-02-1)
  - [2. Token-Modification Adversarial Attacks for Natural Language Processing: A Survey](#2021-03-02-2)
  - [3. M6: A Chinese Multimodal Pretrainer](#2021-03-02-3)
- [2021-03-01](#2021-03-01)
  - [1. Automated essay scoring using efficient transformer-based language models](#2021-03-01-1)
  - [2. Learning Chess Blindfolded: Evaluating Language Models on State Tracking](#2021-03-01-2)
  - [3. Gradient-guided Loss Masking for Neural Machine Translation](#2021-03-01-3)
- [Other Columns](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-03-12

[Return to Index](#Index)



<h2 id="2021-03-12-1">1. FairFil: Contrastive Neural Debiasing Method for Pretrained Text Encoders</h2>

Title: [FairFil: Contrastive Neural Debiasing Method for Pretrained Text Encoders](https://arxiv.org/abs/2103.06413)

Authors: [Pengyu Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng%2C+P), [Weituo Hao](https://arxiv.org/search/cs?searchtype=author&query=Hao%2C+W), [Siyang Yuan](https://arxiv.org/search/cs?searchtype=author&query=Yuan%2C+S), [Shijing Si](https://arxiv.org/search/cs?searchtype=author&query=Si%2C+S), [Lawrence Carin](https://arxiv.org/search/cs?searchtype=author&query=Carin%2C+L)

> Pretrained text encoders, such as BERT, have been applied increasingly in various natural language processing (NLP) tasks, and have recently demonstrated significant performance gains. However, recent studies have demonstrated the existence of social bias in these pretrained NLP models. Although prior works have made progress on word-level debiasing, improved sentence-level fairness of pretrained encoders still lacks exploration. In this paper, we proposed the first neural debiasing method for a pretrained sentence encoder, which transforms the pretrained encoder outputs into debiased representations via a fair filter (FairFil) network. To learn the FairFil, we introduce a contrastive learning framework that not only minimizes the correlation between filtered embeddings and bias words but also preserves rich semantic information of the original sentences. On real-world datasets, our FairFil effectively reduces the bias degree of pretrained text encoders, while continuously showing desirable performance on downstream tasks. Moreover, our post-hoc method does not require any retraining of the text encoders, further enlarging FairFil's application space.

| Comments: | Accepted by the 9th International Conference on Learning Representations (ICLR 2021) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2103.06413](https://arxiv.org/abs/2103.06413) [cs.CL]** |
|           | (or **[arXiv:2103.06413v1](https://arxiv.org/abs/2103.06413v1) [cs.CL]** for this version) |





<h2 id="2021-03-12-2">2. LightMBERT: A Simple Yet Effective Method for Multilingual BERT Distillation</h2>

Title: [LightMBERT: A Simple Yet Effective Method for Multilingual BERT Distillation](https://arxiv.org/abs/2103.06418)

Authors: [Xiaoqi Jiao](https://arxiv.org/search/cs?searchtype=author&query=Jiao%2C+X), [Yichun Yin](https://arxiv.org/search/cs?searchtype=author&query=Yin%2C+Y), [Lifeng Shang](https://arxiv.org/search/cs?searchtype=author&query=Shang%2C+L), [Xin Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+X), [Xiao Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+X), [Linlin Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Fang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+F), [Qun Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Q)

> The multilingual pre-trained language models (e.g, mBERT, XLM and XLM-R) have shown impressive performance on cross-lingual natural language understanding tasks. However, these models are computationally intensive and difficult to be deployed on resource-restricted devices. In this paper, we propose a simple yet effective distillation method (LightMBERT) for transferring the cross-lingual generalization ability of the multilingual BERT to a small student model. The experiment results empirically demonstrate the efficiency and effectiveness of LightMBERT, which is significantly better than the baselines and performs comparable to the teacher mBERT.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2103.06418](https://arxiv.org/abs/2103.06418) [cs.CL]** |
|           | (or **[arXiv:2103.06418v1](https://arxiv.org/abs/2103.06418v1) [cs.CL]** for this version) |





<h2 id="2021-03-12-3">3. Towards Multi-Sense Cross-Lingual Alignment of Contextual Embeddings</h2>

Title: [Towards Multi-Sense Cross-Lingual Alignment of Contextual Embeddings](https://arxiv.org/abs/2103.06459)

Authors: [Linlin Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+L), [Thien Hai Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+T+H), [Shafiq Joty](https://arxiv.org/search/cs?searchtype=author&query=Joty%2C+S), [Lidong Bing](https://arxiv.org/search/cs?searchtype=author&query=Bing%2C+L), [Luo Si](https://arxiv.org/search/cs?searchtype=author&query=Si%2C+L)

> Cross-lingual word embeddings (CLWE) have been proven useful in many cross-lingual tasks. However, most existing approaches to learn CLWE including the ones with contextual embeddings are sense agnostic. In this work, we propose a novel framework to align contextual embeddings at the sense level by leveraging cross-lingual signal from bilingual dictionaries only. We operationalize our framework by first proposing a novel sense-aware cross entropy loss to model word senses explicitly. The monolingual ELMo and BERT models pretrained with our sense-aware cross entropy loss demonstrate significant performance improvement for word sense disambiguation tasks. We then propose a sense alignment objective on top of the sense-aware cross entropy loss for cross-lingual model pretraining, and pretrain cross-lingual models for several language pairs (English to German/Spanish/Japanese/Chinese). Compared with the best baseline results, our cross-lingual models achieve 0.52%, 2.09% and 1.29% average performance improvements on zero-shot cross-lingual NER, sentiment classification and XNLI tasks, respectively.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2103.06459](https://arxiv.org/abs/2103.06459) [cs.CL]** |
|           | (or **[arXiv:2103.06459v1](https://arxiv.org/abs/2103.06459v1) [cs.CL]** for this version) |





<h2 id="2021-03-12-4">4. Active2 Learning: Actively reducing redundancies in Active Learning methods for Sequence Tagging and Machine Translation</h2>

Title: [Active2 Learning: Actively reducing redundancies in Active Learning methods for Sequence Tagging and Machine Translation](https://arxiv.org/abs/2103.06490)

Authors: [Rishi Hazra](https://arxiv.org/search/cs?searchtype=author&query=Hazra%2C+R), [Parag Dutta](https://arxiv.org/search/cs?searchtype=author&query=Dutta%2C+P), [Shubham Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+S), [Mohammed Abdul Qaathir](https://arxiv.org/search/cs?searchtype=author&query=Qaathir%2C+M+A), [Ambedkar Dukkipati](https://arxiv.org/search/cs?searchtype=author&query=Dukkipati%2C+A)

> While deep learning is a powerful tool for natural language processing (NLP) problems, successful solutions to these problems rely heavily on large amounts of annotated samples. However, manually annotating data is expensive and time-consuming. Active Learning (AL) strategies reduce the need for huge volumes of labeled data by iteratively selecting a small number of examples for manual annotation based on their estimated utility in training the given model. In this paper, we argue that since AL strategies choose examples independently, they may potentially select similar examples, all of which may not contribute significantly to the learning process. Our proposed approach, Active**2** Learning (A**2**L), actively adapts to the deep learning model being trained to eliminate further such redundant examples chosen by an AL strategy. We show that A**2**L is widely applicable by using it in conjunction with several different AL strategies and NLP tasks. We empirically demonstrate that the proposed approach is further able to reduce the data requirements of state-of-the-art AL strategies by an absolute percentage reduction of ≈**3****−****25****%** on multiple NLP tasks while achieving the same performance with no additional computation overhead.

| Comments: | Accepted in NAACL-HLT-2021                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Human-Computer Interaction (cs.HC); Machine Learning (cs.LG); Neural and Evolutionary Computing (cs.NE) |
| Cite as:  | **[arXiv:2103.06490](https://arxiv.org/abs/2103.06490) [cs.CL]** |
|           | (or **[arXiv:2103.06490v1](https://arxiv.org/abs/2103.06490v1) [cs.CL]** for this version) |





<h2 id="2021-03-12-5">5. The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language Models</h2>

Title: [The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language Models](https://arxiv.org/abs/2103.06678)

Authors: [Go Inoue](https://arxiv.org/search/cs?searchtype=author&query=Inoue%2C+G), [Bashar Alhafni](https://arxiv.org/search/cs?searchtype=author&query=Alhafni%2C+B), [Nurpeiis Baimukan](https://arxiv.org/search/cs?searchtype=author&query=Baimukan%2C+N), [Houda Bouamor](https://arxiv.org/search/cs?searchtype=author&query=Bouamor%2C+H), [Nizar Habash](https://arxiv.org/search/cs?searchtype=author&query=Habash%2C+N)

> In this paper, we explore the effects of language variants, data sizes, and fine-tuning task types in Arabic pre-trained language models. To do so, we build three pre-trained language models across three variants of Arabic: Modern Standard Arabic (MSA), dialectal Arabic, and classical Arabic, in addition to a fourth language model which is pre-trained on a mix of the three. We also examine the importance of pre-training data size by building additional models that are pre-trained on a scaled-down set of the MSA variant. We compare our different models to each other, as well as to eight publicly available models by fine-tuning them on five NLP tasks spanning 12 datasets. Our results suggest that the variant proximity of pre-training data to fine-tuning data is more important than the pre-training data size. We exploit this insight in defining an optimized system selection model for the studied tasks.

| Comments: | Accepted to WANLP 2021                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2103.06678](https://arxiv.org/abs/2103.06678) [cs.CL]** |
|           | (or **[arXiv:2103.06678v1](https://arxiv.org/abs/2103.06678v1) [cs.CL]** for this version) |





<h2 id="2021-03-12-6">6. Unsupervised Transfer Learning in Multilingual Neural Machine Translation with Cross-Lingual Word Embeddings</h2>

Title: [Unsupervised Transfer Learning in Multilingual Neural Machine Translation with Cross-Lingual Word Embeddings](https://arxiv.org/abs/2103.06689)

Authors: [Carlos Mullov](https://arxiv.org/search/cs?searchtype=author&query=Mullov%2C+C), [Ngoc-Quan Pham](https://arxiv.org/search/cs?searchtype=author&query=Pham%2C+N), [Alexander Waibel](https://arxiv.org/search/cs?searchtype=author&query=Waibel%2C+A)

> In this work we look into adding a new language to a multilingual NMT system in an unsupervised fashion. Under the utilization of pre-trained cross-lingual word embeddings we seek to exploit a language independent multilingual sentence representation to easily generalize to a new language. While using cross-lingual embeddings for word lookup we decode from a yet entirely unseen source language in a process we call blind decoding. Blindly decoding from Portuguese using a basesystem containing several Romance languages we achieve scores of 36.4 BLEU for Portuguese-English and 12.8 BLEU for Russian-English. In an attempt to train the mapping from the encoder sentence representation to a new target language we use our model as an autoencoder. Merely training to translate from Portuguese to Portuguese while freezing the encoder we achieve 26 BLEU on English-Portuguese, and up to 28 BLEU when adding artificial noise to the input. Lastly we explore a more practical adaptation approach through non-iterative backtranslation, exploiting our model's ability to produce high quality translations through blind decoding. This yields us up to 34.6 BLEU on English-Portuguese, attaining near parity with a model adapted on real bilingual data.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2103.06689](https://arxiv.org/abs/2103.06689) [cs.CL]** |
|           | (or **[arXiv:2103.06689v1](https://arxiv.org/abs/2103.06689v1) [cs.CL]** for this version) |





<h2 id="2021-03-12-7">7. Towards Continual Learning for Multilingual Machine Translation via Vocabulary Substitution</h2>

Title: [Towards Continual Learning for Multilingual Machine Translation via Vocabulary Substitution](https://arxiv.org/abs/2103.06799)

Authors: [Xavier Garcia](https://arxiv.org/search/cs?searchtype=author&query=Garcia%2C+X), [Noah Constant](https://arxiv.org/search/cs?searchtype=author&query=Constant%2C+N), [Ankur P. Parikh](https://arxiv.org/search/cs?searchtype=author&query=Parikh%2C+A+P), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O)

> We propose a straightforward vocabulary adaptation scheme to extend the language capacity of multilingual machine translation models, paving the way towards efficient continual learning for multilingual machine translation. Our approach is suitable for large-scale datasets, applies to distant languages with unseen scripts, incurs only minor degradation on the translation performance for the original language pairs and provides competitive performance even in the case where we only possess monolingual data for the new languages.

| Comments: | Accepted at NAACL 2021                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2103.06799](https://arxiv.org/abs/2103.06799) [cs.CL]** |
|           | (or **[arXiv:2103.06799v1](https://arxiv.org/abs/2103.06799v1) [cs.CL]** for this version) |





<h2 id="2021-03-12-8">8. CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation</h2>

Title: [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://arxiv.org/abs/2103.06874)

Authors: [Jonathan H. Clark](https://arxiv.org/search/cs?searchtype=author&query=Clark%2C+J+H), [Dan Garrette](https://arxiv.org/search/cs?searchtype=author&query=Garrette%2C+D), [Iulia Turc](https://arxiv.org/search/cs?searchtype=author&query=Turc%2C+I), [John Wieting](https://arxiv.org/search/cs?searchtype=author&query=Wieting%2C+J)

> Pipelined NLP systems have largely been superseded by end-to-end neural modeling, yet nearly all commonly-used models still require an explicit tokenization step. While recent tokenization approaches based on data-derived subword lexicons are less brittle than manually engineered tokenizers, these techniques are not equally suited to all languages, and the use of any fixed vocabulary may limit a model's ability to adapt. In this paper, we present CANINE, a neural encoder that operates directly on character sequences--without explicit tokenization or vocabulary--and a pre-training strategy with soft inductive biases in place of hard token [this http URL](http://boundaries.to/) use its finer-grained input effectively and efficiently, CANINE combines downsampling, which reduces the input sequence length, with a deep transformer stack, which encodes con-text. CANINE outperforms a comparable mBERT model by >=1 F1 on TyDi QA, a challenging multilingual benchmark, despite having 28% fewer model parameters.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2103.06874](https://arxiv.org/abs/2103.06874) [cs.CL]** |
|           | (or **[arXiv:2103.06874v1](https://arxiv.org/abs/2103.06874v1) [cs.CL]** for this version) |



# 2021-03-11

[Return to Index](#Index)



<h2 id="2021-03-11-1">1. Self-Learning for Zero Shot Neural Machine Translation</h2>

Title: [Self-Learning for Zero Shot Neural Machine Translation](https://arxiv.org/abs/2103.05951)

Authors: [Surafel M. Lakew](https://arxiv.org/search/cs?searchtype=author&query=Lakew%2C+S+M), [Matteo Negri](https://arxiv.org/search/cs?searchtype=author&query=Negri%2C+M), [Marco Turchi](https://arxiv.org/search/cs?searchtype=author&query=Turchi%2C+M)

> Neural Machine Translation (NMT) approaches employing monolingual data are showing steady improvements in resource rich conditions. However, evaluations using real-world low-resource languages still result in unsatisfactory performance. This work proposes a novel zero-shot NMT modeling approach that learns without the now-standard assumption of a pivot language sharing parallel data with the zero-shot source and target languages. Our approach is based on three stages: initialization from any pre-trained NMT model observing at least the target language, augmentation of source sides leveraging target monolingual data, and learning to optimize the initial model to the zero-shot pair, where the latter two constitute a self-learning cycle. Empirical findings involving four diverse (in terms of a language family, script and relatedness) zero-shot pairs show the effectiveness of our approach with up to +5.93 BLEU improvement against a supervised bilingual baseline. Compared to unsupervised NMT, consistent improvements are observed even in a domain-mismatch setting, attesting to the usability of our method.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2103.05951](https://arxiv.org/abs/2103.05951) [cs.CL]** |
|           | (or **[arXiv:2103.05951v1](https://arxiv.org/abs/2103.05951v1) [cs.CL]** for this version) |





<h2 id="2021-03-11-2">2. CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review</h2>

Title: [CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review](https://arxiv.org/abs/2103.06268)

Authors: [Dan Hendrycks](https://arxiv.org/search/cs?searchtype=author&query=Hendrycks%2C+D), [Collin Burns](https://arxiv.org/search/cs?searchtype=author&query=Burns%2C+C), [Anya Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+A), [Spencer Ball](https://arxiv.org/search/cs?searchtype=author&query=Ball%2C+S)

> Many specialized domains remain untouched by deep learning, as large labeled datasets require expensive expert annotators. We address this bottleneck within the legal domain by introducing the Contract Understanding Atticus Dataset (CUAD), a new dataset for legal contract review. CUAD was created with dozens of legal experts from The Atticus Project and consists of over 13,000 annotations. The task is to highlight salient portions of a contract that are important for a human to review. We find that Transformer models have nascent performance, but that this performance is strongly influenced by model design and training dataset size. Despite these promising results, there is still substantial room for improvement. As one of the only large, specialized NLP benchmarks annotated by experts, CUAD can serve as a challenging research benchmark for the broader NLP community.

| Comments: | Code and the CUAD dataset are available at [this https URL](https://github.com/TheAtticusProject/cuad/) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2103.06268](https://arxiv.org/abs/2103.06268) [cs.CL]** |
|           | (or **[arXiv:2103.06268v1](https://arxiv.org/abs/2103.06268v1) [cs.CL]** for this version) |



# 2021-03-10

[Return to Index](#Index)



<h2 id="2021-03-10-1">1. AfriVEC: Word Embedding Models for African Languages. Case Study of Fon and Nobiin</h2>

Title: [AfriVEC: Word Embedding Models for African Languages. Case Study of Fon and Nobiin](https://arxiv.org/abs/2103.05132)

Authors: [Bonaventure F. P. Dossou](https://arxiv.org/search/cs?searchtype=author&query=Dossou%2C+B+F+P), [Mohammed Sabry](https://arxiv.org/search/cs?searchtype=author&query=Sabry%2C+M)

> From Word2Vec to GloVe, word embedding models have played key roles in the current state-of-the-art results achieved in Natural Language Processing. Designed to give significant and unique vectorized representations of words and entities, those models have proven to efficiently extract similarities and establish relationships reflecting semantic and contextual meaning among words and entities. African Languages, representing more than 31% of the worldwide spoken languages, have recently been subject to lots of research. However, to the best of our knowledge, there are currently very few to none word embedding models for those languages words and entities, and none for the languages under study in this paper. After describing Glove, Word2Vec, and Poincaré embeddings functionalities, we build Word2Vec and Poincaré word embedding models for Fon and Nobiin, which show promising results. We test the applicability of transfer learning between these models as a landmark for African Languages to jointly involve in mitigating the scarcity of their resources, and attempt to provide linguistic and social interpretations of our results. Our main contribution is to arouse more interest in creating word embedding models proper to African Languages, ready for use, and that can significantly improve the performances of Natural Language Processing downstream tasks on them. The official repository and implementation is at [this https URL](https://github.com/bonaventuredossou/afrivec)

| Subjects:          | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | Africa NLP, EACL 2021                                        |
| Cite as:           | **[arXiv:2103.05132](https://arxiv.org/abs/2103.05132) [cs.CL]** |
|                    | (or **[arXiv:2103.05132v1](https://arxiv.org/abs/2103.05132v1) [cs.CL]** for this version) |





# 2021-03-09

[Return to Index](#Index)



<h2 id="2021-03-09-1">1. Hierarchical Transformer for Multilingual Machine Translation</h2>

Title: [Translating the Unseen? Yorùbá → English MT in Low-Resource, Morphologically-Unmarked Settings](https://arxiv.org/abs/2103.04225)

Authors: [Ife Adebara Miikka Silfverberg Muhammad Abdul-Mageed](https://arxiv.org/search/cs?searchtype=author&query=Abdul-Mageed%2C+I+A+M+S+M)

> Translating between languages where certain features are marked morphologically in one but absent or marked contextually in the other is an important test case for machine translation. When translating into English which marks (in)definiteness morphologically, from Yorùbá which uses bare nouns but marks these features contextually, ambiguities arise. In this work, we perform fine-grained analysis on how an SMT system compares with two NMT systems (BiLSTM and Transformer) when translating bare nouns in Yorùbá into English. We investigate how the systems what extent they identify BNs, correctly translate them, and compare with human translation patterns. We also analyze the type of errors each model makes and provide a linguistic description of these errors. We glean insights for evaluating model performance in low-resource settings. In translating bare nouns, our results show the transformer model outperforms the SMT and BiLSTM models for 4 categories, the BiLSTM outperforms the SMT model for 3 categories while the SMT outperforms the NMT models for 1 category.

| Subjects:          | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | AfricanNLP @ EACL2021                                        |
| Cite as:           | **[arXiv:2103.04225](https://arxiv.org/abs/2103.04225) [cs.CL]** |
|                    | (or **[arXiv:2103.04225v1](https://arxiv.org/abs/2103.04225v1) [cs.CL]** for this version) |







# 2021-03-08

[Return to Index](#Index)



<h2 id="2021-03-08-1">1. Hierarchical Transformer for Multilingual Machine Translation</h2>

Title: [Hierarchical Transformer for Multilingual Machine Translation](https://arxiv.org/abs/2103.03589)

Authors: [Albina Khusainova](https://arxiv.org/search/cs?searchtype=author&query=Khusainova%2C+A), [Adil Khan](https://arxiv.org/search/cs?searchtype=author&query=Khan%2C+A), [Adín Ramírez Rivera](https://arxiv.org/search/cs?searchtype=author&query=Rivera%2C+A+R), [Vitaly Romanov](https://arxiv.org/search/cs?searchtype=author&query=Romanov%2C+V)

> The choice of parameter sharing strategy in multilingual machine translation models determines how optimally parameter space is used and hence, directly influences ultimate translation quality. Inspired by linguistic trees that show the degree of relatedness between different languages, the new general approach to parameter sharing in multilingual machine translation was suggested recently. The main idea is to use these expert language hierarchies as a basis for multilingual architecture: the closer two languages are, the more parameters they share. In this work, we test this idea using the Transformer architecture and show that despite the success in previous work there are problems inherent to training such hierarchical models. We demonstrate that in case of carefully chosen training strategy the hierarchical architecture can outperform bilingual models and multilingual models with full parameter sharing.

| Comments: | Accepted to VarDial 2021                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2103.03589](https://arxiv.org/abs/2103.03589) [cs.CL]** |
|           | (or **[arXiv:2103.03589v1](https://arxiv.org/abs/2103.03589v1) [cs.CL]** for this version) |





<h2 id="2021-03-08-2">2. WordBias: An Interactive Visual Tool for Discovering Intersectional Biases Encoded in Word Embeddings</h2>

Title: [WordBias: An Interactive Visual Tool for Discovering Intersectional Biases Encoded in Word Embeddings](https://arxiv.org/abs/2103.03598)

Authors: [Bhavya Ghai](https://arxiv.org/search/cs?searchtype=author&query=Ghai%2C+B), [Md Naimul Hoque](https://arxiv.org/search/cs?searchtype=author&query=Hoque%2C+M+N), [Klaus Mueller](https://arxiv.org/search/cs?searchtype=author&query=Mueller%2C+K)

> Intersectional bias is a bias caused by an overlap of multiple social factors like gender, sexuality, race, disability, religion, etc. A recent study has shown that word embedding models can be laden with biases against intersectional groups like African American females, etc. The first step towards tackling such intersectional biases is to identify them. However, discovering biases against different intersectional groups remains a challenging task. In this work, we present WordBias, an interactive visual tool designed to explore biases against intersectional groups encoded in static word embeddings. Given a pretrained static word embedding, WordBias computes the association of each word along different groups based on race, age, etc. and then visualizes them using a novel interactive interface. Using a case study, we demonstrate how WordBias can help uncover biases against intersectional groups like Black Muslim Males, Poor Females, etc. encoded in word embedding. In addition, we also evaluate our tool using qualitative feedback from expert interviews. The source code for this tool can be publicly accessed for reproducibility at [this http URL](http://github.com/bhavyaghai/WordBias).

| Comments: | Accepted to ACM SIGCHI 2021 LBW                              |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Human-Computer Interaction (cs.HC) |
| Cite as:  | **[arXiv:2103.03598](https://arxiv.org/abs/2103.03598) [cs.CL]** |
|           | (or **[arXiv:2103.03598v1](https://arxiv.org/abs/2103.03598v1) [cs.CL]** for this version) |





<h2 id="2021-03-08-3">3. Overcoming Poor Word Embeddings with Word Definitions</h2>

Title: [Overcoming Poor Word Embeddings with Word Definitions](https://arxiv.org/abs/2103.03842)

Authors: [Christopher Malon](https://arxiv.org/search/cs?searchtype=author&query=Malon%2C+C)

> Modern natural language understanding models depend on pretrained subword embeddings, but applications may need to reason about words that were never or rarely seen during pretraining. We show that examples that depend critically on a rarer word are more challenging for natural language inference models. Then we explore how a model could learn to use definitions, provided in natural text, to overcome this handicap. Our model's understanding of a definition is usually weaker than a well-modeled word embedding, but it recovers most of the performance gap from using a completely untrained word.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2103.03842](https://arxiv.org/abs/2103.03842) [cs.CL]** |
|           | (or **[arXiv:2103.03842v1](https://arxiv.org/abs/2103.03842v1) [cs.CL]** for this version) |



# 2021-03-05

[Return to Index](#Index)



<h2 id="2021-03-05-1">1. An empirical analysis of phrase-based and neural machine translation</h2>

Title: [An empirical analysis of phrase-based and neural machine translation](https://arxiv.org/abs/2103.03108)

Authors: [Hamidreza Ghader](https://arxiv.org/search/cs?searchtype=author&query=Ghader%2C+H)

> Two popular types of machine translation (MT) are phrase-based and neural machine translation systems. Both of these types of systems are composed of multiple complex models or layers. Each of these models and layers learns different linguistic aspects of the source language. However, for some of these models and layers, it is not clear which linguistic phenomena are learned or how this information is learned. For phrase-based MT systems, it is often clear what information is learned by each model, and the question is rather how this information is learned, especially for its phrase reordering model. For neural machine translation systems, the situation is even more complex, since for many cases it is not exactly clear what information is learned and how it is learned.
> To shed light on what linguistic phenomena are captured by MT systems, we analyze the behavior of important models in both phrase-based and neural MT systems. We consider phrase reordering models from phrase-based MT systems to investigate which words from inside of a phrase have the biggest impact on defining the phrase reordering behavior. Additionally, to contribute to the interpretability of neural MT systems we study the behavior of the attention model, which is a key component in neural MT systems and the closest model in functionality to phrase reordering models in phrase-based systems. The attention model together with the encoder hidden state representations form the main components to encode source side linguistic information in neural MT. To this end, we also analyze the information captured in the encoder hidden state representations of a neural MT system. We investigate the extent to which syntactic and lexical-semantic information from the source side is captured by hidden state representations of different neural MT architectures.

| Comments: | PhD thesis, University of Amsterdam, October 2020. [this https URL](https://pure.uva.nl/ws/files/51388868/Thesis.pdf) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Neural and Evolutionary Computing (cs.NE) |
| Cite as:  | **[arXiv:2103.03108](https://arxiv.org/abs/2103.03108) [cs.CL]** |
|           | (or **[arXiv:2103.03108v1](https://arxiv.org/abs/2103.03108v1) [cs.CL]** for this version) |





<h2 id="2021-03-05-2">2. An Empirical Study of End-to-end Simultaneous Speech Translation Decoding Strategies</h2>

Title: [An Empirical Study of End-to-end Simultaneous Speech Translation Decoding Strategies](https://arxiv.org/abs/2103.03233)

Authors: [Ha Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+H), [Yannick Estève](https://arxiv.org/search/cs?searchtype=author&query=Estève%2C+Y), [Laurent Besacier](https://arxiv.org/search/cs?searchtype=author&query=Besacier%2C+L)

> This paper proposes a decoding strategy for end-to-end simultaneous speech translation. We leverage end-to-end models trained in offline mode and conduct an empirical study for two language pairs (English-to-German and English-to-Portuguese). We also investigate different output token granularities including characters and Byte Pair Encoding (BPE) units. The results show that the proposed decoding approach allows to control BLEU/Average Lagging trade-off along different latency regimes. Our best decoding settings achieve comparable results with a strong cascade model evaluated on the simultaneous translation track of IWSLT 2020 shared task.

| Comments: | This paper has been accepted for presentation at IEEE ICASSP 2021 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2103.03233](https://arxiv.org/abs/2103.03233) [cs.CL]** |
|           | (or **[arXiv:2103.03233v1](https://arxiv.org/abs/2103.03233v1) [cs.CL]** for this version) |













# 2021-03-04

[Return to Index](#Index)



<h2 id="2021-03-04-1">1. Random Feature Attention</h2>

Title: [Random Feature Attention](https://arxiv.org/abs/2103.02143)

Authors: [Hao Peng](https://arxiv.org/search/cs?searchtype=author&query=Peng%2C+H), [Nikolaos Pappas](https://arxiv.org/search/cs?searchtype=author&query=Pappas%2C+N), [Dani Yogatama](https://arxiv.org/search/cs?searchtype=author&query=Yogatama%2C+D), [Roy Schwartz](https://arxiv.org/search/cs?searchtype=author&query=Schwartz%2C+R), [Noah A. Smith](https://arxiv.org/search/cs?searchtype=author&query=Smith%2C+N+A), [Lingpeng Kong](https://arxiv.org/search/cs?searchtype=author&query=Kong%2C+L)

> Transformers are state-of-the-art models for a variety of sequence modeling tasks. At their core is an attention function which models pairwise interactions between the inputs at every timestep. While attention is powerful, it does not scale efficiently to long sequences due to its quadratic time and space complexity in the sequence length. We propose RFA, a linear time and space attention that uses random feature methods to approximate the softmax function, and explore its application in transformers. RFA can be used as a drop-in replacement for conventional softmax attention and offers a straightforward way of learning with recency bias through an optional gating mechanism. Experiments on language modeling and machine translation demonstrate that RFA achieves similar or better performance compared to strong transformer baselines. In the machine translation experiment, RFA decodes twice as fast as a vanilla transformer. Compared to existing efficient transformer variants, RFA is competitive in terms of both accuracy and efficiency on three long text classification datasets. Our analysis shows that RFA's efficiency gains are especially notable on long sequences, suggesting that RFA will be particularly useful in tasks that require working with large inputs, fast decoding speed, or low memory footprints.

| Comments: | ICLR 2021                                                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2103.02143](https://arxiv.org/abs/2103.02143) [cs.CL]** |
|           | (or **[arXiv:2103.02143v1](https://arxiv.org/abs/2103.02143v1) [cs.CL]** for this version) |





<h2 id="2021-03-04-2">2. Meta-Curriculum Learning for Domain Adaptation in Neural Machine Translation</h2>

Title: [Meta-Curriculum Learning for Domain Adaptation in Neural Machine Translation](https://arxiv.org/abs/2103.02262)

Authors: [Runzhe Zhan](https://arxiv.org/search/cs?searchtype=author&query=Zhan%2C+R), [Xuebo Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+X), [Derek F. Wong](https://arxiv.org/search/cs?searchtype=author&query=Wong%2C+D+F), [Lidia S. Chao](https://arxiv.org/search/cs?searchtype=author&query=Chao%2C+L+S)

> Meta-learning has been sufficiently validated to be beneficial for low-resource neural machine translation (NMT). However, we find that meta-trained NMT fails to improve the translation performance of the domain unseen at the meta-training stage. In this paper, we aim to alleviate this issue by proposing a novel meta-curriculum learning for domain adaptation in NMT. During meta-training, the NMT first learns the similar curricula from each domain to avoid falling into a bad local optimum early, and finally learns the curricula of individualities to improve the model robustness for learning domain-specific knowledge. Experimental results on 10 different low-resource domains show that meta-curriculum learning can improve the translation performance of both familiar and unfamiliar domains. All the codes and data are freely available at [this https URL](https://github.com/NLP2CT/Meta-Curriculum).

| Comments: | Accepted to AAAI 2021                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2103.02262](https://arxiv.org/abs/2103.02262) [cs.CL]** |
|           | (or **[arXiv:2103.02262v1](https://arxiv.org/abs/2103.02262v1) [cs.CL]** for this version) |





<h2 id="2021-03-04-3">3. Lex2vec: making Explainable Word Embedding via Distant Supervision</h2>

Title: [Lex2vec: making Explainable Word Embedding via Distant Supervision](https://arxiv.org/abs/2103.02269)

Authors: [Fabio Celli](https://arxiv.org/search/cs?searchtype=author&query=Celli%2C+F)

> In this technical report we propose an algorithm, called Lex2vec, that exploits lexical resources to inject information into word embeddings and name the embedding dimensions by means of distant supervision. We evaluate the optimal parameters to extract a number of informative labels that is readable and has a good coverage for the embedding dimensions.

| Comments: | 3 pages, 1 figure, 1 table                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2103.02269](https://arxiv.org/abs/2103.02269) [cs.CL]** |
|           | (or **[arXiv:2103.02269v1](https://arxiv.org/abs/2103.02269v1) [cs.CL]** for this version) |





<h2 id="2021-03-04-4">4. NeurIPS 2020 NLC2CMD Competition: Translating Natural Language to Bash Commands</h2>

Title: [NeurIPS 2020 NLC2CMD Competition: Translating Natural Language to Bash Commands](https://arxiv.org/abs/2103.02523)

Authors: [Mayank Agarwal](https://arxiv.org/search/cs?searchtype=author&query=Agarwal%2C+M), [Tathagata Chakraborti](https://arxiv.org/search/cs?searchtype=author&query=Chakraborti%2C+T), [Quchen Fu](https://arxiv.org/search/cs?searchtype=author&query=Fu%2C+Q), [David Gros](https://arxiv.org/search/cs?searchtype=author&query=Gros%2C+D), [Xi Victoria Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+X+V), [Jaron Maene](https://arxiv.org/search/cs?searchtype=author&query=Maene%2C+J), [Kartik Talamadupula](https://arxiv.org/search/cs?searchtype=author&query=Talamadupula%2C+K), [Zhongwei Teng](https://arxiv.org/search/cs?searchtype=author&query=Teng%2C+Z), [Jules White](https://arxiv.org/search/cs?searchtype=author&query=White%2C+J)

> The NLC2CMD Competition hosted at NeurIPS 2020 aimed to bring the power of natural language processing to the command line. Participants were tasked with building models that can transform descriptions of command line tasks in English to their Bash syntax. This is a report on the competition with details of the task, metrics, data, attempted solutions, and lessons learned.

| Comments: | Competition URL: [this http URL](http://ibm.biz/nlc2cmd)     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2103.02523](https://arxiv.org/abs/2103.02523) [cs.CL]** |
|           | (or **[arXiv:2103.02523v1](https://arxiv.org/abs/2103.02523v1) [cs.CL]** for this version) |



# 2021-03-03

[Return to Index](#Index)



<h2 id="2021-03-03-1">1. WIT: Wikipedia-based Image Text Dataset for Multimodal Multilingual Machine Learning</h2>

Title: [WIT: Wikipedia-based Image Text Dataset for Multimodal Multilingual Machine Learning](https://arxiv.org/abs/2103.01913)

Authors: [Krishna Srinivasan](https://arxiv.org/search/cs?searchtype=author&query=Srinivasan%2C+K), [Karthik Raman](https://arxiv.org/search/cs?searchtype=author&query=Raman%2C+K), [Jiecao Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+J), [Michael Bendersky](https://arxiv.org/search/cs?searchtype=author&query=Bendersky%2C+M), [Marc Najork](https://arxiv.org/search/cs?searchtype=author&query=Najork%2C+M)

> The milestone improvements brought about by deep representation learning and pre-training techniques have led to large performance gains across downstream NLP, IR and Vision tasks. Multimodal modeling techniques aim to leverage large high-quality visio-linguistic datasets for learning complementary information (across image and text modalities). In this paper, we introduce the Wikipedia-based Image Text (WIT) Dataset\footnote{\url{[this https URL](https://github.com/google-research-datasets/wit)}} to better facilitate multimodal, multilingual learning. WIT is composed of a curated set of 37.6 million entity rich image-text examples with 11.5 million unique images across 108 Wikipedia languages. Its size enables WIT to be used as a pretraining dataset for multimodal models, as we show when applied to downstream tasks such as image-text retrieval. WIT has four main and unique advantages. First, WIT is the largest multimodal dataset by the number of image-text examples by 3x (at the time of writing). Second, WIT is massively multilingual (first of its kind) with coverage over 100+ languages (each of which has at least 12K examples) and provides cross-lingual texts for many images. Third, WIT represents a more diverse set of concepts and real world entities relative to what previous datasets cover. Lastly, WIT provides a very challenging real-world test set, as we empirically illustrate using an image-text retrieval task as an example.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Information Retrieval (cs.IR) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2103.01913](https://arxiv.org/abs/2103.01913) [cs.CV]** |
|           | (or **[arXiv:2103.01913v1](https://arxiv.org/abs/2103.01913v1) [cs.CV]** for this version) |





<h2 id="2021-03-03-2">2. On the Effectiveness of Dataset Embeddings in Mono-lingual,Multi-lingual and Zero-shot Conditions</h2>

Title: [On the Effectiveness of Dataset Embeddings in Mono-lingual,Multi-lingual and Zero-shot Conditions](https://arxiv.org/abs/2103.01273)

Authors: [Rob van der Goot](https://arxiv.org/search/cs?searchtype=author&query=van+der+Goot%2C+R), [Ahmet Üstün](https://arxiv.org/search/cs?searchtype=author&query=Üstün%2C+A), [Barbara Plank](https://arxiv.org/search/cs?searchtype=author&query=Plank%2C+B)

> Recent complementary strands of research have shown that leveraging information on the data source through encoding their properties into embeddings can lead to performance increase when training a single model on heterogeneous data sources. However, it remains unclear in which situations these dataset embeddings are most effective, because they are used in a large variety of settings, languages and tasks. Furthermore, it is usually assumed that gold information on the data source is available, and that the test data is from a distribution seen during training. In this work, we compare the effect of dataset embeddings in mono-lingual settings, multi-lingual settings, and with predicted data source label in a zero-shot setting. We evaluate on three morphosyntactic tasks: morphological tagging, lemmatization, and dependency parsing, and use 104 datasets, 66 languages, and two different dataset grouping strategies. Performance increases are highest when the datasets are of the same language, and we know from which distribution the test-instance is drawn. In contrast, for setups where the data is from an unseen distribution, performance increase vanishes.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2103.01273](https://arxiv.org/abs/2103.01273) [cs.CL]** |
|           | (or **[arXiv:2103.01273v1](https://arxiv.org/abs/2103.01273v1) [cs.CL]** for this version) |





<h2 id="2021-03-03-3">3. Contrastive Explanations for Model Interpretability</h2>

Title: [Contrastive Explanations for Model Interpretability](https://arxiv.org/abs/2103.01378)

Authors: [Alon Jacovi](https://arxiv.org/search/cs?searchtype=author&query=Jacovi%2C+A), [Swabha Swayamdipta](https://arxiv.org/search/cs?searchtype=author&query=Swayamdipta%2C+S), [Shauli Ravfogel](https://arxiv.org/search/cs?searchtype=author&query=Ravfogel%2C+S), [Yanai Elazar](https://arxiv.org/search/cs?searchtype=author&query=Elazar%2C+Y), [Yejin Choi](https://arxiv.org/search/cs?searchtype=author&query=Choi%2C+Y), [Yoav Goldberg](https://arxiv.org/search/cs?searchtype=author&query=Goldberg%2C+Y)

> Contrastive explanations clarify why an event occurred in contrast to another. They are more inherently intuitive to humans to both produce and comprehend. We propose a methodology to produce contrastive explanations for classification models by modifying the representation to disregard non-contrastive information, and modifying model behavior to only be based on contrastive reasoning. Our method is based on projecting model representation to a latent space that captures only the features that are useful (to the model) to differentiate two potential decisions. We demonstrate the value of contrastive explanations by analyzing two different scenarios, using both high-level abstract concept attribution and low-level input token/span attribution, on two widely used text classification tasks. Specifically, we produce explanations for answering: for which label, and against which alternative label, is some aspect of the input useful? And which aspects of the input are useful for and against particular decisions? Overall, our findings shed light on the ability of label-contrastive explanations to provide a more accurate and finer-grained interpretability of a model's decision.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2103.01378](https://arxiv.org/abs/2103.01378) [cs.CL]** |
|           | (or **[arXiv:2103.01378v1](https://arxiv.org/abs/2103.01378v1) [cs.CL]** for this version) |





<h2 id="2021-03-03-4">4. MultiSubs: A Large-scale Multimodal and Multilingual Dataset</h2>

Title: [MultiSubs: A Large-scale Multimodal and Multilingual Dataset](https://arxiv.org/abs/2103.01910)

Authors: [Josiah Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J), [Pranava Madhyastha](https://arxiv.org/search/cs?searchtype=author&query=Madhyastha%2C+P), [Josiel Figueiredo](https://arxiv.org/search/cs?searchtype=author&query=Figueiredo%2C+J), [Chiraag Lala](https://arxiv.org/search/cs?searchtype=author&query=Lala%2C+C), [Lucia Specia](https://arxiv.org/search/cs?searchtype=author&query=Specia%2C+L)

> This paper introduces a large-scale multimodal and multilingual dataset that aims to facilitate research on grounding words to images in their contextual usage in language. The dataset consists of images selected to unambiguously illustrate concepts expressed in sentences from movie subtitles. The dataset is a valuable resource as (i) the images are aligned to text fragments rather than whole sentences; (ii) multiple images are possible for a text fragment and a sentence; (iii) the sentences are free-form and real-world like; (iv) the parallel texts are multilingual. We set up a fill-in-the-blank game for humans to evaluate the quality of the automatic image selection process of our dataset. We show the utility of the dataset on two automatic tasks: (i) fill-in-the blank; (ii) lexical translation. Results of the human evaluation and automatic models demonstrate that images can be a useful complement to the textual context. The dataset will benefit research on visual grounding of words especially in the context of free-form sentences.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2103.01910](https://arxiv.org/abs/2103.01910) [cs.CL]** |
|           | (or **[arXiv:2103.01910v1](https://arxiv.org/abs/2103.01910v1) [cs.CL]** for this version) |







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