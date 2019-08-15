# Daily arXiv: Machine Translation - Aug., 2019

### Index

- [2019-08-15](#2019-08-15)
  - [1. On The Evaluation of Machine Translation Systems Trained With Back-Translation](#2019-08-15-1)

- [2019-08-14](#2019-08-14)
  - [1. Neural Text Generation with Unlikelihood Training](#2019-08-14-1)
  - [2. LSTM vs. GRU vs. Bidirectional RNN for script generation](#2019-08-14-2)
  - [3. Attention is not not Explanation](#2019-08-14-3)
  - [4. Neural Machine Translation with Noisy Lexical Constraints](#2019-08-14-4)

- [2019-08-13](#2019-08-13)
  - [1. On the Validity of Self-Attention as Explanation in Transformer Models](#2019-08-13-1)
- [2019-08-12](#2019-08-12)
  - [1. Exploiting Cross-Lingual Speaker and Phonetic Diversity for Unsupervised Subword Modeling](#2019-08-12-1)
  - [2. UdS Submission for the WMT 19 Automatic Post-Editing Task](#2019-08-12-2)
- [2019-08-09](#2019-08-09)
  - [1. A Test Suite and Manual Evaluation of Document-Level NMT at WMT19](#2019-08-09-1)
- [2019-08-07](#2019-08-07)
  - [1. MacNet: Transferring Knowledge from Machine Comprehension to Sequence-to-Sequence Models](#2019-08-07-1)
  - [2. A Translate-Edit Model for Natural Language Question to SQL Query Generation on Multi-relational Healthcare Data](#2019-08-07-2)
  - [3. Self-Knowledge Distillation in Natural Language Processing](#2019-08-07-3)
- [2019-08-06](#2019-08-06)
  - [1. Invariance-based Adversarial Attack on Neural Machine Translation Systems](#2019-08-06-1)
  - [2. Performance Evaluation of Supervised Machine Learning Techniques for Efficient Detection of Emotions from Online Content](#2019-08-06-2)
  - [3. The TALP-UPC System for the WMT Similar Language Task: Statistical vs Neural Machine Translation](#2019-08-06-3)
  - [4. JUMT at WMT2019 News Translation Task: A Hybrid approach to Machine Translation for Lithuanian to English](#2019-08-06-4)
  - [5. Beyond English-only Reading Comprehension: Experiments in Zero-Shot Multilingual Transfer for Bulgarian](#2019-08-06-5)
  - [6. Predicting Actions to Help Predict Translations](#2019-08-06-6)
  - [7. Thoth: Improved Rapid Serial Visual Presentation using Natural Language Processing](#2019-08-06-7)
- [2019-08-02](#2019-08-02)
  - [1. Tree-Transformer: A Transformer-Based Method for Correction of Tree-Structured Data](#2019-08-02-1)
  - [2. Learning Joint Acoustic-Phonetic Word Embeddings](#2019-08-02-2)
  - [3. JUCBNMT at WMT2018 News Translation Task: Character Based Neural Machine Translation of Finnish to English](#2019-08-02-3)

* [2019-07](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-07.md)
* [2019-06](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-06.md)
* [2019-05](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-05.md)
* [2019-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-04.md)
* [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)



# 2019-08-15

[Return to Index](#Index)



<h2 id="2019-08-15-1">1. On The Evaluation of Machine Translation Systems Trained With Back-Translation</h2> 

Title: [On The Evaluation of Machine Translation Systems Trained With Back-Translation](https://arxiv.org/abs/1908.05204)

Authors: [Sergey Edunov](https://arxiv.org/search/cs?searchtype=author&query=Edunov%2C+S), [Myle Ott](https://arxiv.org/search/cs?searchtype=author&query=Ott%2C+M), [Marc'Aurelio Ranzato](https://arxiv.org/search/cs?searchtype=author&query=Ranzato%2C+M), [Michael Auli](https://arxiv.org/search/cs?searchtype=author&query=Auli%2C+M)

*(Submitted on 14 Aug 2019)*

> Back-translation is a widely used data augmentation technique which leverages target monolingual data. However, its effectiveness has been challenged since automatic metrics such as BLEU only show significant improvements for test examples where the source itself is a translation, or translationese. This is believed to be due to translationese inputs better matching the back-translated training data. In this work, we show that this conjecture is not empirically supported and that back-translation improves translation quality of both naturally occurring text as well as translationese according to professional human translators. We provide empirical evidence to support the view that back-translation is preferred by humans because it produces more fluent outputs. BLEU cannot capture human preferences because references are translationese when source sentences are natural text. We recommend complementing BLEU with a language model score to measure fluency.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1908.05204 [cs.CL]**                         |
|           | (or **arXiv:1908.05204v1 [cs.CL]** for this version) |



# 2019-08-14

[Return to Index](#Index)



<h2 id="2019-08-14-1">1. Neural Text Generation with Unlikelihood Training</h2> 
Title: [Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319)

Authors: [Sean Welleck](https://arxiv.org/search/cs?searchtype=author&query=Welleck%2C+S), [Ilia Kulikov](https://arxiv.org/search/cs?searchtype=author&query=Kulikov%2C+I), [Stephen Roller](https://arxiv.org/search/cs?searchtype=author&query=Roller%2C+S), [Emily Dinan](https://arxiv.org/search/cs?searchtype=author&query=Dinan%2C+E), [Kyunghyun Cho](https://arxiv.org/search/cs?searchtype=author&query=Cho%2C+K), [Jason Weston](https://arxiv.org/search/cs?searchtype=author&query=Weston%2C+J)

*(Submitted on 12 Aug 2019)*

> Neural text generation is a key tool in natural language applications, but it is well known there are major problems at its core. In particular, standard likelihood training and decoding leads to dull and repetitive responses. While some post-hoc fixes have been proposed, in particular top-k and nucleus sampling, they do not address the fact that the token-level probabilities predicted by the model itself are poor. In this paper we show that the likelihood objective itself is at fault, resulting in a model that assigns too much probability to sequences that contain repeats and frequent words unlike the human training distribution. We propose a new objective, unlikelihood training, which forces unlikely generations to be assigned lower probability by the model. We show that both token and sequence level unlikelihood training give less repetitive, less dull text while maintaining perplexity, giving far superior generations using standard greedy or beam search. Our approach provides a strong alternative to traditional training.

| Comments: | Sean Welleck and Ilia Kulikov contributed equally            |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Machine Learning (stat.ML) |
| Cite as:  | **arXiv:1908.04319 [cs.LG]**                                 |
|           | (or **arXiv:1908.04319v1 [cs.LG]** for this version)         |





<h2 id="2019-08-14-2">2. LSTM vs. GRU vs. Bidirectional RNN for script generation</h2> 
Title: [LSTM vs. GRU vs. Bidirectional RNN for script generation](https://arxiv.org/abs/1908.04332)

Authors: [Sanidhya Mangal](https://arxiv.org/search/cs?searchtype=author&query=Mangal%2C+S), [Poorva Joshi](https://arxiv.org/search/cs?searchtype=author&query=Joshi%2C+P), [Rahul Modak](https://arxiv.org/search/cs?searchtype=author&query=Modak%2C+R)

*(Submitted on 12 Aug 2019)*

> Scripts are an important part of any TV series. They narrate movements, actions and expressions of characters. In this paper, a case study is presented on how different sequence to sequence deep learning models perform in the task of generating new conversations between characters as well as new scenarios on the basis of a script (previous conversations). A comprehensive comparison between these models, namely, LSTM, GRU and Bidirectional RNN is presented. All the models are designed to learn the sequence of recurring characters from the input sequence. Each input sequence will contain, say "n" characters, and the corresponding targets will contain the same number of characters, except, they will be shifted one character to the right. In this manner, input and output sequences are generated and used to train the models. A closer analysis of explored models performance and efficiency is delineated with the help of graph plots and generated texts by taking some input string. These graphs describe both, intraneural performance and interneural model performance for each model.

| Comments: | 7 pages, 7 figures                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1908.04332 [cs.CL]**                                 |
|           | (or **arXiv:1908.04332v1 [cs.CL]** for this version)         |





<h2 id="2019-08-14-3">3. Attention is not not Explanation</h2> 
Title: [Attention is not not Explanation](https://arxiv.org/abs/1908.04626)

Authors: [Sarah Wiegreffe](https://arxiv.org/search/cs?searchtype=author&query=Wiegreffe%2C+S), [Yuval Pinter](https://arxiv.org/search/cs?searchtype=author&query=Pinter%2C+Y)

*(Submitted on 13 Aug 2019)*

> Attention mechanisms play a central role in NLP systems, especially within recurrent neural network (RNN) models. Recently, there has been increasing interest in whether or not the intermediate representations offered by these modules may be used to explain the reasoning for a model's prediction, and consequently reach insights regarding the model's decision-making process. A recent paper claims that `Attention is not Explanation' (Jain and Wallace, 2019). We challenge many of the assumptions underlying this work, arguing that such a claim depends on one's definition of explanation, and that testing it needs to take into account all elements of the model, using a rigorous experimental design. We propose four alternative tests to determine when/whether attention can be used as explanation: a simple uniform-weights baseline; a variance calibration based on multiple random seed runs; a diagnostic framework using frozen weights from pretrained models; and an end-to-end adversarial attention training protocol. Each allows for meaningful interpretation of attention mechanisms in RNN models. We show that even when reliable adversarial distributions can be found, they don't perform well on the simple diagnostic, indicating that prior work does not disprove the usefulness of attention mechanisms for explainability.

| Comments: | Accepted to EMNLP 2019; related blog post at [this https URL](https://medium.com/@yuvalpinter/attention-is-not-not-explanation-dbc25b534017) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1908.04626 [cs.CL]**                                 |
|           | (or **arXiv:1908.04626v1 [cs.CL]** for this version)         |





<h2 id="2019-08-14-4">4. Neural Machine Translation with Noisy Lexical Constraints</h2> 
Title: [Neural Machine Translation with Noisy Lexical Constraints](https://arxiv.org/abs/1908.04664)

Authors: [Huayang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+H), [Guoping Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+G), [Lemao Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+L)

*(Submitted on 13 Aug 2019)*

> Lexically constrained decoding for machine translation has shown to be beneficial in previous studies. Unfortunately, constraints provided by users may contain mistakes in real-world situations. It is still an open question that how to manipulate these noisy constraints in such practical scenarios. We present a novel framework that treats constraints as external memories. In this soft manner, a mistaken constraint can be corrected. Experiments demonstrate that our approach can achieve substantial BLEU gains in handling noisy constraints. These results motivate us to apply the proposed approach on a new scenario where constraints are generated without the help of users. Experiments show that our approach can indeed improve the translation quality with the automatically generated constraints.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1908.04664 [cs.CL]**                         |
|           | (or **arXiv:1908.04664v1 [cs.CL]** for this version) |



# 2019-08-13

[Return to Index](#Index)

<h2 id="2019-08-13-1">1. On the Validity of Self-Attention as Explanation in Transformer Models</h2> 
Title: [On the Validity of Self-Attention as Explanation in Transformer Models](https://arxiv.org/abs/1908.04211)

Authors: [Gino Brunner](https://arxiv.org/search/cs?searchtype=author&query=Brunner%2C+G), [Yang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Damián Pascual](https://arxiv.org/search/cs?searchtype=author&query=Pascual%2C+D), [Oliver Richter](https://arxiv.org/search/cs?searchtype=author&query=Richter%2C+O), [Roger Wattenhofer](https://arxiv.org/search/cs?searchtype=author&query=Wattenhofer%2C+R)

*(Submitted on 12 Aug 2019)*

> Explainability of deep learning systems is a vital requirement for many applications. However, it is still an unsolved problem. Recent self-attention based models for natural language processing, such as the Transformer or BERT, offer hope of greater explainability by providing attention maps that can be directly inspected. Nevertheless, by just looking at the attention maps one often overlooks that the attention is not over words but over hidden embeddings, which themselves can be mixed representations of multiple embeddings. We investigate to what extent the implicit assumption made in many recent papers - that hidden embeddings at all layers still correspond to the underlying words - is justified. We quantify how much embeddings are mixed based on a gradient based attribution method and find that already after the first layer less than 50% of the embedding is attributed to the underlying word, declining thereafter to a median contribution of 7.5% in the last layer. While throughout the layers the underlying word remains as the one contributing most to the embedding, we argue that attention visualizations are misleading and should be treated with care when explaining the underlying deep learning system.

| Comments:    | Preprint. Work in progress                                   |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| MSC classes: | 46-04                                                        |
| ACM classes: | I.2.7; I.7.0                                                 |
| Cite as:     | **arXiv:1908.04211 [cs.CL]**                                 |
|              | (or **arXiv:1908.04211v1 [cs.CL]** for this version)         |



# 2019-08-12

[Return to Index](#Index)

<h2 id="2019-08-12-1">1. Exploiting Cross-Lingual Speaker and Phonetic Diversity for Unsupervised Subword Modeling</h2> 
Title: [Exploiting Cross-Lingual Speaker and Phonetic Diversity for Unsupervised Subword Modeling](https://arxiv.org/abs/1908.03538)

Authors: [Siyuan Feng](https://arxiv.org/search/eess?searchtype=author&query=Feng%2C+S), [Tan Lee](https://arxiv.org/search/eess?searchtype=author&query=Lee%2C+T)

*(Submitted on 9 Aug 2019)*

> This research addresses the problem of acoustic modeling of low-resource languages for which transcribed training data is absent. The goal is to learn robust frame-level feature representations that can be used to identify and distinguish subword-level speech units. The proposed feature representations comprise various types of multilingual bottleneck features (BNFs) that are obtained via multi-task learning of deep neural networks (MTL-DNN). One of the key problems is how to acquire high-quality frame labels for untranscribed training data to facilitate supervised DNN training. It is shown that learning of robust BNF representations can be achieved by effectively leveraging transcribed speech data and well-trained automatic speech recognition (ASR) systems from one or more out-of-domain (resource-rich) languages. Out-of-domain ASR systems can be applied to perform speaker adaptation with untranscribed training data of the target language, and to decode the training speech into frame-level labels for DNN training. It is also found that better frame labels can be generated by considering temporal dependency in speech when performing frame clustering. The proposed methods of feature learning are evaluated on the standard task of unsupervised subword modeling in Track 1 of the ZeroSpeech 2017 Challenge. The best performance achieved by our system is 9.7% in terms of across-speaker triphone minimal-pair ABX error rate, which is comparable to the best systems reported recently. Lastly, our investigation reveals that the closeness between target languages and out-of-domain languages and the amount of available training data for individual target languages could have significant impact on the goodness of learned features.

| Comments: | 12 pages, 6 figures. This manuscript has been accepted for publication as a regular paper in the IEEE Transactions on Audio, Speech and Language Processing |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL) |
| Cite as:  | **arXiv:1908.03538 [eess.AS]**                               |
|           | (or **arXiv:1908.03538v1 [eess.AS]** for this version)       |







<h2 id="2019-08-12-2">2. UdS Submission for the WMT 19 Automatic Post-Editing Task</h2> 
Title: [UdS Submission for the WMT 19 Automatic Post-Editing Task](https://arxiv.org/abs/1908.03402)

Authors: [Hongfei Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+H), [Qiuhui Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Q), [Josef van Genabith](https://arxiv.org/search/cs?searchtype=author&query=van+Genabith%2C+J)

*(Submitted on 9 Aug 2019)*

> In this paper, we describe our submission to the English-German APE shared task at WMT 2019. We utilize and adapt an NMT architecture originally developed for exploiting context information to APE, implement this in our own transformer model and explore joint training of the APE task with a de-noising encoder.

| Comments: | WMT 2019 Automatic Post-Editing Shared Task Paper    |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1908.03402 [cs.CL]**                         |
|           | (or **arXiv:1908.03402v1 [cs.CL]** for this version) |









# 2019-08-09

[Return to Index](#Index)

<h2 id="2019-08-09-1">1. A Test Suite and Manual Evaluation of Document-Level NMT at WMT19</h2> 
Title: [A Test Suite and Manual Evaluation of Document-Level NMT at WMT19](https://arxiv.org/abs/1908.03043)

Authors: [Kateřina Rysová](https://arxiv.org/search/cs?searchtype=author&query=Rysová%2C+K), [Magdaléna Rysová](https://arxiv.org/search/cs?searchtype=author&query=Rysová%2C+M), [Tomáš Musil](https://arxiv.org/search/cs?searchtype=author&query=Musil%2C+T), [Lucie Poláková](https://arxiv.org/search/cs?searchtype=author&query=Poláková%2C+L), [Ondřej Bojar](https://arxiv.org/search/cs?searchtype=author&query=Bojar%2C+O)

*(Submitted on 8 Aug 2019)*

> As the quality of machine translation rises and neural machine translation (NMT) is moving from sentence to document level translations, it is becoming increasingly difficult to evaluate the output of translation systems. 
> We provide a test suite for WMT19 aimed at assessing discourse phenomena of MT systems participating in the News Translation Task. We have manually checked the outputs and identified types of translation errors that are relevant to document-level translation.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1908.03043 [cs.CL]**                         |
|           | (or **arXiv:1908.03043v1 [cs.CL]** for this version) |



# 2019-08-07

[Return to Index](#Index)

<h2 id="2019-08-07-1">1. MacNet: Transferring Knowledge from Machine Comprehension to Sequence-to-Sequence Models</h2> 
Title: [MacNet: Transferring Knowledge from Machine Comprehension to Sequence-to-Sequence Models](https://arxiv.org/abs/1908.01816)

Authors: [Boyuan Pan](https://arxiv.org/search/cs?searchtype=author&query=Pan%2C+B), [Yazheng Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Y), [Hao Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+H), [Zhou Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+Z), [Yueting Zhuang](https://arxiv.org/search/cs?searchtype=author&query=Zhuang%2C+Y), [Deng Cai](https://arxiv.org/search/cs?searchtype=author&query=Cai%2C+D), [Xiaofei He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+X)

*(Submitted on 23 Jul 2019)*

> Machine Comprehension (MC) is one of the core problems in natural language processing, requiring both understanding of the natural language and knowledge about the world. Rapid progress has been made since the release of several benchmark datasets, and recently the state-of-the-art models even surpass human performance on the well-known SQuAD evaluation. In this paper, we transfer knowledge learned from machine comprehension to the sequence-to-sequence tasks to deepen the understanding of the text. We propose MacNet: a novel encoder-decoder supplementary architecture to the widely used attention-based sequence-to-sequence models. Experiments on neural machine translation (NMT) and abstractive text summarization show that our proposed framework can significantly improve the performance of the baseline models, and our method for the abstractive text summarization achieves the state-of-the-art results on the Gigaword dataset.

| Comments: | Accepted In NeurIPS 2018                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1908.01816 [cs.CL]**                                 |
|           | (or **arXiv:1908.01816v1 [cs.CL]** for this version)         |





<h2 id="2019-08-07-2">2. A Translate-Edit Model for Natural Language Question to SQL Query Generation on Multi-relational Healthcare Data</h2> 
Title: [A Translate-Edit Model for Natural Language Question to SQL Query Generation on Multi-relational Healthcare Data](https://arxiv.org/abs/1908.01839)

Authors: [Ping Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+P), [Tian Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+T), [Chandan K. Reddy](https://arxiv.org/search/cs?searchtype=author&query=Reddy%2C+C+K)

*(Submitted on 28 Jul 2019)*

> Electronic health record (EHR) data contains most of the important patient health information and is typically stored in a relational database with multiple tables. One important way for doctors to make use of EHR data is to retrieve intuitive information by posing a sequence of questions against it. However, due to a large amount of information stored in it, effectively retrieving patient information from EHR data in a short time is still a challenging issue for medical experts since it requires a good understanding of a query language to get access to the database. We tackle this challenge by developing a deep learning based approach that can translate a natural language question on multi-relational EHR data into its corresponding SQL query, which is referred to as a Question-to-SQL generation task. Most of the existing methods cannot solve this problem since they primarily focus on tackling the questions related to a single table under the table-aware assumption. While in our problem, it is possible that questions asked by clinicians are related to multiple unspecified tables. In this paper, we first create a new question to query dataset designed for healthcare to perform the Question-to-SQL generation task, named MIMICSQL, based on a publicly available electronic medical database. To address the challenge of generating queries on multi-relational databases from natural language questions, we propose a TRanslate-Edit Model for Question-to-SQL query (TREQS), which adopts the sequence-to-sequence model to directly generate SQL query for a given question, and further edits it with an attentive-copying mechanism and task-specific look-up tables. Both quantitative and qualitative experimental results indicate the flexibility and efficiency of our proposed method in tackling challenges that are unique in MIMICSQL.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Information Retrieval (cs.IR); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1908.01839 [cs.CL]**                                 |
|           | (or **arXiv:1908.01839v1 [cs.CL]** for this version)         |





<h2 id="2019-08-07-3">3. Self-Knowledge Distillation in Natural Language Processing</h2> 
Title: [Self-Knowledge Distillation in Natural Language Processing](https://arxiv.org/abs/1908.01851)

Authors: [Sangchul Hahn](https://arxiv.org/search/cs?searchtype=author&query=Hahn%2C+S), [Heeyoul Choi](https://arxiv.org/search/cs?searchtype=author&query=Choi%2C+H)

*(Submitted on 2 Aug 2019)*

> Since deep learning became a key player in natural language processing (NLP), many deep learning models have been showing remarkable performances in a variety of NLP tasks, and in some cases, they are even outperforming humans. Such high performance can be explained by efficient knowledge representation of deep learning models. While many methods have been proposed to learn more efficient representation, knowledge distillation from pretrained deep networks suggest that we can use more information from the soft target probability to train other neural networks. In this paper, we propose a new knowledge distillation method self-knowledge distillation, based on the soft target probabilities of the training model itself, where multimode information is distilled from the word embedding space right below the softmax layer. Due to the time complexity, our method approximates the soft target probabilities. In experiments, we applied the proposed method to two different and fundamental NLP tasks: language model and neural machine translation. The experiment results show that our proposed method improves performance on the tasks.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1908.01851 [cs.CL]**                                 |
|           | (or **arXiv:1908.01851v1 [cs.CL]** for this version)         |



# 2019-08-06

[Return to Index](#Index)

<h2 id="2019-08-06-1">1. Invariance-based Adversarial Attack on Neural Machine Translation Systems</h2> 
Title: [Invariance-based Adversarial Attack on Neural Machine Translation Systems](https://arxiv.org/abs/1908.01165)

Authors: [Akshay Chaturvedi](https://arxiv.org/search/cs?searchtype=author&query=Chaturvedi%2C+A), [Abijith KP](https://arxiv.org/search/cs?searchtype=author&query=KP%2C+A), [Utpal Garain](https://arxiv.org/search/cs?searchtype=author&query=Garain%2C+U)

*(Submitted on 3 Aug 2019)*

> Recently, NLP models have been shown to be susceptible to adversarial attacks. In this paper, we explore adversarial attacks on neural machine translation (NMT) systems. Given a sentence in the source language, the goal of the proposed attack is to change multiple words while ensuring that the predicted translation remains unchanged. In order to choose the word from the source vocabulary, we propose a soft-attention based technique. The experiments are conducted on two language pairs: English-German (en-de) and English-French (en-fr) and two state-of-the-art NMT systems: BLSTM-based encoder-decoder with attention and Transformer. The proposed soft-attention based technique outperforms existing methods like HotFlip by a significant margin for all the conducted experiments The results demonstrate that state-of-the-art NMT systems are unable to capture the semantics of the source language.

| Comments: | Under review in IEEE/ACM TASLP                               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Cryptography and Security (cs.CR); Machine Learning (stat.ML) |
| Cite as:  | **arXiv:1908.01165 [cs.LG]**                                 |
|           | (or **arXiv:1908.01165v1 [cs.LG]** for this version)         |



<h2 id="2019-08-06-2">2. Performance Evaluation of Supervised Machine Learning Techniques for Efficient Detection of Emotions from Online Content</h2> 
Title: [Performance Evaluation of Supervised Machine Learning Techniques for Efficient Detection of Emotions from Online Content](https://arxiv.org/abs/1908.01587)

Authors: [Muhammad Zubair Asghar](https://arxiv.org/search/cs?searchtype=author&query=Asghar%2C+M+Z), [Fazli Subhan](https://arxiv.org/search/cs?searchtype=author&query=Subhan%2C+F), [Muhammad Imran](https://arxiv.org/search/cs?searchtype=author&query=Imran%2C+M), [Fazal Masud Kundi](https://arxiv.org/search/cs?searchtype=author&query=Kundi%2C+F+M), [Shahboddin Shamshirband](https://arxiv.org/search/cs?searchtype=author&query=Shamshirband%2C+S), [Amir Mosavi](https://arxiv.org/search/cs?searchtype=author&query=Mosavi%2C+A), [Peter Csiba](https://arxiv.org/search/cs?searchtype=author&query=Csiba%2C+P), [Annamaria R. Varkonyi-Koczy](https://arxiv.org/search/cs?searchtype=author&query=Varkonyi-Koczy%2C+A+R)

*(Submitted on 5 Aug 2019)*

> Emotion detection from the text is an important and challenging problem in text analytics. The opinion-mining experts are focusing on the development of emotion detection applications as they have received considerable attention of online community including users and business organization for collecting and interpreting public emotions. However, most of the existing works on emotion detection used less efficient machine learning classifiers with limited datasets, resulting in performance degradation. To overcome this issue, this work aims at the evaluation of the performance of different machine learning classifiers on a benchmark emotion dataset. The experimental results show the performance of different machine learning classifiers in terms of different evaluation metrics like precision, recall ad f-measure. Finally, a classifier with the best performance is recommended for the emotion classification.

| Comments:    | 30 pages, 13 tables, 1 figure                                |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Information Retrieval (cs.IR)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| MSC classes: | 68T01                                                        |
| DOI:         | [10.20944/preprints201908.0019.v1](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.20944%2Fpreprints201908.0019.v1&v=7cef63df) |
| Cite as:     | **arXiv:1908.01587 [cs.IR]**                                 |
|              | (or **arXiv:1908.01587v1 [cs.IR]** for this version)         |





<h2 id="2019-08-06-3">3. The TALP-UPC System for the WMT Similar Language Task: Statistical vs Neural Machine Translation</h2> 
Title: [The TALP-UPC System for the WMT Similar Language Task: Statistical vs Neural Machine Translation](https://arxiv.org/abs/1908.01192)

Authors: [Magdalena Biesialska](https://arxiv.org/search/cs?searchtype=author&query=Biesialska%2C+M), [Lluis Guardia](https://arxiv.org/search/cs?searchtype=author&query=Guardia%2C+L), [Marta R. Costa-jussà](https://arxiv.org/search/cs?searchtype=author&query=Costa-jussà%2C+M+R)

*(Submitted on 3 Aug 2019)*

> Although the problem of similar language translation has been an area of research interest for many years, yet it is still far from being solved. In this paper, we study the performance of two popular approaches: statistical and neural. We conclude that both methods yield similar results; however, the performance varies depending on the language pair. While the statistical approach outperforms the neural one by a difference of 6 BLEU points for the Spanish-Portuguese language pair, the proposed neural model surpasses the statistical one by a difference of 2 BLEU points for Czech-Polish. In the former case, the language similarity (based on perplexity) is much higher than in the latter case. Additionally, we report negative results for the system combination with back-translation. Our TALP-UPC system submission won 1st place for Czech-to-Polish and 2nd place for Spanish-to-Portuguese in the official evaluation of the 1st WMT Similar Language Translation task.

| Comments: | WMT 2019 Shared Task paper                           |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1908.01192 [cs.CL]**                         |
|           | (or **arXiv:1908.01192v1 [cs.CL]** for this version) |





<h2 id="2019-08-06-4">4. JUMT at WMT2019 News Translation Task: A Hybrid approach to Machine Translation for Lithuanian to English</h2> 
Title: [JUMT at WMT2019 News Translation Task: A Hybrid approach to Machine Translation for Lithuanian to English](https://arxiv.org/abs/1908.01349)

Authors: [Sainik Kumar Mahata](https://arxiv.org/search/cs?searchtype=author&query=Mahata%2C+S+K), [Avishek Garain](https://arxiv.org/search/cs?searchtype=author&query=Garain%2C+A), [Adityar Rayala](https://arxiv.org/search/cs?searchtype=author&query=Rayala%2C+A), [Dipankar Das](https://arxiv.org/search/cs?searchtype=author&query=Das%2C+D), [Sivaji Bandyopadhyay](https://arxiv.org/search/cs?searchtype=author&query=Bandyopadhyay%2C+S)

*(Submitted on 1 Aug 2019)*

> In the current work, we present a description of the system submitted to WMT 2019 News Translation Shared task. The system was created to translate news text from Lithuanian to English. To accomplish the given task, our system used a Word Embedding based Neural Machine Translation model to post edit the outputs generated by a Statistical Machine Translation model. The current paper documents the architecture of our model, descriptions of the various modules and the results produced using the same. Our system garnered a BLEU score of 17.6.

| Comments: | arXiv admin note: substantial text overlap with [arXiv:1908.00323](https://arxiv.org/abs/1908.00323) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1908.01349 [cs.CL]**                                 |
|           | (or **arXiv:1908.01349v1 [cs.CL]** for this version)         |





<h2 id="2019-08-06-5">5. Beyond English-only Reading Comprehension: Experiments in Zero-Shot Multilingual Transfer for Bulgarian</h2> 
Title: [Beyond English-only Reading Comprehension: Experiments in Zero-Shot Multilingual Transfer for Bulgarian](https://arxiv.org/abs/1908.01519)

Authors: [Momchil Hardalov](https://arxiv.org/search/cs?searchtype=author&query=Hardalov%2C+M), [Ivan Koychev](https://arxiv.org/search/cs?searchtype=author&query=Koychev%2C+I), [Preslav Nakov](https://arxiv.org/search/cs?searchtype=author&query=Nakov%2C+P)

*(Submitted on 5 Aug 2019)*

> Recently, reading comprehension models achieved near-human performance on large-scale datasets such as SQuAD, CoQA, MS Macro, RACE, etc. This is largely due to the release of pre-trained contextualized representations such as BERT and ELMo, which can be fine-tuned for the target task. Despite those advances and the creation of more challenging datasets, most of the work is still done for English. Here, we study the effectiveness of multilingual BERT fine-tuned on large-scale English datasets for reading comprehension (e.g., for RACE), and we apply it to Bulgarian multiple-choice reading comprehension. We propose a new dataset containing 2,221 questions from matriculation exams for twelfth grade in various subjects -history, biology, geography and philosophy-, and 412 additional questions from online quizzes in history. While the quiz authors gave no relevant context, we incorporate knowledge from Wikipedia, retrieving documents matching the combination of question + each answer option. Moreover, we experiment with different indexing and pre-training strategies. The evaluation results show accuracy of 42.23%, which is well above the baseline of 24.89%.

| Comments: | Accepted at RANLP 2019 (13 pages, 2 figures, 6 tables)       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Information Retrieval (cs.IR) |
| Cite as:  | **arXiv:1908.01519 [cs.CL]**                                 |
|           | (or **arXiv:1908.01519v1 [cs.CL]** for this version)         |





<h2 id="2019-08-06-6">6. Predicting Actions to Help Predict Translations</h2> 
Title: [Predicting Actions to Help Predict Translations](https://arxiv.org/abs/1908.01665)

Authors: [Zixiu Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+Z), [Julia Ive](https://arxiv.org/search/cs?searchtype=author&query=Ive%2C+J), [Josiah Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J), [Pranava Madhyastha](https://arxiv.org/search/cs?searchtype=author&query=Madhyastha%2C+P), [Lucia Specia](https://arxiv.org/search/cs?searchtype=author&query=Specia%2C+L)

*(Submitted on 5 Aug 2019)*

> We address the task of text translation on the How2 dataset using a state of the art transformer-based multimodal approach. The question we ask ourselves is whether visual features can support the translation process, in particular, given that this is a dataset extracted from videos, we focus on the translation of actions, which we believe are poorly captured in current static image-text datasets currently used for multimodal translation. For that purpose, we extract different types of action features from the videos and carefully investigate how helpful this visual information is by testing whether it can increase translation quality when used in conjunction with (i) the original text and (ii) the original text where action-related words (or all verbs) are masked out. The latter is a simulation that helps us assess the utility of the image in cases where the text does not provide enough context about the action, or in the presence of noise in the input text.

| Comments: | Accepted to workshop "The How2 Challenge: New Tasks for Vision & Language" of International Conference on Machine Learning 2019 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1908.01665 [cs.CL]**                                 |
|           | (or **arXiv:1908.01665v1 [cs.CL]** for this version)         |





<h2 id="2019-08-06-7">7. Thoth: Improved Rapid Serial Visual Presentation using Natural Language Processing</h2> 
Title: [Thoth: Improved Rapid Serial Visual Presentation using Natural Language Processing](https://arxiv.org/abs/1908.01699)

Authors: [David Awad](https://arxiv.org/search/cs?searchtype=author&query=Awad%2C+D)

*(Submitted on 5 Aug 2019)*

> Thoth is a tool designed to combine many different types of speed reading technology. The largest insight is using natural language parsing for more optimal rapid serial visual presentation and more effective reading information.

| Comments: | 10 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Human-Computer Interaction (cs.HC) |
| Cite as:  | **arXiv:1908.01699 [cs.CL]**                                 |
|           | (or **arXiv:1908.01699v1 [cs.CL]** for this version)         |





# 2019-08-02

[Return to Index](#Index)

<h2 id="2019-08-02-1">1. Tree-Transformer: A Transformer-Based Method for Correction of Tree-Structured Data</h2> 
Title: [Tree-Transformer: A Transformer-Based Method for Correction of Tree-Structured Data](https://arxiv.org/abs/1908.00449)

Authors: [Jacob Harer](https://arxiv.org/search/cs?searchtype=author&query=Harer%2C+J), [Chris Reale](https://arxiv.org/search/cs?searchtype=author&query=Reale%2C+C), [Peter Chin](https://arxiv.org/search/cs?searchtype=author&query=Chin%2C+P)

*(Submitted on 1 Aug 2019)*

> Many common sequential data sources, such as source code and natural language, have a natural tree-structured representation. These trees can be generated by fitting a sequence to a grammar, yielding a hierarchical ordering of the tokens in the sequence. This structure encodes a high degree of syntactic information, making it ideal for problems such as grammar correction. However, little work has been done to develop neural networks that can operate on and exploit tree-structured data. In this paper we present the Tree-Transformer \textemdash{} a novel neural network architecture designed to translate between arbitrary input and output trees. We applied this architecture to correction tasks in both the source code and natural language domains. On source code, our model achieved an improvement of 25% F0.5 over the best sequential method. On natural language, we achieved comparable results to the most complex state of the art systems, obtaining a 10% improvement in recall on the CoNLL 2014 benchmark and the highest to date F0.5 score on the AESW benchmark of 50.43.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1908.00449 [cs.LG]**                                 |
|           | (or **arXiv:1908.00449v1 [cs.LG]** for this version)         |



<h2 id="2019-08-02-2">2. Learning Joint Acoustic-Phonetic Word Embeddings</h2> 
Title: [Learning Joint Acoustic-Phonetic Word Embeddings](https://arxiv.org/abs/1908.00493)

Authors: [Mohamed El-Geish](https://arxiv.org/search/cs?searchtype=author&query=El-Geish%2C+M)

*(Submitted on 1 Aug 2019)*

> Most speech recognition tasks pertain to mapping words across two modalities: acoustic and orthographic. In this work, we suggest learning encoders that map variable-length, acoustic or phonetic, sequences that represent words into fixed-dimensional vectors in a shared latent space; such that the distance between two word vectors represents how closely the two words sound. Instead of directly learning the distances between word vectors, we employ weak supervision and model a binary classification task to predict whether two inputs, one of each modality, represent the same word given a distance threshold. We explore various deep-learning models, bimodal contrastive losses, and techniques for mining hard negative examples such as the semi-supervised technique of self-labeling. Our best model achieves an F1 score of 0.95 for the binary classification task.

| Comments: | 8 pages, 4 figures                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS); Machine Learning (stat.ML) |
| Cite as:  | **arXiv:1908.00493 [cs.LG]**                                 |
|           | (or **arXiv:1908.00493v1 [cs.LG]** for this version)         |



<h2 id="2019-08-02-3">3. JUCBNMT at WMT2018 News Translation Task: Character Based Neural Machine Translation of Finnish to English</h2> 
Title: [JUCBNMT at WMT2018 News Translation Task: Character Based Neural Machine Translation of Finnish to English](https://arxiv.org/abs/1908.00323)

Authors: [Sainik Kumar Mahata](https://arxiv.org/search/cs?searchtype=author&query=Mahata%2C+S+K), [Dipankar Das](https://arxiv.org/search/cs?searchtype=author&query=Das%2C+D), [Sivaji Bandyopadhyay](https://arxiv.org/search/cs?searchtype=author&query=Bandyopadhyay%2C+S)

*(Submitted on 1 Aug 2019)*

> In the current work, we present a description of the system submitted to WMT 2018 News Translation Shared task. The system was created to translate news text from Finnish to English. The system used a Character Based Neural Machine Translation model to accomplish the given task. The current paper documents the preprocessing steps, the description of the submitted system and the results produced using the same. Our system garnered a BLEU score of 12.9.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1908.00323 [cs.CL]**                         |
|           | (or **arXiv:1908.00323v1 [cs.CL]** for this version) |