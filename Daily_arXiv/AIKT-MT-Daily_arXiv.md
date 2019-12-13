# Daily arXiv: Machine Translation - Dec., 2019

# Index

- [2019-12-13](#2019-12-13)
  - [1. Improving Interpretability of Word Embeddings by Generating Definition and Usage](#2019-12-13-1)
- [2019-12-12](#2019-12-12)
  - [1. Lifelong learning for text retrieval and recognition in historical handwritten document collections](#2019-12-12-1)
  - [2. Unsupervised Neural Dialect Translation with Commonality and Diversity Modeling](#2019-12-12-2)
  - [3. Automatic Spanish Translation of the SQuAD Dataset for Multilingual Question Answering](#2019-12-12-3)
  - [4. MetaMT,a MetaLearning Method Leveraging Multiple Domain Data for Low Resource Machine Translation](#2019-12-12-4)
- [2019-12-11](#2019-12-11)
  - [1. Cross-Language Aphasia Detection using Optimal Transport Domain Adaptation](#2019-12-11-1)
  - [2. GeBioToolkit: Automatic Extraction of Gender-Balanced Multilingual Corpus of Wikipedia Biographies](#2019-12-11-2)
- [2019-12-10](#2019-12-10)
  - [1. Bidirectional Scene Text Recognition with a Single Decoder](#2019-12-10-1)
  - [2. Explaining Sequence-Level Knowledge Distillation as Data-Augmentation for Neural Machine Translation](#2019-12-10-2)
  - [3. Re-Translation Strategies For Long Form, Simultaneous, Spoken Language Translation](#2019-12-10-3)
  - [4. PidginUNMT: Unsupervised Neural Machine Translation from West African Pidgin to English](#2019-12-10-4)
- [2019-12-09](#2019-12-09)
  - [1. Machine Translation Evaluation Meets Community Question Answering](#2019-12-09-1)
  - [2. Pairwise Neural Machine Translation Evaluation](#2019-12-09-2)
- [2019-12-06](#2019-12-06)
  - [1. Exploration of Neural Machine Translation in Autoformalization of Mathematics in Mizar](#2019-12-06-1)
- [2019-12-05](#2019-12-05)
  - [1. Neural Machine Translation: A Review](#2019-12-05-1)
  - [2. A Robust Self-Learning Method for Fully Unsupervised Cross-Lingual Mappings of Word Embeddings: Making the Method Robustly Reproducible as Well](#2019-12-05-2)
  - [3. Acquiring Knowledge from Pre-trained Model to Neural Machine Translation](#2019-12-05-3)
- [2019-12-04](#2019-12-04)
  - [1. Cross-lingual Pre-training Based Transfer for Zero-shot Neural Machine Translation](#2019-12-04-1)
- [2019-12-03](#2019-12-03)
  - [1. Not All Attention Is Needed: Gated Attention Network for Sequence Data](#2019-12-03-1)
  - [2. Modeling Fluency and Faithfulness for Diverse Neural Machine Translation](#2019-12-03-2)
  - [3. Merging External Bilingual Pairs into Neural Machine Translation](#2019-12-03-3)
- [2019-12-02](#2019-12-02)
  - [1. DiscoTK: Using Discourse Structure for Machine Translation Evaluation](#2019-12-02-1)
  - [2. Multimodal Machine Translation through Visuals and Speech](#2019-12-02-2)
  - [3. GitHub Typo Corpus: A Large-Scale Multilingual Dataset of Misspellings and Grammatical Errors](#2019-12-02-3)
  - [4. Neural Chinese Word Segmentation as Sequence to Sequence Translation](#2019-12-02-4)
- [2019-11](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-11.md)
- [2019-10](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-10.md)
- [2019-09](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-09.md)
- [2019-08](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-08.md)
- [2019-07](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-07.md)
- [2019-06](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-06.md)
- [2019-05](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-05.md)
- [2019-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-04.md)
- [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)



# 2019-12-13

[Return to Index](#Index)



<h2 id="2019-12-13-1">1. Improving Interpretability of Word Embeddings by Generating Definition and Usage</h2>

Title: [Improving Interpretability of Word Embeddings by Generating Definition and Usage](https://arxiv.org/abs/1912.05898)

Authors: [Haitong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+H), [Yongping Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+Y), [Jiaxin Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+J), [Qingxiao Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Q)

*(Submitted on 12 Dec 2019)*

> Word Embeddings, which encode semantic and syntactic features, have achieved success in many natural language processing tasks recently. However, the lexical semantics captured by these embeddings are difficult to interpret due to the dense vector representations. In order to improve the interpretability of word vectors, we explore definition modeling task and propose a novel framework (Semantics-Generator) to generate more reasonable and understandable context-dependent definitions. Moreover, we introduce usage modeling and study whether it is possible to utilize distributed representations to generate example sentences of words. These ways of semantics generation are a more direct and explicit expression of embedding's semantics. Two multi-task learning methods are used to combine usage modeling and definition modeling. To verify our approach, we construct Oxford-2019 dataset, where each entry contains word, context, example sentence and corresponding definition. Experimental results show that Semantics-Generator achieves the state-of-the-art result in definition modeling and the multi-task learning methods are helpful for two tasks to improve the performance.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1912.05898](https://arxiv.org/abs/1912.05898) [cs.CL] |
|           | (or [arXiv:1912.05898v1](https://arxiv.org/abs/1912.05898v1) [cs.CL] for this version) |



# 2019-12-12

[Return to Index](#Index)



<h2 id="2019-12-12-1">1. Lifelong learning for text retrieval and recognition in historical handwritten document collections</h2>
Title: [Lifelong learning for text retrieval and recognition in historical handwritten document collections](https://arxiv.org/abs/1912.05156)

Authors: [Lambert Schomaker](https://arxiv.org/search/cs?searchtype=author&query=Schomaker%2C+L)

*(Submitted on 11 Dec 2019)*

> This chapter provides an overview of the problems that need to be dealt with when constructing a lifelong-learning retrieval, recognition and indexing engine for large historical document collections in multiple scripts and languages, the Monk system. This application is highly variable over time, since the continuous labeling by end users changes the concept of what a 'ground truth' constitutes. Although current advances in deep learning provide a huge potential in this application domain, the scale of the problem, i.e., more than 520 hugely diverse books, documents and manuscripts precludes the current meticulous and painstaking human effort which is required in designing and developing successful deep-learning systems. The ball-park principle is introduced, which describes the evolution from the sparsely-labeled stage that can only be addressed by traditional methods or nearest-neighbor methods on embedded vectors of pre-trained neural networks, up to the other end of the spectrum where massive labeling allows reliable training of deep-learning methods. Contents: Introduction, Expectation management, Deep learning, The ball-park principle, Technical realization, Work flow, Quality and quantity of material, Industrialization and scalability, Human effort, Algorithms, Object of recognition, Processing pipeline, Performance,Compositionality, Conclusion.

| Comments: | To appear as chapter in book: Handwritten Historical Document Analysis, Recognition, and Retrieval -- State of the Art and Future Trends, in the book series: Series in Machine Perception and Artificial Intelligence World Scientific, ISSN (print): 1793-0839 Original version deposited at Zenodo: [this https URL](https://zenodo.org/record/2346885#.XfCfsq5ytpg) on December 17, 2018 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1912.05156](https://arxiv.org/abs/1912.05156) [cs.CV] |
|           | (or [arXiv:1912.05156v1](https://arxiv.org/abs/1912.05156v1) [cs.CV] for this version) |





<h2 id="2019-12-12-2">2. Unsupervised Neural Dialect Translation with Commonality and Diversity Modeling</h2>
Title: [Unsupervised Neural Dialect Translation with Commonality and Diversity Modeling](https://arxiv.org/abs/1912.05134)

Authors: [Yu Wan](https://arxiv.org/search/cs?searchtype=author&query=Wan%2C+Y), [Baosong Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+B), [Derek F. Wong](https://arxiv.org/search/cs?searchtype=author&query=Wong%2C+D+F), [Lidia S. Chao](https://arxiv.org/search/cs?searchtype=author&query=Chao%2C+L+S), [Haihua Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+H), [Ben C.H. Ao](https://arxiv.org/search/cs?searchtype=author&query=Ao%2C+B+C)

*(Submitted on 11 Dec 2019)*

> As a special machine translation task, dialect translation has two main characteristics: 1) lack of parallel training corpus; and 2) possessing similar grammar between two sides of the translation. In this paper, we investigate how to exploit the commonality and diversity between dialects thus to build unsupervised translation models merely accessing to monolingual data. Specifically, we leverage pivot-private embedding, layer coordination, as well as parameter sharing to sufficiently model commonality and diversity among source and target, ranging from lexical, through syntactic, to semantic levels. In order to examine the effectiveness of the proposed models, we collect 20 million monolingual corpus for each of Mandarin and Cantonese, which are official language and the most widely used dialect in China. Experimental results reveal that our methods outperform rule-based simplified and traditional Chinese conversion and conventional unsupervised translation models over 12 BLEU scores.

| Comments: | AAAI 2020                                                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1912.05134](https://arxiv.org/abs/1912.05134) [cs.CL] |
|           | (or [arXiv:1912.05134v1](https://arxiv.org/abs/1912.05134v1) [cs.CL] for this version) |





<h2 id="2019-12-12-3">3. Automatic Spanish Translation of the SQuAD Dataset for Multilingual Question Answering</h2>
Title: [Automatic Spanish Translation of the SQuAD Dataset for Multilingual Question Answering](https://arxiv.org/abs/1912.05200)

Authors: [Casimiro Pio Carrino](https://arxiv.org/search/cs?searchtype=author&query=Carrino%2C+C+P), [Marta Ruiz Costa-jussà](https://arxiv.org/search/cs?searchtype=author&query=Costa-jussà%2C+M+R), [José Adrián Rodríguez Fonollosa](https://arxiv.org/search/cs?searchtype=author&query=Fonollosa%2C+J+A+R)

*(Submitted on 11 Dec 2019)*

> Recently, multilingual question answering became a crucial research topic, and it is receiving increased interest in the NLP community. However, the unavailability of large-scale datasets makes it challenging to train multilingual QA systems with performance comparable to the English ones. In this work, we develop the Translate Align Retrieve (TAR) method to automatically translate the Stanford Question Answering Dataset (SQuAD) v1.1 to Spanish. We then used this dataset to train Spanish QA systems by fine-tuning a Multilingual-BERT model. Finally, we evaluated our QA models with the recently proposed MLQA and XQuAD benchmarks for cross-lingual Extractive QA. Experimental results show that our models outperform the previous Multilingual-BERT baselines achieving the new state-of-the-art value of 68.1 F1 points on the Spanish MLQA corpus and 77.6 F1 and 61.8 Exact Match points on the Spanish XQuAD corpus. The resulting, synthetically generated SQuAD-es v1.1 corpora, with almost 100% of data contained in the original English version, to the best of our knowledge, is the first large-scale QA training resource for Spanish.

| Comments: | Submitted to LREC 2020                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1912.05200](https://arxiv.org/abs/1912.05200) [cs.CL] |
|           | (or [arXiv:1912.05200v1](https://arxiv.org/abs/1912.05200v1) [cs.CL] for this version) |





<h2 id="2019-12-12-4">4. MetaMT,a MetaLearning Method Leveraging Multiple Domain Data for Low Resource Machine Translation</h2>
Title: [MetaMT,a MetaLearning Method Leveraging Multiple Domain Data for Low Resource Machine Translation](https://arxiv.org/abs/1912.05467)

Authors: [Rumeng Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+R), [Xun Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Hong Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+H)

*(Submitted on 11 Dec 2019)*

> Manipulating training data leads to robust neural models for MT.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1912.05467](https://arxiv.org/abs/1912.05467) [cs.CL] |
|           | (or [arXiv:1912.05467v1](https://arxiv.org/abs/1912.05467v1) [cs.CL] for this version) |





# 2019-12-11

[Return to Index](#Index)



<h2 id="2019-12-11-1">1. Cross-Language Aphasia Detection using Optimal Transport Domain Adaptation</h2>
Title: [Cross-Language Aphasia Detection using Optimal Transport Domain Adaptation](https://arxiv.org/abs/1912.04370)

Authors: [Aparna Balagopalan](https://arxiv.org/search/eess?searchtype=author&query=Balagopalan%2C+A), [Jekaterina Novikova](https://arxiv.org/search/eess?searchtype=author&query=Novikova%2C+J), [Matthew B. A. McDermott](https://arxiv.org/search/eess?searchtype=author&query=McDermott%2C+M+B+A), [Bret Nestor](https://arxiv.org/search/eess?searchtype=author&query=Nestor%2C+B), [Tristan Naumann](https://arxiv.org/search/eess?searchtype=author&query=Naumann%2C+T), [Marzyeh Ghassemi](https://arxiv.org/search/eess?searchtype=author&query=Ghassemi%2C+M)

*(Submitted on 4 Dec 2019)*

> Multi-language speech datasets are scarce and often have small sample sizes in the medical domain. Robust transfer of linguistic features across languages could improve rates of early diagnosis and therapy for speakers of low-resource languages when detecting health conditions from speech. We utilize out-of-domain, unpaired, single-speaker, healthy speech data for training multiple Optimal Transport (OT) domain adaptation systems. We learn mappings from other languages to English and detect aphasia from linguistic characteristics of speech, and show that OT domain adaptation improves aphasia detection over unilingual baselines for French (6% increased F1) and Mandarin (5% increased F1). Further, we show that adding aphasic data to the domain adaptation system significantly increases performance for both French and Mandarin, increasing the F1 scores further (10% and 8% increase in F1 scores for French and Mandarin, respectively, over unilingual baselines).

| Comments: | Accepted to ML4H at NeurIPS 2019                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD); Machine Learning (stat.ML) |
| Cite as:  | [arXiv:1912.04370](https://arxiv.org/abs/1912.04370) [eess.AS] |
|           | (or [arXiv:1912.04370v1](https://arxiv.org/abs/1912.04370v1) [eess.AS] for this version) |





<h2 id="2019-12-11-2">2. GeBioToolkit: Automatic Extraction of Gender-Balanced Multilingual Corpus of Wikipedia Biographies</h2>
Title: [GeBioToolkit: Automatic Extraction of Gender-Balanced Multilingual Corpus of Wikipedia Biographies](https://arxiv.org/abs/1912.04778)

Authors: [Marta R. Costa-jussà](https://arxiv.org/search/cs?searchtype=author&query=Costa-jussà%2C+M+R), [Pau Li Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+P+L), [Cristina España-Bonet](https://arxiv.org/search/cs?searchtype=author&query=España-Bonet%2C+C)

*(Submitted on 10 Dec 2019)*

> We introduce GeBioToolkit, a tool for extracting multilingual parallel corpora at sentence level, with document and gender information from Wikipedia biographies. Despite thegender inequalitiespresent in Wikipedia, the toolkit has been designed to extract corpus balanced in gender. While our toolkit is customizable to any number of languages (and different domains), in this work we present a corpus of 2,000 sentences in English, Spanish and Catalan, which has been post-edited by native speakers to become a high-quality dataset for machinetranslation evaluation. While GeBioCorpus aims at being one of the first non-synthetic gender-balanced test datasets, GeBioToolkit aims at paving the path to standardize procedures to produce gender-balanced datasets

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1912.04778](https://arxiv.org/abs/1912.04778) [cs.CL] |
|           | (or [arXiv:1912.04778v1](https://arxiv.org/abs/1912.04778v1) [cs.CL] for this version) |







# 2019-12-10

[Return to Index](#Index)



<h2 id="2019-12-10-1">1. Bidirectional Scene Text Recognition with a Single Decoder</h2>
Title: [Bidirectional Scene Text Recognition with a Single Decoder](https://arxiv.org/abs/1912.03656)

Authors: [Maurits Bleeker](https://arxiv.org/search/cs?searchtype=author&query=Bleeker%2C+M), [Maarten de Rijke](https://arxiv.org/search/cs?searchtype=author&query=de+Rijke%2C+M)

*(Submitted on 8 Dec 2019)*

> Scene Text Recognition (STR) is the problem of recognizing the correct word or character sequence in a cropped word image. To obtain more robust output sequences, the notion of bidirectional STR has been introduced. So far, bidirectional STRs have been implemented by using two separate decoders; one for left-to-right decoding and one for right-to-left. Having two separate decoders for almost the same task with the same output space is undesirable from a computational and optimization point of view. We introduce the bidirectional Scene Text Transformer (Bi-STET), a novel bidirectional STR method with a single decoder for bidirectional text decoding. With its single decoder, Bi-STET outperforms methods that apply bidirectional decoding by using two separate decoders while also being more efficient than those methods, Furthermore, we achieve or beat state-of-the-art (SOTA) methods on all STR benchmarks with Bi-STET. Finally, we provide analyses and insights into the performance of Bi-STET.

| Comments: | 8 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1912.03656](https://arxiv.org/abs/1912.03656) [cs.CV] |
|           | (or [arXiv:1912.03656v1](https://arxiv.org/abs/1912.03656v1) [cs.CV] for this version) |





<h2 id="2019-12-10-2">2. Explaining Sequence-Level Knowledge Distillation as Data-Augmentation for Neural Machine Translation</h2>
Title: [Explaining Sequence-Level Knowledge Distillation as Data-Augmentation for Neural Machine Translation](https://arxiv.org/abs/1912.03334)

Authors: [Mitchell A. Gordon](https://arxiv.org/search/cs?searchtype=author&query=Gordon%2C+M+A), [Kevin Duh](https://arxiv.org/search/cs?searchtype=author&query=Duh%2C+K)

*(Submitted on 6 Dec 2019)*

> Sequence-level knowledge distillation (SLKD) is a model compression technique that leverages large, accurate teacher models to train smaller, under-parameterized student models. Why does pre-processing MT data with SLKD help us train smaller models? We test the common hypothesis that SLKD addresses a capacity deficiency in students by "simplifying" noisy data points and find it unlikely in our case. Models trained on concatenations of original and "simplified" datasets generalize just as well as baseline SLKD. We then propose an alternative hypothesis under the lens of data augmentation and regularization. We try various augmentation strategies and observe that dropout regularization can become unnecessary. Our methods achieve BLEU gains of 0.7-1.2 on TED Talks.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1912.03334](https://arxiv.org/abs/1912.03334) [cs.CL] |
|           | (or [arXiv:1912.03334v1](https://arxiv.org/abs/1912.03334v1) [cs.CL] for this version) |





<h2 id="2019-12-10-3">3. Re-Translation Strategies For Long Form, Simultaneous, Spoken Language Translation</h2>
Title: [Re-Translation Strategies For Long Form, Simultaneous, Spoken Language Translation](https://arxiv.org/abs/1912.03393)

Authors: [Naveen Arivazhagan](https://arxiv.org/search/cs?searchtype=author&query=Arivazhagan%2C+N), [Colin Cherry](https://arxiv.org/search/cs?searchtype=author&query=Cherry%2C+C), [Te I](https://arxiv.org/search/cs?searchtype=author&query=I%2C+T), [Wolfgang Macherey](https://arxiv.org/search/cs?searchtype=author&query=Macherey%2C+W), [Pallavi Baljekar](https://arxiv.org/search/cs?searchtype=author&query=Baljekar%2C+P), [George Foster](https://arxiv.org/search/cs?searchtype=author&query=Foster%2C+G)

*(Submitted on 6 Dec 2019)*

> We investigate the problem of simultaneous machine translation of long-form speech content. We target a continuous speech-to-text scenario, generating translated captions for a live audio feed, such as a lecture or play-by-play commentary. As this scenario allows for revisions to our incremental translations, we adopt a re-translation approach to simultaneous translation, where the source is repeatedly translated from scratch as it grows. This approach naturally exhibits very low latency and high final quality, but at the cost of incremental instability as the output is continuously refined. We experiment with a pipeline of industry-grade speech recognition and translation tools, augmented with simple inference heuristics to improve stability. We use TED Talks as a source of multilingual test data, developing our techniques on English-to-German spoken language translation. Our minimalist approach to simultaneous translation allows us to easily scale our final evaluation to six more target languages, dramatically improving incremental stability for all of them.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1912.03393](https://arxiv.org/abs/1912.03393) [cs.CL] |
|           | (or [arXiv:1912.03393v1](https://arxiv.org/abs/1912.03393v1) [cs.CL] for this version) |





<h2 id="2019-12-10-4">4. PidginUNMT: Unsupervised Neural Machine Translation from West African Pidgin to English</h2>
Title: [PidginUNMT: Unsupervised Neural Machine Translation from West African Pidgin to English](https://arxiv.org/abs/1912.03444)

Authors: [Kelechi Ogueji](https://arxiv.org/search/cs?searchtype=author&query=Ogueji%2C+K), [Orevaoghene Ahia](https://arxiv.org/search/cs?searchtype=author&query=Ahia%2C+O)

*(Submitted on 7 Dec 2019)*

> Over 800 languages are spoken across West Africa. Despite the obvious diversity among people who speak these languages, one language significantly unifies them all - West African Pidgin English. There are at least 80 million speakers of West African Pidgin English. However, there is no known natural language processing (NLP) work on this language. In this work, we perform the first NLP work on the most popular variant of the language, providing three major contributions. First, the provision of a Pidgin corpus of over 56000 sentences, which is the largest we know of. Secondly, the training of the first ever cross-lingual embedding between Pidgin and English. This aligned embedding will be helpful in the performance of various downstream tasks between English and Pidgin. Thirdly, the training of an Unsupervised Neural Machine Translation model between Pidgin and English which achieves BLEU scores of 7.93 from Pidgin to English, and 5.18 from English to Pidgin. In all, this work greatly reduces the barrier of entry for future NLP works on West African Pidgin English.

| Comments: | Presented at NeurIPS 2019 Workshop on Machine Learning for the Developing World |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1912.03444](https://arxiv.org/abs/1912.03444) [cs.CL] |
|           | (or [arXiv:1912.03444v1](https://arxiv.org/abs/1912.03444v1) [cs.CL] for this version) |





# 2019-12-09

[Return to Index](#Index)



<h2 id="2019-12-09-1">1. Machine Translation Evaluation Meets Community Question Answering</h2>
Title: [Machine Translation Evaluation Meets Community Question Answering](https://arxiv.org/abs/1912.02998)

Authors: [Francisco Guzmán](https://arxiv.org/search/cs?searchtype=author&query=Guzmán%2C+F), [Lluís Màrquez](https://arxiv.org/search/cs?searchtype=author&query=Màrquez%2C+L), [Preslav Nakov](https://arxiv.org/search/cs?searchtype=author&query=Nakov%2C+P)

*(Submitted on 6 Dec 2019)*

> We explore the applicability of machine translation evaluation (MTE) methods to a very different problem: answer ranking in community Question Answering. In particular, we adopt a pairwise neural network (NN) architecture, which incorporates MTE features, as well as rich syntactic and semantic embeddings, and which efficiently models complex non-linear interactions. The evaluation results show state-of-the-art performance, with sizeable contribution from both the MTE features and from the pairwise NN architecture.

| Comments:          | community question answering, machine translation evaluation, pairwise ranking, learning to rank |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**; Information Retrieval (cs.IR); Machine Learning (cs.LG) |
| MSC classes:       | 68T50                                                        |
| ACM classes:       | I.2.7                                                        |
| Journal reference: | Annual meeting of the Association for Computational Linguistics (ACL-2016) |
| Cite as:           | [arXiv:1912.02998](https://arxiv.org/abs/1912.02998) [cs.CL] |
|                    | (or [arXiv:1912.02998v1](https://arxiv.org/abs/1912.02998v1) [cs.CL] for this version) |





<h2 id="2019-12-09-2">2. Pairwise Neural Machine Translation Evaluation</h2>
Title: [Pairwise Neural Machine Translation Evaluation](https://arxiv.org/abs/1912.03135)

Authors: [Francisco Guzman](https://arxiv.org/search/cs?searchtype=author&query=Guzman%2C+F), [Shafiq Joty](https://arxiv.org/search/cs?searchtype=author&query=Joty%2C+S), [Lluis Marquez](https://arxiv.org/search/cs?searchtype=author&query=Marquez%2C+L), [Preslav Nakov](https://arxiv.org/search/cs?searchtype=author&query=Nakov%2C+P)

*(Submitted on 5 Dec 2019)*

> We present a novel framework for machine translation evaluation using neural networks in a pairwise setting, where the goal is to select the better translation from a pair of hypotheses, given the reference translation. In this framework, lexical, syntactic and semantic information from the reference and the two hypotheses is compacted into relatively small distributed vector representations, and fed into a multi-layer neural network that models the interaction between each of the hypotheses and the reference, as well as between the two hypotheses. These compact representations are in turn based on word and sentence embeddings, which are learned using neural networks. The framework is flexible, allows for efficient learning and classification, and yields correlation with humans that rivals the state of the art.

| Comments:          | machine translation evaluation, machine translation, pairwise ranking, learning to rank. arXiv admin note: substantial text overlap with [arXiv:1710.02095](https://arxiv.org/abs/1710.02095) |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**; Information Retrieval (cs.IR); Machine Learning (cs.LG) |
| MSC classes:       | 68T50                                                        |
| ACM classes:       | I.2.7                                                        |
| Journal reference: | Conference of the Association for Computational Linguistics (ACL'2015) |
| Cite as:           | [arXiv:1912.03135](https://arxiv.org/abs/1912.03135) [cs.CL] |
|                    | (or [arXiv:1912.03135v1](https://arxiv.org/abs/1912.03135v1) [cs.CL] for this version) |









# 2019-12-06

[Return to Index](#Index)



<h2 id="2019-12-06-1">1. Exploration of Neural Machine Translation in Autoformalization of Mathematics in Mizar</h2>
Title: [Exploration of Neural Machine Translation in Autoformalization of Mathematics in Mizar](https://arxiv.org/abs/1912.02636)

Authors: [Qingxiang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Q), [Chad Brown](https://arxiv.org/search/cs?searchtype=author&query=Brown%2C+C), [Cezary Kaliszyk](https://arxiv.org/search/cs?searchtype=author&query=Kaliszyk%2C+C), [Josef Urban](https://arxiv.org/search/cs?searchtype=author&query=Urban%2C+J)

*(Submitted on 5 Dec 2019)*

> In this paper we share several experiments trying to automatically translate informal mathematics into formal mathematics. In our context informal mathematics refers to human-written mathematical sentences in the LaTeX format; and formal mathematics refers to statements in the Mizar language. We conducted our experiments against three established neural network-based machine translation models that are known to deliver competitive results on translating between natural languages. To train these models we also prepared four informal-to-formal datasets. We compare and analyze our results according to whether the model is supervised or unsupervised. In order to augment the data available for auto-formalization and improve the results, we develop a custom type-elaboration mechanism and integrate it in the supervised translation.

| Comments: | Submitted to POPL/CPP'2020                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Logic in Computer Science (cs.LO)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1912.02636](https://arxiv.org/abs/1912.02636) [cs.LO] |
|           | (or [arXiv:1912.02636v1](https://arxiv.org/abs/1912.02636v1) [cs.LO] for this version) |





# 2019-12-05

[Return to Index](#Index)



<h2 id="2019-12-05-1">1. Neural Machine Translation: A Review</h2>
Title: [Neural Machine Translation: A Review](https://arxiv.org/abs/1912.02047)

Authors: [Felix Stahlberg](https://arxiv.org/search/cs?searchtype=author&query=Stahlberg%2C+F)

*(Submitted on 4 Dec 2019)*

> The field of machine translation (MT), the automatic translation of written text from one natural language into another, has experienced a major paradigm shift in recent years. Statistical MT, which mainly relies on various count-based models and which used to dominate MT research for decades, has largely been superseded by neural machine translation (NMT), which tackles translation with a single neural network. In this work we will trace back the origins of modern NMT architectures to word and sentence embeddings and earlier examples of the encoder-decoder network family. We will conclude with a survey of recent trends in the field.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1912.02047](https://arxiv.org/abs/1912.02047) [cs.CL] |
|           | (or [arXiv:1912.02047v1](https://arxiv.org/abs/1912.02047v1) [cs.CL] for this version) |





<h2 id="2019-12-05-2">2. A Robust Self-Learning Method for Fully Unsupervised Cross-Lingual Mappings of Word Embeddings: Making the Method Robustly Reproducible as Well</h2>
Title: [A Robust Self-Learning Method for Fully Unsupervised Cross-Lingual Mappings of Word Embeddings: Making the Method Robustly Reproducible as Well](https://arxiv.org/abs/1912.01706)

Authors: [Nicolas Garneau](https://arxiv.org/search/cs?searchtype=author&query=Garneau%2C+N), [Mathieu Godbout](https://arxiv.org/search/cs?searchtype=author&query=Godbout%2C+M), [David Beauchemin](https://arxiv.org/search/cs?searchtype=author&query=Beauchemin%2C+D), [Audrey Durand](https://arxiv.org/search/cs?searchtype=author&query=Durand%2C+A), [Luc Lamontagne](https://arxiv.org/search/cs?searchtype=author&query=Lamontagne%2C+L)

*(Submitted on 3 Dec 2019)*

> In this paper, we reproduce the experiments of Artetxe et al. (2018b) regarding the robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings. We show that the reproduction of their method is indeed feasible with some minor assumptions. We further investigate the robustness of their model by introducing four new languages that are less similar to English than the ones proposed by the original paper. In order to assess the stability of their model, we also conduct a grid search over sensible hyperparameters. We then propose key recommendations applicable to any research project in order to deliver fully reproducible research.

| Comments: | Submitted to REPROLANG@LREC2020                              |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Machine Learning (stat.ML) |
| Cite as:  | [arXiv:1912.01706](https://arxiv.org/abs/1912.01706) [cs.LG] |
|           | (or [arXiv:1912.01706v1](https://arxiv.org/abs/1912.01706v1) [cs.LG] for this version) |





<h2 id="2019-12-05-3">3. Acquiring Knowledge from Pre-trained Model to Neural Machine Translation</h2>
Title: [Acquiring Knowledge from Pre-trained Model to Neural Machine Translation](https://arxiv.org/abs/1912.01774)

Authors: [Rongxiang Weng](https://arxiv.org/search/cs?searchtype=author&query=Weng%2C+R), [Heng Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+H), [Shujian Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Shanbo Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng%2C+S), [Weihua Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+W)

*(Submitted on 4 Dec 2019)*

> Pre-training and fine-tuning have achieved great success in the natural language process field. The standard paradigm of exploiting them includes two steps: first, pre-training a model, e.g. BERT, with a large scale unlabeled monolingual data. Then, fine-tuning the pre-trained model with labeled data from downstream tasks. However, in neural machine translation (NMT), we address the problem that the training objective of the bilingual task is far different from the monolingual pre-trained model. This gap leads that only using fine-tuning in NMT can not fully utilize prior language knowledge. In this paper, we propose an APT framework for acquiring knowledge from the pre-trained model to NMT. The proposed approach includes two modules: 1). a dynamic fusion mechanism to fuse task-specific features adapted from general knowledge into NMT network, 2). a knowledge distillation paradigm to learn language knowledge continuously during the NMT training process. The proposed approach could integrate suitable knowledge from pre-trained models to improve the NMT. Experimental results on WMT English to German, German to English and Chinese to English machine translation tasks show that our model outperforms strong baselines and the fine-tuning counterparts.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1912.01774](https://arxiv.org/abs/1912.01774) [cs.CL] |
|           | (or [arXiv:1912.01774v1](https://arxiv.org/abs/1912.01774v1) [cs.CL] for this version) |







# 2019-12-04

[Return to Index](#Index)



<h2 id="2019-12-04-1">1. Cross-lingual Pre-training Based Transfer for Zero-shot Neural Machine Translation</h2>
Title: [Cross-lingual Pre-training Based Transfer for Zero-shot Neural Machine Translation](https://arxiv.org/abs/1912.01214)

Authors: [Baijun Ji](https://arxiv.org/search/cs?searchtype=author&query=Ji%2C+B), [Zhirui Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Xiangyu Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan%2C+X), [Min Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M), [Boxing Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+B), [Weihua Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+W)

*(Submitted on 3 Dec 2019)*

> Transfer learning between different language pairs has shown its effectiveness for Neural Machine Translation (NMT) in low-resource scenario. However, existing transfer methods involving a common target language are far from success in the extreme scenario of zero-shot translation, due to the language space mismatch problem between transferor (the parent model) and transferee (the child model) on the source side. To address this challenge, we propose an effective transfer learning approach based on cross-lingual pre-training. Our key idea is to make all source languages share the same feature space and thus enable a smooth transition for zero-shot translation. To this end, we introduce one monolingual pre-training method and two bilingual pre-training methods to obtain a universal encoder for different languages. Once the universal encoder is constructed, the parent model built on such encoder is trained with large-scale annotated data and then directly applied in zero-shot translation scenario. Experiments on two public datasets show that our approach significantly outperforms strong pivot-based baseline and various multilingual NMT approaches.

| Comments: | Accepted as a conference paper at AAAI 2020 (oral presentation) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1912.01214](https://arxiv.org/abs/1912.01214) [cs.CL] |
|           | (or [arXiv:1912.01214v1](https://arxiv.org/abs/1912.01214v1) [cs.CL] for this version) |





# 2019-12-03

[Return to Index](#Index)



<h2 id="2019-12-03-1">1. Not All Attention Is Needed: Gated Attention Network for Sequence Data</h2>
Title: [Not All Attention Is Needed: Gated Attention Network for Sequence Data](https://arxiv.org/abs/1912.00349)

Authors: [Lanqing Xue](https://arxiv.org/search/cs?searchtype=author&query=Xue%2C+L), [Xiaopeng Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+X), [Nevin L. Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+N+L)

*(Submitted on 1 Dec 2019)*

> Although deep neural networks generally have fixed network structures, the concept of dynamic mechanism has drawn more and more attention in recent years. Attention mechanisms compute input-dependent dynamic attention weights for aggregating a sequence of hidden states. Dynamic network configuration in convolutional neural networks (CNNs) selectively activates only part of the network at a time for different inputs. In this paper, we combine the two dynamic mechanisms for text classification tasks. Traditional attention mechanisms attend to the whole sequence of hidden states for an input sentence, while in most cases not all attention is needed especially for long sequences. We propose a novel method called Gated Attention Network (GA-Net) to dynamically select a subset of elements to attend to using an auxiliary network, and compute attention weights to aggregate the selected elements. It avoids a significant amount of unnecessary computation on unattended elements, and allows the model to pay attention to important parts of the sequence. Experiments in various datasets show that the proposed method achieves better performance compared with all baseline models with global or local attention while requiring less computation and achieving better interpretability. It is also promising to extend the idea to more complex attention-based models, such as transformers and seq-to-seq models.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1912.00349](https://arxiv.org/abs/1912.00349) [cs.LG] |
|           | (or [arXiv:1912.00349v1](https://arxiv.org/abs/1912.00349v1) [cs.LG] for this version) |





<h2 id="2019-12-03-2">2. Modeling Fluency and Faithfulness for Diverse Neural Machine Translation</h2>
Title: [Modeling Fluency and Faithfulness for Diverse Neural Machine Translation](https://arxiv.org/abs/1912.00178)

Authors: [Yang Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+Y), [Wanying Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+W), [Shuhao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+S), [Chenze Shao](https://arxiv.org/search/cs?searchtype=author&query=Shao%2C+C), [Wen Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+W), [Zhengxin Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Dong Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+D)

*(Submitted on 30 Nov 2019)*

> Neural machine translation models usually adopt the teacher forcing strategy for training which requires the predicted sequence matches ground truth word by word and forces the probability of each prediction to approach a 0-1 distribution. However, the strategy casts all the portion of the distribution to the ground truth word and ignores other words in the target vocabulary even when the ground truth word cannot dominate the distribution. To address the problem of teacher forcing, we propose a method to introduce an evaluation module to guide the distribution of the prediction. The evaluation module accesses each prediction from the perspectives of fluency and faithfulness to encourage the model to generate the word which has a fluent connection with its past and future translation and meanwhile tends to form a translation equivalent in meaning to the source. The experiments on multiple translation tasks show that our method can achieve significant improvements over strong baselines.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1912.00178](https://arxiv.org/abs/1912.00178) [cs.CL] |
|           | (or [arXiv:1912.00178v1](https://arxiv.org/abs/1912.00178v1) [cs.CL] for this version) |



<h2 id="2019-12-03-3">3. Merging External Bilingual Pairs into Neural Machine Translation</h2>
Title: [Merging External Bilingual Pairs into Neural Machine Translation](https://arxiv.org/abs/1912.00567)

Authors: [Tao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+T), [Shaohui Kuang](https://arxiv.org/search/cs?searchtype=author&query=Kuang%2C+S), [Deyi Xiong](https://arxiv.org/search/cs?searchtype=author&query=Xiong%2C+D), [António Branco](https://arxiv.org/search/cs?searchtype=author&query=Branco%2C+A)

*(Submitted on 2 Dec 2019)*

> As neural machine translation (NMT) is not easily amenable to explicit correction of errors, incorporating pre-specified translations into NMT is widely regarded as a non-trivial challenge. In this paper, we propose and explore three methods to endow NMT with pre-specified bilingual pairs. Instead, for instance, of modifying the beam search algorithm during decoding or making complex modifications to the attention mechanism --- mainstream approaches to tackling this challenge ---, we experiment with the training data being appropriately pre-processed to add information about pre-specified translations. Extra embeddings are also used to distinguish pre-specified tokens from the other tokens. Extensive experimentation and analysis indicate that over 99% of the pre-specified phrases are successfully translated (given a 85% baseline) and that there is also a substantive improvement in translation quality with the methods explored here.

| Comments:    | 7 pages, 3 figures, 5 tables                                 |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**                         |
| MSC classes: | 68T50                                                        |
| ACM classes: | I.2.7                                                        |
| Cite as:     | [arXiv:1912.00567](https://arxiv.org/abs/1912.00567) [cs.CL] |
|              | (or [arXiv:1912.00567v1](https://arxiv.org/abs/1912.00567v1) [cs.CL] for this version) |





# 2019-12-02

[Return to Index](#Index)



<h2 id="2019-12-02-1">1. DiscoTK: Using Discourse Structure for Machine Translation Evaluation</h2>
Title: [DiscoTK: Using Discourse Structure for Machine Translation Evaluation](https://arxiv.org/abs/1911.12547)

Authors: [Shafiq Joty](https://arxiv.org/search/cs?searchtype=author&query=Joty%2C+S), [Francisco Guzman](https://arxiv.org/search/cs?searchtype=author&query=Guzman%2C+F), [Lluis Marquez](https://arxiv.org/search/cs?searchtype=author&query=Marquez%2C+L), [Preslav Nakov](https://arxiv.org/search/cs?searchtype=author&query=Nakov%2C+P)

*(Submitted on 28 Nov 2019)*

> We present novel automatic metrics for machine translation evaluation that use discourse structure and convolution kernels to compare the discourse tree of an automatic translation with that of the human reference. We experiment with five transformations and augmentations of a base discourse tree representation based on the rhetorical structure theory, and we combine the kernel scores for each of them into a single score. Finally, we add other metrics from the ASIYA MT evaluation toolkit, and we tune the weights of the combination on actual human judgments. Experiments on the WMT12 and WMT13 metrics shared task datasets show correlation with human judgments that outperforms what the best systems that participated in these years achieved, both at the segment and at the system level.

| Comments:          | machine translation evaluation, machine translation, tree kernels, discourse, convolutional kernels, discourse tree, RST, rhetorical structure theory, ASIYA |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| MSC classes:       | 68T50                                                        |
| ACM classes:       | I.2.7                                                        |
| Journal reference: | WMT-2014                                                     |
| Cite as:           | [arXiv:1911.12547](https://arxiv.org/abs/1911.12547) [cs.CL] |
|                    | (or [arXiv:1911.12547v1](https://arxiv.org/abs/1911.12547v1) [cs.CL] for this version) |





<h2 id="2019-12-02-2">2. Multimodal Machine Translation through Visuals and Speech</h2>
Title: [Multimodal Machine Translation through Visuals and Speech](https://arxiv.org/abs/1911.12798)

Authors: [Umut Sulubacak](https://arxiv.org/search/cs?searchtype=author&query=Sulubacak%2C+U), [Ozan Caglayan](https://arxiv.org/search/cs?searchtype=author&query=Caglayan%2C+O), [Stig-Arne Grönroos](https://arxiv.org/search/cs?searchtype=author&query=Grönroos%2C+S), [Aku Rouhe](https://arxiv.org/search/cs?searchtype=author&query=Rouhe%2C+A), [Desmond Elliott](https://arxiv.org/search/cs?searchtype=author&query=Elliott%2C+D), [Lucia Specia](https://arxiv.org/search/cs?searchtype=author&query=Specia%2C+L), [Jörg Tiedemann](https://arxiv.org/search/cs?searchtype=author&query=Tiedemann%2C+J)

*(Submitted on 28 Nov 2019)*

> Multimodal machine translation involves drawing information from more than one modality, based on the assumption that the additional modalities will contain useful alternative views of the input data. The most prominent tasks in this area are spoken language translation, image-guided translation, and video-guided translation, which exploit audio and visual modalities, respectively. These tasks are distinguished from their monolingual counterparts of speech recognition, image captioning, and video captioning by the requirement of models to generate outputs in a different language. This survey reviews the major data resources for these tasks, the evaluation campaigns concentrated around them, the state of the art in end-to-end and pipeline approaches, and also the challenges in performance evaluation. The paper concludes with a discussion of directions for future research in these areas: the need for more expansive and challenging datasets, for targeted evaluations of model performance, and for multimodality in both the input and output space.

| Comments: | 34 pages, 4 tables, 8 figures. Submitted (Nov 2019) to the Machine Translation journal (Springer) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1911.12798](https://arxiv.org/abs/1911.12798) [cs.CL] |
|           | (or [arXiv:1911.12798v1](https://arxiv.org/abs/1911.12798v1) [cs.CL] for this version) |





<h2 id="2019-12-02-3">3. GitHub Typo Corpus: A Large-Scale Multilingual Dataset of Misspellings and Grammatical Errors</h2>
Title: [GitHub Typo Corpus: A Large-Scale Multilingual Dataset of Misspellings and Grammatical Errors](https://arxiv.org/abs/1911.12893)

Authors: [Masato Hagiwara](https://arxiv.org/search/cs?searchtype=author&query=Hagiwara%2C+M), [Masato Mita](https://arxiv.org/search/cs?searchtype=author&query=Mita%2C+M)

*(Submitted on 28 Nov 2019)*

> The lack of large-scale datasets has been a major hindrance to the development of NLP tasks such as spelling correction and grammatical error correction (GEC). As a complementary new resource for these tasks, we present the GitHub Typo Corpus, a large-scale, multilingual dataset of misspellings and grammatical errors along with their corrections harvested from GitHub, a large and popular platform for hosting and sharing git repositories. The dataset, which we have made publicly available, contains more than 350k edits and 65M characters in more than 15 languages, making it the largest dataset of misspellings to date. We also describe our process for filtering true typo edits based on learned classifiers on a small annotated subset, and demonstrate that typo edits can be identified with F1 ~ 0.9 using a very simple classifier with only three features. The detailed analyses of the dataset show that existing spelling correctors merely achieve an F-measure of approx. 0.5, suggesting that the dataset serves as a new, rich source of spelling errors that complement existing datasets.

| Comments: | Submitted at LREC 2020                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1911.12893](https://arxiv.org/abs/1911.12893) [cs.CL] |
|           | (or [arXiv:1911.12893v1](https://arxiv.org/abs/1911.12893v1) [cs.CL] for this version) |





<h2 id="2019-12-02-4">4. Neural Chinese Word Segmentation as Sequence to Sequence Translation</h2>
Title: [Neural Chinese Word Segmentation as Sequence to Sequence Translation](https://arxiv.org/abs/1911.12982)

Authors: [Xuewen Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+X), [Heyan Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+H), [Ping Jian](https://arxiv.org/search/cs?searchtype=author&query=Jian%2C+P), [Yuhang Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+Y), [Xiaochi Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+X), [Yi-Kun Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+Y)

*(Submitted on 29 Nov 2019)*

> Recently, Chinese word segmentation (CWS) methods using neural networks have made impressive progress. Most of them regard the CWS as a sequence labeling problem which construct models based on local features rather than considering global information of input sequence. In this paper, we cast the CWS as a sequence translation problem and propose a novel sequence-to-sequence CWS model with an attention-based encoder-decoder framework. The model captures the global information from the input and directly outputs the segmented sequence. It can also tackle other NLP tasks with CWS jointly in an end-to-end mode. Experiments on Weibo, PKU and MSRA benchmark datasets show that our approach has achieved competitive performances compared with state-of-the-art methods. Meanwhile, we successfully applied our proposed model to jointly learning CWS and Chinese spelling correction, which demonstrates its applicability of multi-task fusion.

| Comments: | In proceedings of SMP 2017 (Chinese National Conference on Social Media Processing) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| DOI:      | [10.1007/978-981-10-6805-8_8](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1007%2F978-981-10-6805-8_8&v=a5b3c37f) |
| Cite as:  | [arXiv:1911.12982](https://arxiv.org/abs/1911.12982) [cs.CL] |
|           | (or [arXiv:1911.12982v1](https://arxiv.org/abs/1911.12982v1) [cs.CL] for this version) |







