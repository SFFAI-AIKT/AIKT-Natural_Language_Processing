# Daily arXiv: Machine Translation - April, 2021

# Index

- [2021-04-13](#2021-04-13)
  - [1. Achieving Model Robustness through Discrete Adversarial Training](#2021-04-13-1)
  - [2. TransWiC at SemEval-2021 Task 2: Transformer-based Multilingual and Cross-lingual Word-in-Context Disambiguation](#2021-04-13-2)
  - [3. Not All Attention Is All You Need](#2021-04-13-3)
  - [4. Sentiment-based Candidate Selection for NMT](#2021-04-13-4)
  - [5. Disentangled Contrastive Learning for Learning Robust Textual Representations](#2021-04-13-5)
  - [6. Assessing Reference-Free Peer Evaluation for Machine Translation](#2021-04-13-6)
  - [7. FUDGE: Controlled Text Generation With Future Discriminators](#2021-04-13-7)
  - [8. Machine Translation Decoding beyond Beam Search](#2021-04-13-8)
  - [9. Self-Training with Weak Supervision](#2021-04-13-9)
  - [10. Survey on reinforcement learning for language processing](#2021-04-13-10)
  - [11. Backtranslation Feedback Improves User Confidence in MT, Not Quality](#2021-04-13-11)
- [2021-04-12](#2021-04-12)
  - [1. Video-aided Unsupervised Grammar Induction](#2021-04-12-1)
  - [2. Design and Implementation of English To Yoruba Verb Phrase Machine Translation System](#2021-04-12-2)
  - [3. Efficient Large-Scale Language Model Training on GPU Clusters](#2021-04-12-3)
  - [4. Chinese Character Decomposition for Neural MT with Multi-Word Expressions](#2021-04-12-4)
- [2021-04-09](#2021-04-09)
  - [1. Extended Parallel Corpus for Amharic-English Machine Translation](#2021-04-09-1)
  - [2. BSTC: A Large-Scale Chinese-English Speech Translation Dataset](#2021-04-09-2)
  - [3. A Simple Geometric Method for Cross-Lingual Linguistic Transformations with Pre-trained Autoencoders](#2021-04-09-3)
  - [4. Probing BERT in Hyperbolic Spaces](#2021-04-09-4)
- [2021-04-08](#2021-04-08)
  - [1. VERB: Visualizing and Interpreting Bias Mitigation Techniques for Word Representations](#2021-04-08-1)
  - [2. Better Neural Machine Translation by Extracting Linguistic Information from BERT](#2021-04-08-2)
  - [3. GrammarTagger: A Multilingual, Minimally-Supervised Grammar Profiler for Language Education](#2021-04-08-3)
- [2021-04-07](#2021-04-07)
  - [1. Semantic Distance: A New Metric for ASR Performance Analysis Towards Spoken Language Understanding](#2021-04-07-1)
  - [2. ODE Transformer: An Ordinary Differential Equation-Inspired Model for Neural Machine Translation](#2021-04-07-2)
- [2021-04-06](#2021-04-06)
  - [1. TSNAT: Two-Step Non-Autoregressvie Transformer Models for Speech Recognition](#2021-04-06-1)
  - [2. Attention Forcing for Machine Translation](#2021-04-06-2)
  - [3. WhiteningBERT: An Easy Unsupervised Sentence Embedding Approach](#2021-04-06-3)
  - [4. Rethinking Perturbations in Encoder-Decoders for Fast Training](#2021-04-06-4)
- [2021-04-02](#2021-04-02)
  - [1. Is Label Smoothing Truly Incompatible with Knowledge Distillation: An Empirical Study](#2021-04-02-1)
	- [2. Domain-specific MT for Low-resource Languages: The case of Bambara-French](#2021-04-02-2)
  - [3. Zero-Shot Language Transfer vs Iterative Back Translation for Unsupervised Machine Translation](#2021-04-02-3)
  - [4. Detecting over/under-translation errors for determining adequacy in human translations](#2021-04-02-4)
  - [5. Many-to-English Machine Translation Tools, Data, and Pretrained Models](#2021-04-02-5)
  - [6. Low-Resource Neural Machine Translation for South-Eastern African Languages](#2021-04-02-6)
  - [7. WakaVT: A Sequential Variational Transformer for Waka Generation](#2021-04-02-7)
  - [8. Sampling and Filtering of Neural Machine Translation Distillation Data](#2021-04-02-8)
- [2021-04-01](#2021-04-01)
  - [1. An Exploration of Data Augmentation Techniques for Improving English to Tigrinya Translation](#2021-04-01-1)
	- [2. Few-shot learning through contextual data augmentation](#2021-04-01-2)
  - [3. UA-GEC: Grammatical Error Correction and Fluency Corpus for the Ukrainian Language](#2021-04-01-3)
  - [4. Divide and Rule: Training Context-Aware Multi-Encoder Translation Models with Little Resources](#2021-04-01-4)
  - [5. Leveraging Neural Machine Translation for Word Alignment](#2021-04-01-5)
- [2021-03-31](#2021-03-31)	
  - [1. Diagnosing Vision-and-Language Navigation: What Really Matters](#2021-03-31-1)
- [Other Columns](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-04-13

[Return to Index](#Index)



<h2 id="2021-04-13-1">1. Achieving Model Robustness through Discrete Adversarial Training
</h2>

Title: [Achieving Model Robustness through Discrete Adversarial Training](https://arxiv.org/abs/2104.05062)

Authors: [Maor Ivgi](https://arxiv.org/search/cs?searchtype=author&query=Ivgi%2C+M), [Jonathan Berant](https://arxiv.org/search/cs?searchtype=author&query=Berant%2C+J)

> Discrete adversarial attacks are symbolic perturbations to a language input that preserve the output label but lead to a prediction error. While such attacks have been extensively explored for the purpose of evaluating model robustness, their utility for improving robustness has been limited to offline augmentation only, i.e., given a trained model, attacks are used to generate perturbed (adversarial) examples, and the model is re-trained exactly once. In this work, we address this gap and leverage discrete attacks for online augmentation, where adversarial examples are generated at every step, adapting to the changing nature of the model. We also consider efficient attacks based on random sampling, that unlike prior work are not based on expensive search-based procedures. As a second contribution, we provide a general formulation for multiple search-based attacks from past work, and propose a new attack based on best-first search. Surprisingly, we find that random sampling leads to impressive gains in robustness, outperforming the commonly-used offline augmentation, while leading to a speedup at training time of ~10x. Furthermore, online augmentation with search-based attacks justifies the higher training cost, significantly improving robustness on three datasets. Last, we show that our proposed algorithm substantially improves robustness compared to prior methods.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2104.05062](https://arxiv.org/abs/2104.05062) [cs.LG]** |
|           | (or **[arXiv:2104.05062v1](https://arxiv.org/abs/2104.05062v1) [cs.LG]** for this version) |





<h2 id="2021-04-13-2">2. TransWiC at SemEval-2021 Task 2: Transformer-based Multilingual and Cross-lingual Word-in-Context Disambiguation
</h2>

Title: [TransWiC at SemEval-2021 Task 2: Transformer-based Multilingual and Cross-lingual Word-in-Context Disambiguation](https://arxiv.org/abs/2104.04632)

Authors: [Hansi Hettiarachchi](https://arxiv.org/search/cs?searchtype=author&query=Hettiarachchi%2C+H), [Tharindu Ranasinghe](https://arxiv.org/search/cs?searchtype=author&query=Ranasinghe%2C+T)

> Identifying whether a word carries the same meaning or different meaning in two contexts is an important research area in natural language processing which plays a significant role in many applications such as question answering, document summarisation, information retrieval and information extraction. Most of the previous work in this area rely on language-specific resources making it difficult to generalise across languages. Considering this limitation, our approach to SemEval-2021 Task 2 is based only on pretrained transformer models and does not use any language-specific processing and resources. Despite that, our best model achieves 0.90 accuracy for English-English subtask which is very compatible compared to the best result of the subtask; 0.93 accuracy. Our approach also achieves satisfactory results in other monolingual and cross-lingual language pairs as well.

| Comments: | Accepted to SemEval-2021                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2104.04632](https://arxiv.org/abs/2104.04632) [cs.CL]** |
|           | (or **[arXiv:2104.04632v1](https://arxiv.org/abs/2104.04632v1) [cs.CL]** for this version) |







<h2 id="2021-04-13-3">3. Not All Attention Is All You Need
</h2>

Title: [Not All Attention Is All You Need](https://arxiv.org/abs/2104.04692)

Authors: [Hongqiu Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+H), [Hai Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+H), [Min Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M)

> Self-attention based models have achieved remarkable success in natural language processing. However, the self-attention network design is questioned as suboptimal in recent studies, due to its veiled validity and high redundancy. In this paper, we focus on pre-trained language models with self-pruning training design on task-specific tuning. We demonstrate that the lighter state-of-the-art models with nearly 80% of self-attention layers pruned, may achieve even better results on multiple tasks, including natural language understanding, document classification, named entity recognition and POS tagging, with nearly twice faster inference.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2104.04692](https://arxiv.org/abs/2104.04692) [cs.CL]** |
|           | (or **[arXiv:2104.04692v1](https://arxiv.org/abs/2104.04692v1) [cs.CL]** for this version) |







<h2 id="2021-04-13-4">4. Sentiment-based Candidate Selection for NMT
</h2>

Title: [Sentiment-based Candidate Selection for NMT](https://arxiv.org/abs/2104.04840)

Authors: [Alex Jones](https://arxiv.org/search/cs?searchtype=author&query=Jones%2C+A), [Derry Tanti Wijaya](https://arxiv.org/search/cs?searchtype=author&query=Wijaya%2C+D+T)

> The explosion of user-generated content (UGC)--e.g. social media posts, comments, and reviews--has motivated the development of NLP applications tailored to these types of informal texts. Prevalent among these applications have been sentiment analysis and machine translation (MT). Grounded in the observation that UGC features highly idiomatic, sentiment-charged language, we propose a decoder-side approach that incorporates automatic sentiment scoring into the MT candidate selection process. We train separate English and Spanish sentiment classifiers, then, using n-best candidates generated by a baseline MT model with beam search, select the candidate that minimizes the absolute difference between the sentiment score of the source sentence and that of the translation, and perform a human evaluation to assess the produced translations. Unlike previous work, we select this minimally divergent translation by considering the sentiment scores of the source sentence and translation on a continuous interval, rather than using e.g. binary classification, allowing for more fine-grained selection of translation candidates. The results of human evaluations show that, in comparison to the open-source MT baseline model on top of which our sentiment-based pipeline is built, our pipeline produces more accurate translations of colloquial, sentiment-heavy source texts.

| Comments:    | 14 pages, 1 figure                                           |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| ACM classes: | I.2.7                                                        |
| Cite as:     | **[arXiv:2104.04840](https://arxiv.org/abs/2104.04840) [cs.CL]** |
|              | (or **[arXiv:2104.04840v1](https://arxiv.org/abs/2104.04840v1) [cs.CL]** for this version) |







<h2 id="2021-04-13-5">5. Disentangled Contrastive Learning for Learning Robust Textual Representations
</h2>

Title: [Disentangled Contrastive Learning for Learning Robust Textual Representations](https://arxiv.org/abs/2104.04907)

Authors: [Xiang Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+X), [Xin Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+X), [Zhen Bi](https://arxiv.org/search/cs?searchtype=author&query=Bi%2C+Z), [Hongbin Ye](https://arxiv.org/search/cs?searchtype=author&query=Ye%2C+H), [Shumin Deng](https://arxiv.org/search/cs?searchtype=author&query=Deng%2C+S), [Ningyu Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+N), [Huajun Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+H)

> Although the self-supervised pre-training of transformer models has resulted in the revolutionizing of natural language processing (NLP) applications and the achievement of state-of-the-art results with regard to various benchmarks, this process is still vulnerable to small and imperceptible permutations originating from legitimate inputs. Intuitively, the representations should be similar in the feature space with subtle input permutations, while large variations occur with different meanings. This motivates us to investigate the learning of robust textual representation in a contrastive manner. However, it is non-trivial to obtain opposing semantic instances for textual samples. In this study, we propose a disentangled contrastive learning method that separately optimizes the uniformity and alignment of representations without negative sampling. Specifically, we introduce the concept of momentum representation consistency to align features and leverage power normalization while conforming the uniformity. Our experimental results for the NLP benchmarks demonstrate that our approach can obtain better results compared with the baselines, as well as achieve promising improvements with invariance tests and adversarial attacks. The code is available in [this https URL](https://github.com/zjunlp/DCL).

| Comments: | Work in progress                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2104.04907](https://arxiv.org/abs/2104.04907) [cs.CL]** |
|           | (or **[arXiv:2104.04907v1](https://arxiv.org/abs/2104.04907v1) [cs.CL]** for this version) |







<h2 id="2021-04-13-6">6. Assessing Reference-Free Peer Evaluation for Machine Translation
</h2>

Title: [Assessing Reference-Free Peer Evaluation for Machine Translation](https://arxiv.org/abs/2104.05146)

Authors: [Sweta Agrawal](https://arxiv.org/search/cs?searchtype=author&query=Agrawal%2C+S), [George Foster](https://arxiv.org/search/cs?searchtype=author&query=Foster%2C+G), [Markus Freitag](https://arxiv.org/search/cs?searchtype=author&query=Freitag%2C+M), [Colin Cherry](https://arxiv.org/search/cs?searchtype=author&query=Cherry%2C+C)

> Reference-free evaluation has the potential to make machine translation evaluation substantially more scalable, allowing us to pivot easily to new languages or domains. It has been recently shown that the probabilities given by a large, multilingual model can achieve state of the art results when used as a reference-free metric. We experiment with various modifications to this model and demonstrate that by scaling it up we can match the performance of BLEU. We analyze various potential weaknesses of the approach and find that it is surprisingly robust and likely to offer reasonable performance across a broad spectrum of domains and different system qualities.

| Comments: | NAACL 2021                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2104.05146](https://arxiv.org/abs/2104.05146) [cs.CL]** |
|           | (or **[arXiv:2104.05146v1](https://arxiv.org/abs/2104.05146v1) [cs.CL]** for this version) |







<h2 id="2021-04-13-7">7. FUDGE: Controlled Text Generation With Future Discriminators
</h2>

Title: [FUDGE: Controlled Text Generation With Future Discriminators](https://arxiv.org/abs/2104.05218)

Authors: [Kevin Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+K), [Dan Klein](https://arxiv.org/search/cs?searchtype=author&query=Klein%2C+D)

> We propose Future Discriminators for Generation (FUDGE), a flexible and modular method for controlled text generation. Given a pre-existing model G for generating text from a distribution of interest, FUDGE enables conditioning on a desired attribute a (for example, formality) while requiring access only to G's output logits. FUDGE learns an attribute predictor operating on a partial sequence, and uses this predictor's outputs to adjust G's original probabilities. We show that FUDGE models terms corresponding to a Bayesian decomposition of the conditional distribution of G given attribute a. Moreover, FUDGE can easily compose predictors for multiple desired attributes. We evaluate FUDGE on three tasks -- couplet completion in poetry, topic control in language generation, and formality change in machine translation -- and observe gains in all three tasks.

| Comments: | To appear at NAACL 2021                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2104.05218](https://arxiv.org/abs/2104.05218) [cs.CL]** |
|           | (or **[arXiv:2104.05218v1](https://arxiv.org/abs/2104.05218v1) [cs.CL]** for this version) |







<h2 id="2021-04-13-8">8. Machine Translation Decoding beyond Beam Search
</h2>

Title: [Machine Translation Decoding beyond Beam Search](https://arxiv.org/abs/2104.05336)

Authors: [Rémi Leblond](https://arxiv.org/search/cs?searchtype=author&query=Leblond%2C+R), [Jean-Baptiste Alayrac](https://arxiv.org/search/cs?searchtype=author&query=Alayrac%2C+J), [Laurent Sifre](https://arxiv.org/search/cs?searchtype=author&query=Sifre%2C+L), [Miruna Pislar](https://arxiv.org/search/cs?searchtype=author&query=Pislar%2C+M), [Jean-Baptiste Lespiau](https://arxiv.org/search/cs?searchtype=author&query=Lespiau%2C+J), [Ioannis Antonoglou](https://arxiv.org/search/cs?searchtype=author&query=Antonoglou%2C+I), [Karen Simonyan](https://arxiv.org/search/cs?searchtype=author&query=Simonyan%2C+K), [Oriol Vinyals](https://arxiv.org/search/cs?searchtype=author&query=Vinyals%2C+O)

> Beam search is the go-to method for decoding auto-regressive machine translation models. While it yields consistent improvements in terms of BLEU, it is only concerned with finding outputs with high model likelihood, and is thus agnostic to whatever end metric or score practitioners care about. Our aim is to establish whether beam search can be replaced by a more powerful metric-driven search technique. To this end, we explore numerous decoding algorithms, including some which rely on a value function parameterised by a neural network, and report results on a variety of metrics. Notably, we introduce a Monte-Carlo Tree Search (MCTS) based method and showcase its competitiveness. We provide a blueprint for how to use MCTS fruitfully in language applications, which opens promising future directions. We find that which algorithm is best heavily depends on the characteristics of the goal metric; we believe that our extensive experiments and analysis will inform further research in this area.

| Comments: | 23 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2104.05336](https://arxiv.org/abs/2104.05336) [cs.CL]** |
|           | (or **[arXiv:2104.05336v1](https://arxiv.org/abs/2104.05336v1) [cs.CL]** for this version) |







<h2 id="2021-04-13-9">9. Self-Training with Weak Supervision
</h2>

Title: [Self-Training with Weak Supervision](https://arxiv.org/abs/2104.05514)

Authors: [Giannis Karamanolakis](https://arxiv.org/search/cs?searchtype=author&query=Karamanolakis%2C+G), [Subhabrata Mukherjee](https://arxiv.org/search/cs?searchtype=author&query=Mukherjee%2C+S), [Guoqing Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+G), [Ahmed Hassan Awadallah](https://arxiv.org/search/cs?searchtype=author&query=Awadallah%2C+A+H)

> State-of-the-art deep neural networks require large-scale labeled training data that is often expensive to obtain or not available for many tasks. Weak supervision in the form of domain-specific rules has been shown to be useful in such settings to automatically generate weakly labeled training data. However, learning with weak rules is challenging due to their inherent heuristic and noisy nature. An additional challenge is rule coverage and overlap, where prior work on weak supervision only considers instances that are covered by weak rules, thus leaving valuable unlabeled data behind.
> In this work, we develop a weak supervision framework (ASTRA) that leverages all the available data for a given task. To this end, we leverage task-specific unlabeled data through self-training with a model (student) that considers contextualized representations and predicts pseudo-labels for instances that may not be covered by weak rules. We further develop a rule attention network (teacher) that learns how to aggregate student pseudo-labels with weak rule labels, conditioned on their fidelity and the underlying context of an instance. Finally, we construct a semi-supervised learning objective for end-to-end training with unlabeled data, domain-specific rules, and a small amount of labeled data. Extensive experiments on six benchmark datasets for text classification demonstrate the effectiveness of our approach with significant improvements over state-of-the-art baselines.

| Comments: | Accepted to NAACL 2021 (Long Paper)                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (stat.ML) |
| Cite as:  | **[arXiv:2104.05514](https://arxiv.org/abs/2104.05514) [cs.CL]** |
|           | (or **[arXiv:2104.05514v1](https://arxiv.org/abs/2104.05514v1) [cs.CL]** for this version) |







<h2 id="2021-04-13-10">10. Survey on reinforcement learning for language processing
</h2>

Title: [Survey on reinforcement learning for language processing](https://arxiv.org/abs/2104.05565)

Authors: [Victor Uc-Cetina](https://arxiv.org/search/cs?searchtype=author&query=Uc-Cetina%2C+V), [Nicolas Navarro-Guerrero](https://arxiv.org/search/cs?searchtype=author&query=Navarro-Guerrero%2C+N), [Anabel Martin-Gonzalez](https://arxiv.org/search/cs?searchtype=author&query=Martin-Gonzalez%2C+A), [Cornelius Weber](https://arxiv.org/search/cs?searchtype=author&query=Weber%2C+C), [Stefan Wermter](https://arxiv.org/search/cs?searchtype=author&query=Wermter%2C+S)

> In recent years some researchers have explored the use of reinforcement learning (RL) algorithms as key components in the solution of various natural language processing tasks. For instance, some of these algorithms leveraging deep neural learning have found their way into conversational systems. This paper reviews the state of the art of RL methods for their possible use for different problems of natural language processing, focusing primarily on conversational systems, mainly due to their growing relevance. We provide detailed descriptions of the problems as well as discussions of why RL is well-suited to solve them. Also, we analyze the advantages and limitations of these methods. Finally, we elaborate on promising research directions in natural language processing that might benefit from reinforcement learning.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2104.05565](https://arxiv.org/abs/2104.05565) [cs.CL]** |
|           | (or **[arXiv:2104.05565v1](https://arxiv.org/abs/2104.05565v1) [cs.CL]** for this version) |







<h2 id="2021-04-13-11">11. Backtranslation Feedback Improves User Confidence in MT, Not Quality
</h2>

Title: [Backtranslation Feedback Improves User Confidence in MT, Not Quality](https://arxiv.org/abs/2104.05688)

Authors: [Vilém Zouhar](https://arxiv.org/search/cs?searchtype=author&query=Zouhar%2C+V), [Michal Novák](https://arxiv.org/search/cs?searchtype=author&query=Novák%2C+M), [Matúš Žilinec](https://arxiv.org/search/cs?searchtype=author&query=Žilinec%2C+M), [Ondřej Bojar](https://arxiv.org/search/cs?searchtype=author&query=Bojar%2C+O), [Mateo Obregón](https://arxiv.org/search/cs?searchtype=author&query=Obregón%2C+M), [Robin L. Hill](https://arxiv.org/search/cs?searchtype=author&query=Hill%2C+R+L), [Frédéric Blain](https://arxiv.org/search/cs?searchtype=author&query=Blain%2C+F), [Marina Fomicheva](https://arxiv.org/search/cs?searchtype=author&query=Fomicheva%2C+M), [Lucia Specia](https://arxiv.org/search/cs?searchtype=author&query=Specia%2C+L), [Lisa Yankovskaya](https://arxiv.org/search/cs?searchtype=author&query=Yankovskaya%2C+L)

> Translating text into a language unknown to the text's author, dubbed outbound translation, is a modern need for which the user experience has significant room for improvement, beyond the basic machine translation facility. We demonstrate this by showing three ways in which user confidence in the outbound translation, as well as its overall final quality, can be affected: backward translation, quality estimation (with alignment) and source paraphrasing. In this paper, we describe an experiment on outbound translation from English to Czech and Estonian. We examine the effects of each proposed feedback module and further focus on how the quality of machine translation systems influence these findings and the user perception of success. We show that backward translation feedback has a mixed effect on the whole process: it increases user confidence in the produced translation, but not the objective quality.

| Comments: | 9 pages (excluding references); to appear at NAACL-HWT 2021  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Human-Computer Interaction (cs.HC) |
| Cite as:  | **[arXiv:2104.05688](https://arxiv.org/abs/2104.05688) [cs.CL]** |
|           | (or **[arXiv:2104.05688v1](https://arxiv.org/abs/2104.05688v1) [cs.CL]** for this version) |













# 2021-04-12

[Return to Index](#Index)



<h2 id="2021-04-12-1">1. Video-aided Unsupervised Grammar Induction
</h2>

Title: [Video-aided Unsupervised Grammar Induction](https://arxiv.org/abs/2104.04369)

Authors: [Songyang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+S), [Linfeng Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+L), [Lifeng Jin](https://arxiv.org/search/cs?searchtype=author&query=Jin%2C+L), [Kun Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+K), [Dong Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+D), [Jiebo Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+J)

> We investigate video-aided grammar induction, which learns a constituency parser from both unlabeled text and its corresponding video. Existing methods of multi-modal grammar induction focus on learning syntactic grammars from text-image pairs, with promising results showing that the information from static images is useful in induction. However, videos provide even richer information, including not only static objects but also actions and state changes useful for inducing verb phrases. In this paper, we explore rich features (e.g. action, object, scene, audio, face, OCR and speech) from videos, taking the recent Compound PCFG model as the baseline. We further propose a Multi-Modal Compound PCFG model (MMC-PCFG) to effectively aggregate these rich features from different modalities. Our proposed MMC-PCFG is trained end-to-end and outperforms each individual modality and previous state-of-the-art systems on three benchmarks, i.e. DiDeMo, YouCook2 and MSRVTT, confirming the effectiveness of leveraging video information for unsupervised grammar induction.

| Comments: | This paper is accepted by NAACL'21                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2104.04369](https://arxiv.org/abs/2104.04369) [cs.CV]** |
|           | (or **[arXiv:2104.04369v1](https://arxiv.org/abs/2104.04369v1) [cs.CV]** for this version) |





<h2 id="2021-04-12-2">2. Design and Implementation of English To Yoruba Verb Phrase Machine Translation System
</h2>

Title: [Design and Implementation of English To Yoruba Verb Phrase Machine Translation System](https://arxiv.org/abs/2104.04125)

Authors: [Safiriyu Eludiora](https://arxiv.org/search/cs?searchtype=author&query=Eludiora%2C+S), [Benjamin Ajibade](https://arxiv.org/search/cs?searchtype=author&query=Ajibade%2C+B)

> We aim to develop an English to Yoruba machine translation system which can translate English verb phrase text to its Yoruba equivalent.Words from both languages Source Language and Target Language were collected for the verb phrase group in the home domain.The lexical translation is done by assigning values of the matching word in the dictionary.The syntax of the two languages was realized using Context-Free Grammar,we validated the rewrite rules with finite state automata.The human evaluation method was used and expert fluency scored.The evaluation shows the system performed better than that of sampled Google translation with over 70 percent of the response matching that of the system's output.

| Comments: | 9 pages, 9 figures                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2104.04125](https://arxiv.org/abs/2104.04125) [cs.CL]** |
|           | (or **[arXiv:2104.04125v1](https://arxiv.org/abs/2104.04125v1) [cs.CL]** for this version) |





<h2 id="2021-04-12-3">3. Efficient Large-Scale Language Model Training on GPU Clusters
</h2>

Title: [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473)

Authors: [Deepak Narayanan](https://arxiv.org/search/cs?searchtype=author&query=Narayanan%2C+D), [Mohammad Shoeybi](https://arxiv.org/search/cs?searchtype=author&query=Shoeybi%2C+M), [Jared Casper](https://arxiv.org/search/cs?searchtype=author&query=Casper%2C+J), [Patrick LeGresley](https://arxiv.org/search/cs?searchtype=author&query=LeGresley%2C+P), [Mostofa Patwary](https://arxiv.org/search/cs?searchtype=author&query=Patwary%2C+M), [Vijay Korthikanti](https://arxiv.org/search/cs?searchtype=author&query=Korthikanti%2C+V), [Dmitri Vainbrand](https://arxiv.org/search/cs?searchtype=author&query=Vainbrand%2C+D), [Prethvi Kashinkunti](https://arxiv.org/search/cs?searchtype=author&query=Kashinkunti%2C+P), [Julie Bernauer](https://arxiv.org/search/cs?searchtype=author&query=Bernauer%2C+J), [Bryan Catanzaro](https://arxiv.org/search/cs?searchtype=author&query=Catanzaro%2C+B), [Amar Phanishayee](https://arxiv.org/search/cs?searchtype=author&query=Phanishayee%2C+A), [Matei Zaharia](https://arxiv.org/search/cs?searchtype=author&query=Zaharia%2C+M)

> Large language models have led to state-of-the-art accuracies across a range of tasks. However, training these large models efficiently is challenging for two reasons: a) GPU memory capacity is limited, making it impossible to fit large models on a single GPU or even on a multi-GPU server; and b) the number of compute operations required to train these models can result in unrealistically long training times. New methods of model parallelism such as tensor and pipeline parallelism have been proposed to address these challenges; unfortunately, naive usage leads to fundamental scaling issues at thousands of GPUs due to various reasons, e.g., expensive cross-node communication or idle periods waiting on other devices.
> In this work, we show how to compose different types of parallelism methods (tensor, pipeline, and data paralleism) to scale to thousands of GPUs, achieving a two-order-of-magnitude increase in the sizes of models we can efficiently train compared to existing systems. We discuss various implementations of pipeline parallelism and propose a novel schedule that can improve throughput by more than 10% with comparable memory footprint compared to previously-proposed approaches. We quantitatively study the trade-offs between tensor, pipeline, and data parallelism, and provide intuition as to how to configure distributed training of a large model. The composition of these techniques allows us to perform training iterations on a model with 1 trillion parameters at 502 petaFLOP/s on 3072 GPUs with achieved per-GPU throughput of 52% of peak; previous efforts to train similar-sized models achieve much lower throughput (36% of theoretical peak). Our code has been open-sourced at [this https URL](https://github.com/nvidia/megatron-lm).

| Subjects: | **Computation and Language (cs.CL)**; Distributed, Parallel, and Cluster Computing (cs.DC) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2104.04473](https://arxiv.org/abs/2104.04473) [cs.CL]** |
|           | (or **[arXiv:2104.04473v1](https://arxiv.org/abs/2104.04473v1) [cs.CL]** for this version) |





<h2 id="2021-04-12-4">4. Chinese Character Decomposition for Neural MT with Multi-Word Expressions
</h2>

Title: [Chinese Character Decomposition for Neural MT with Multi-Word Expressions](https://arxiv.org/abs/2104.04497)

Authors: [Lifeng Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+L), [Gareth J. F. Jones](https://arxiv.org/search/cs?searchtype=author&query=Jones%2C+G+J+F), [Alan F. Smeaton](https://arxiv.org/search/cs?searchtype=author&query=Smeaton%2C+A+F), [Paolo Bolzoni](https://arxiv.org/search/cs?searchtype=author&query=Bolzoni%2C+P)

> Chinese character decomposition has been used as a feature to enhance Machine Translation (MT) models, combining radicals into character and word level models. Recent work has investigated ideograph or stroke level embedding. However, questions remain about different decomposition levels of Chinese character representations, radical and strokes, best suited for MT. To investigate the impact of Chinese decomposition embedding in detail, i.e., radical, stroke, and intermediate levels, and how well these decompositions represent the meaning of the original character sequences, we carry out analysis with both automated and human evaluation of MT. Furthermore, we investigate if the combination of decomposed Multiword Expressions (MWEs) can enhance the model learning. MWE integration into MT has seen more than a decade of exploration. However, decomposed MWEs has not previously been explored.

| Comments: | Accepted to publish in NoDaLiDa2021                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2104.04497](https://arxiv.org/abs/2104.04497) [cs.CL]** |
|           | (or **[arXiv:2104.04497v1](https://arxiv.org/abs/2104.04497v1) [cs.CL]** for this version) |










# 2021-04-09

[Return to Index](#Index)



<h2 id="2021-04-09-1">1. Extended Parallel Corpus for Amharic-English Machine Translation
</h2>

Title: [Extended Parallel Corpus for Amharic-English Machine Translation](https://arxiv.org/abs/2104.03543)

Authors: [Andargachew Mekonnen Gezmu](https://arxiv.org/search/cs?searchtype=author&query=Gezmu%2C+A+M), [Andreas Nürnberger](https://arxiv.org/search/cs?searchtype=author&query=Nürnberger%2C+A), [Tesfaye Bayu Bati](https://arxiv.org/search/cs?searchtype=author&query=Bati%2C+T+B)

> This paper describes the acquisition, preprocessing, segmentation, and alignment of an Amharic-English parallel corpus. It will be useful for machine translation of an under-resourced language, Amharic. The corpus is larger than previously compiled corpora; it is released for research purposes. We trained neural machine translation and phrase-based statistical machine translation models using the corpus. In the automatic evaluation, neural machine translation models outperform phrase-based statistical machine translation models.

| Comments: | Accepted to AfricanNLP workshop under EACL 2021              |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2104.03543](https://arxiv.org/abs/2104.03543) [cs.CL]** |
|           | (or **[arXiv:2104.03543v1](https://arxiv.org/abs/2104.03543v1) [cs.CL]** for this version) |





<h2 id="2021-04-09-2">2. BSTC: A Large-Scale Chinese-English Speech Translation Dataset
</h2>

Title: [BSTC: A Large-Scale Chinese-English Speech Translation Dataset](https://arxiv.org/abs/2104.03575)

Authors: [Ruiqing Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+R), [Xiyang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Chuanqiang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+C), [Zhongjun HeHua Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+Z+H), [Zhi Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Haifeng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+H), [Ying Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Qinfei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Q)

> This paper presents BSTC (Baidu Speech Translation Corpus), a large-scale Chinese-English speech translation dataset. This dataset is constructed based on a collection of licensed videos of talks or lectures, including about 68 hours of Mandarin data, their manual transcripts and translations into English, as well as automated transcripts by an automatic speech recognition (ASR) model. We have further asked three experienced interpreters to simultaneously interpret the testing talks in a mock conference setting. This corpus is expected to promote the research of automatic simultaneous translation as well as the development of practical systems. We have organized simultaneous translation tasks and used this corpus to evaluate automatic simultaneous translation systems.

| Comments: | 8 pages, 6 figures                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2104.03575](https://arxiv.org/abs/2104.03575) [cs.CL]** |
|           | (or **[arXiv:2104.03575v1](https://arxiv.org/abs/2104.03575v1) [cs.CL]** for this version) |





<h2 id="2021-04-09-3">3. A Simple Geometric Method for Cross-Lingual Linguistic Transformations with Pre-trained Autoencoders
</h2>

Title: [A Simple Geometric Method for Cross-Lingual Linguistic Transformations with Pre-trained Autoencoders](https://arxiv.org/abs/2104.03630)

Authors: [Maarten De Raedt](https://arxiv.org/search/cs?searchtype=author&query=De+Raedt%2C+M), [Fréderic Godin](https://arxiv.org/search/cs?searchtype=author&query=Godin%2C+F), [Pieter Buteneers](https://arxiv.org/search/cs?searchtype=author&query=Buteneers%2C+P), [Chris Develder](https://arxiv.org/search/cs?searchtype=author&query=Develder%2C+C), [Thomas Demeester](https://arxiv.org/search/cs?searchtype=author&query=Demeester%2C+T)

> Powerful sentence encoders trained for multiple languages are on the rise. These systems are capable of embedding a wide range of linguistic properties into vector representations. While explicit probing tasks can be used to verify the presence of specific linguistic properties, it is unclear whether the vector representations can be manipulated to indirectly steer such properties. We investigate the use of a geometric mapping in embedding space to transform linguistic properties, without any tuning of the pre-trained sentence encoder or decoder. We validate our approach on three linguistic properties using a pre-trained multilingual autoencoder and analyze the results in both monolingual and cross-lingual settings.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2104.03630](https://arxiv.org/abs/2104.03630) [cs.CL]** |
|           | (or **[arXiv:2104.03630v1](https://arxiv.org/abs/2104.03630v1) [cs.CL]** for this version) |





<h2 id="2021-04-09-4">4. Probing BERT in Hyperbolic Spaces
</h2>

Title: [Probing BERT in Hyperbolic Spaces](https://arxiv.org/abs/2104.03869)

Authors: [Boli Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+B), [Yao Fu](https://arxiv.org/search/cs?searchtype=author&query=Fu%2C+Y), [Guangwei Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+G), [Pengjun Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+P), [Chuanqi Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+C), [Mosha Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+M), [Liping Jing](https://arxiv.org/search/cs?searchtype=author&query=Jing%2C+L)

> Recently, a variety of probing tasks are proposed to discover linguistic properties learned in contextualized word embeddings. Many of these works implicitly assume these embeddings lay in certain metric spaces, typically the Euclidean space. This work considers a family of geometrically special spaces, the hyperbolic spaces, that exhibit better inductive biases for hierarchical structures and may better reveal linguistic hierarchies encoded in contextualized representations. We introduce a Poincare probe, a structural probe projecting these embeddings into a Poincare subspace with explicitly defined hierarchies. We focus on two probing objectives: (a) dependency trees where the hierarchy is defined as head-dependent structures; (b) lexical sentiments where the hierarchy is defined as the polarity of words (positivity and negativity). We argue that a key desideratum of a probe is its sensitivity to the existence of linguistic structures. We apply our probes on BERT, a typical contextualized embedding model. In a syntactic subspace, our probe better recovers tree structures than Euclidean probes, revealing the possibility that the geometry of BERT syntax may not necessarily be Euclidean. In a sentiment subspace, we reveal two possible meta-embeddings for positive and negative sentiments and show how lexically-controlled contextualization would change the geometric localization of embeddings. We demonstrate the findings with our Poincare probe via extensive experiments and visualization. Our results can be reproduced at [this https URL](https://github.com/FranxYao/PoincareProbe).

| Comments: | ICLR 2021 Camera ready                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2104.03869](https://arxiv.org/abs/2104.03869) [cs.CL]** |
|           | (or **[arXiv:2104.03869v1](https://arxiv.org/abs/2104.03869v1) [cs.CL]** for this version) |





# 2021-04-08

[Return to Index](#Index)



<h2 id="2021-04-08-1">1. VERB: Visualizing and Interpreting Bias Mitigation Techniques for Word Representations
</h2>

Title: [VERB: Visualizing and Interpreting Bias Mitigation Techniques for Word Representations](https://arxiv.org/abs/2104.02797)

Authors: [Archit Rathore](https://arxiv.org/search/cs?searchtype=author&query=Rathore%2C+A), [Sunipa Dev](https://arxiv.org/search/cs?searchtype=author&query=Dev%2C+S), [Jeff M. Phillips](https://arxiv.org/search/cs?searchtype=author&query=Phillips%2C+J+M), [Vivek Srikumar](https://arxiv.org/search/cs?searchtype=author&query=Srikumar%2C+V), [Yan Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+Y), [Chin-Chia Michael Yeh](https://arxiv.org/search/cs?searchtype=author&query=Yeh%2C+C+M), [Junpeng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J), [Wei Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+W), [Bei Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+B)

> Word vector embeddings have been shown to contain and amplify biases in data they are extracted from. Consequently, many techniques have been proposed to identify, mitigate, and attenuate these biases in word representations. In this paper, we utilize interactive visualization to increase the interpretability and accessibility of a collection of state-of-the-art debiasing techniques. To aid this, we present Visualization of Embedding Representations for deBiasing system ("VERB"), an open-source web-based visualization tool that helps the users gain a technical understanding and visual intuition of the inner workings of debiasing techniques, with a focus on their geometric properties. In particular, VERB offers easy-to-follow use cases in exploring the effects of these debiasing techniques on the geometry of high-dimensional word vectors. To help understand how various debiasing techniques change the underlying geometry, VERB decomposes each technique into interpretable sequences of primitive transformations and highlights their effect on the word vectors using dimensionality reduction and interactive visual exploration. VERB is designed to target natural language processing (NLP) practitioners who are designing decision-making systems on top of word embeddings, and also researchers working with fairness and ethics of machine learning systems in NLP. It can also serve as a visual medium for education, which helps an NLP novice to understand and mitigate biases in word embeddings.

| Comments: | 11 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Human-Computer Interaction (cs.HC) |
| Cite as:  | **[arXiv:2104.02797](https://arxiv.org/abs/2104.02797) [cs.CL]** |
|           | (or **[arXiv:2104.02797v1](https://arxiv.org/abs/2104.02797v1) [cs.CL]** for this version) |





<h2 id="2021-04-08-2">2. Better Neural Machine Translation by Extracting Linguistic Information from BERT
</h2>

Title: [Better Neural Machine Translation by Extracting Linguistic Information from BERT](https://arxiv.org/abs/2104.02831)

Authors: [Hassan S. Shavarani](https://arxiv.org/search/cs?searchtype=author&query=Shavarani%2C+H+S), [Anoop Sarkar](https://arxiv.org/search/cs?searchtype=author&query=Sarkar%2C+A)

> Adding linguistic information (syntax or semantics) to neural machine translation (NMT) has mostly focused on using point estimates from pre-trained models. Directly using the capacity of massive pre-trained contextual word embedding models such as BERT (Devlin et al., 2019) has been marginally useful in NMT because effective fine-tuning is difficult to obtain for NMT without making training brittle and unreliable. We augment NMT by extracting dense fine-tuned vector-based linguistic information from BERT instead of using point estimates. Experimental results show that our method of incorporating linguistic information helps NMT to generalize better in a variety of training contexts and is no more difficult to train than conventional Transformer-based NMT.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2104.02831](https://arxiv.org/abs/2104.02831) [cs.CL]** |
|           | (or **[arXiv:2104.02831v1](https://arxiv.org/abs/2104.02831v1) [cs.CL]** for this version) |



<h2 id="2021-04-08-3">3. GrammarTagger: A Multilingual, Minimally-Supervised Grammar Profiler for Language Education
</h2>

Title: [GrammarTagger: A Multilingual, Minimally-Supervised Grammar Profiler for Language Education](https://arxiv.org/abs/2104.03190)

Authors: [Masato Hagiwara](https://arxiv.org/search/cs?searchtype=author&query=Hagiwara%2C+M), [Joshua Tanner](https://arxiv.org/search/cs?searchtype=author&query=Tanner%2C+J), [Keisuke Sakaguchi](https://arxiv.org/search/cs?searchtype=author&query=Sakaguchi%2C+K)

> We present GrammarTagger, an open-source grammar profiler which, given an input text, identifies grammatical features useful for language education. The model architecture enables it to learn from a small amount of texts annotated with spans and their labels, which 1) enables easier and more intuitive annotation, 2) supports overlapping spans, and 3) is less prone to error propagation, compared to complex hand-crafted rules defined on constituency/dependency parses. We show that we can bootstrap a grammar profiler model with F1≈0.6 from only a couple hundred sentences both in English and Chinese, which can be further boosted via learning a multilingual model. With GrammarTagger, we also build Octanove Learn, a search engine of language learning materials indexed by their reading difficulty and grammatical features. The code and pretrained models are publicly available at \url{[this https URL](https://github.com/octanove/grammartagger)}.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2104.03190](https://arxiv.org/abs/2104.03190) [cs.CL]** |
|           | (or **[arXiv:2104.03190v1](https://arxiv.org/abs/2104.03190v1) [cs.CL]** for this version) |









# 2021-04-07

[Return to Index](#Index)



<h2 id="2021-04-07-1">1. Semantic Distance: A New Metric for ASR Performance Analysis Towards Spoken Language Understanding
</h2>

Title: [Semantic Distance: A New Metric for ASR Performance Analysis Towards Spoken Language Understanding](https://arxiv.org/abs/2104.02138)

Authors: [Suyoun Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+S), [Abhinav Arora](https://arxiv.org/search/cs?searchtype=author&query=Arora%2C+A), [Duc Le](https://arxiv.org/search/cs?searchtype=author&query=Le%2C+D), [Ching-Feng Yeh](https://arxiv.org/search/cs?searchtype=author&query=Yeh%2C+C), [Christian Fuegen](https://arxiv.org/search/cs?searchtype=author&query=Fuegen%2C+C), [Ozlem Kalinli](https://arxiv.org/search/cs?searchtype=author&query=Kalinli%2C+O), [Michael L. Seltzer](https://arxiv.org/search/cs?searchtype=author&query=Seltzer%2C+M+L)

> Word Error Rate (WER) has been the predominant metric used to evaluate the performance of automatic speech recognition (ASR) systems. However, WER is sometimes not a good indicator for downstream Natural Language Understanding (NLU) tasks, such as intent recognition, slot filling, and semantic parsing in task-oriented dialog systems. This is because WER takes into consideration only literal correctness instead of semantic correctness, the latter of which is typically more important for these downstream tasks. In this study, we propose a novel Semantic Distance (SemDist) measure as an alternative evaluation metric for ASR systems to address this issue. We define SemDist as the distance between a reference and hypothesis pair in a sentence-level embedding space. To represent the reference and hypothesis as a sentence embedding, we exploit RoBERTa, a state-of-the-art pre-trained deep contextualized language model based on the transformer architecture. We demonstrate the effectiveness of our proposed metric on various downstream tasks, including intent recognition, semantic parsing, and named entity recognition.

| Comments: | submitted to Interspeech 2021                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2104.02138](https://arxiv.org/abs/2104.02138) [cs.CL]** |
|           | (or **[arXiv:2104.02138v1](https://arxiv.org/abs/2104.02138v1) [cs.CL]** for this version) |





<h2 id="2021-04-07-2">2. ODE Transformer: An Ordinary Differential Equation-Inspired Model for Neural Machine Translation
</h2>

Title: [ODE Transformer: An Ordinary Differential Equation-Inspired Model for Neural Machine Translation](https://arxiv.org/abs/2104.02308)

Authors: [Bei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+B), [Quan Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+Q), [Tao Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+T), [Shuhan Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+S), [Xin Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+X), [Tong Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+T), [Jingbo Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+J)

> It has been found that residual networks are an Euler discretization of solutions to Ordinary Differential Equations (ODEs). In this paper, we explore a deeper relationship between Transformer and numerical methods of ODEs. We show that a residual block of layers in Transformer can be described as a higher-order solution to ODEs. This leads us to design a new architecture (call it ODE Transformer) analogous to the Runge-Kutta method that is well motivated in ODEs. As a natural extension to Transformer, ODE Transformer is easy to implement and parameter efficient. Our experiments on three WMT tasks demonstrate the genericity of this model, and large improvements in performance over several strong baselines. It achieves 30.76 and 44.11 BLEU scores on the WMT'14 En-De and En-Fr test data. This sets a new state-of-the-art on the WMT'14 En-Fr task.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2104.02308](https://arxiv.org/abs/2104.02308) [cs.CL]** |
|           | (or **[arXiv:2104.02308v1](https://arxiv.org/abs/2104.02308v1) [cs.CL]** for this version) |







# 2021-04-06

[Return to Index](#Index)



<h2 id="2021-04-06-1">1. TSNAT: Two-Step Non-Autoregressvie Transformer Models for Speech Recognition
</h2>

Title: [TSNAT: Two-Step Non-Autoregressvie Transformer Models for Speech Recognition](https://arxiv.org/abs/2104.01522)

Authors: [Zhengkun Tian](https://arxiv.org/search/eess?searchtype=author&query=Tian%2C+Z), [Jiangyan Yi](https://arxiv.org/search/eess?searchtype=author&query=Yi%2C+J), [Jianhua Tao](https://arxiv.org/search/eess?searchtype=author&query=Tao%2C+J), [Ye Bai](https://arxiv.org/search/eess?searchtype=author&query=Bai%2C+Y), [Shuai Zhang](https://arxiv.org/search/eess?searchtype=author&query=Zhang%2C+S), [Zhengqi Wen](https://arxiv.org/search/eess?searchtype=author&query=Wen%2C+Z), [Xuefei Liu](https://arxiv.org/search/eess?searchtype=author&query=Liu%2C+X)

> The autoregressive (AR) models, such as attention-based encoder-decoder models and RNN-Transducer, have achieved great success in speech recognition. They predict the output sequence conditioned on the previous tokens and acoustic encoded states, which is inefficient on GPUs. The non-autoregressive (NAR) models can get rid of the temporal dependency between the output tokens and predict the entire output tokens in at least one step. However, the NAR model still faces two major problems. On the one hand, there is still a great gap in performance between the NAR models and the advanced AR models. On the other hand, it's difficult for most of the NAR models to train and converge. To address these two problems, we propose a new model named the two-step non-autoregressive transformer(TSNAT), which improves the performance and accelerating the convergence of the NAR model by learning prior knowledge from a parameters-sharing AR model. Furthermore, we introduce the two-stage method into the inference process, which improves the model performance greatly. All the experiments are conducted on a public Chinese mandarin dataset ASIEHLL-1. The results show that the TSNAT can achieve a competitive performance with the AR model and outperform many complicated NAR models.

| Comments: | Submitted to Interspeech2021                                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2104.01522](https://arxiv.org/abs/2104.01522) [eess.AS]** |
|           | (or **[arXiv:2104.01522v1](https://arxiv.org/abs/2104.01522v1) [eess.AS]** for this version) |





<h2 id="2021-04-06-2">2. Attention Forcing for Machine Translation
</h2>

Title: [Attention Forcing for Machine Translation](https://arxiv.org/abs/2104.01264)

Authors: [Qingyun Dou](https://arxiv.org/search/cs?searchtype=author&query=Dou%2C+Q), [Yiting Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+Y), [Potsawee Manakul](https://arxiv.org/search/cs?searchtype=author&query=Manakul%2C+P), [Xixin Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+X), [Mark J. F. Gales](https://arxiv.org/search/cs?searchtype=author&query=Gales%2C+M+J+F)

> Auto-regressive sequence-to-sequence models with attention mechanisms have achieved state-of-the-art performance in various tasks including Text-To-Speech (TTS) and Neural Machine Translation (NMT). The standard training approach, teacher forcing, guides a model with the reference output history. At inference stage, the generated output history must be used. This mismatch can impact performance. However, it is highly challenging to train the model using the generated output. Several approaches have been proposed to address this problem, normally by selectively using the generated output history. To make training stable, these approaches often require a heuristic schedule or an auxiliary classifier. This paper introduces attention forcing for NMT. This approach guides the model with the generated output history and reference attention, and can reduce the training-inference mismatch without a schedule or a classifier. Attention forcing has been successful in TTS, but its application to NMT is more challenging, due to the discrete and multi-modal nature of the output space. To tackle this problem, this paper adds a selection scheme to vanilla attention forcing, which automatically selects a suitable training approach for each pair of training data. Experiments show that attention forcing can improve the overall translation quality and the diversity of the translations.

| Comments: | arXiv admin note: text overlap with [arXiv:1909.12289](https://arxiv.org/abs/1909.12289) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2104.01264](https://arxiv.org/abs/2104.01264) [cs.CL]** |
|           | (or **[arXiv:2104.01264v1](https://arxiv.org/abs/2104.01264v1) [cs.CL]** for this version) |







<h2 id="2021-04-06-3">3. WhiteningBERT: An Easy Unsupervised Sentence Embedding Approach
</h2>

Title: [WhiteningBERT: An Easy Unsupervised Sentence Embedding Approach](https://arxiv.org/abs/2104.01767)

Authors: [Junjie Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+J), [Duyu Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+D), [Wanjun Zhong](https://arxiv.org/search/cs?searchtype=author&query=Zhong%2C+W), [Shuai Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+S), [Linjun Shou](https://arxiv.org/search/cs?searchtype=author&query=Shou%2C+L), [Ming Gong](https://arxiv.org/search/cs?searchtype=author&query=Gong%2C+M), [Daxin Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+D), [Nan Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan%2C+N)

> Producing the embedding of a sentence in an unsupervised way is valuable to natural language matching and retrieval problems in practice. In this work, we conduct a thorough examination of pretrained model based unsupervised sentence embeddings. We study on four pretrained models and conduct massive experiments on seven datasets regarding sentence semantics. We have there main findings. First, averaging all tokens is better than only using [CLS] vector. Second, combining both top andbottom layers is better than only using top layers. Lastly, an easy whitening-based vector normalization strategy with less than 10 lines of code consistently boosts the performance.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2104.01767](https://arxiv.org/abs/2104.01767) [cs.CL]** |
|           | (or **[arXiv:2104.01767v1](https://arxiv.org/abs/2104.01767v1) [cs.CL]** for this version) |







<h2 id="2021-04-06-4">4. Rethinking Perturbations in Encoder-Decoders for Fast Training
</h2>

Title: [Rethinking Perturbations in Encoder-Decoders for Fast Training](https://arxiv.org/abs/2104.01853)

Authors: [Sho Takase](https://arxiv.org/search/cs?searchtype=author&query=Takase%2C+S), [Shun Kiyono](https://arxiv.org/search/cs?searchtype=author&query=Kiyono%2C+S)

> We often use perturbations to regularize neural models. For neural encoder-decoders, previous studies applied the scheduled sampling (Bengio et al., 2015) and adversarial perturbations (Sato et al., 2019) as perturbations but these methods require considerable computational time. Thus, this study addresses the question of whether these approaches are efficient enough for training time. We compare several perturbations in sequence-to-sequence problems with respect to computational time. Experimental results show that the simple techniques such as word dropout (Gal and Ghahramani, 2016) and random replacement of input tokens achieve comparable (or better) scores to the recently proposed perturbations, even though these simple methods are faster. Our code is publicly available at [this https URL](https://github.com/takase/rethink_perturbations).

| Comments: | Accepted at NAACL-HLT 2021                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2104.01853](https://arxiv.org/abs/2104.01853) [cs.CL]** |
|           | (or **[arXiv:2104.01853v1](https://arxiv.org/abs/2104.01853v1) [cs.CL]** for this version) |







# 2021-04-02

[Return to Index](#Index)



<h2 id="2021-04-02-1">1. Is Label Smoothing Truly Incompatible with Knowledge Distillation: An Empirical Study
</h2>

Title: [Is Label Smoothing Truly Incompatible with Knowledge Distillation: An Empirical Study](https://arxiv.org/abs/2104.00676)

Authors: [Zhiqiang Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+Z), [Zechun Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Dejia Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+D), [Zitian Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Z), [Kwang-Ting Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng%2C+K), [Marios Savvides](https://arxiv.org/search/cs?searchtype=author&query=Savvides%2C+M)

> This work aims to empirically clarify a recently discovered perspective that label smoothing is incompatible with knowledge distillation. We begin by introducing the motivation behind on how this incompatibility is raised, i.e., label smoothing erases relative information between teacher logits. We provide a novel connection on how label smoothing affects distributions of semantically similar and dissimilar classes. Then we propose a metric to quantitatively measure the degree of erased information in sample's representation. After that, we study its one-sidedness and imperfection of the incompatibility view through massive analyses, visualizations and comprehensive experiments on Image Classification, Binary Networks, and Neural Machine Translation. Finally, we broadly discuss several circumstances wherein label smoothing will indeed lose its effectiveness. Project page: [this http URL](http://zhiqiangshen.com/projects/LS_and_KD/index.html).

| Comments: | ICLR 2021. Project page: [this http URL](http://zhiqiangshen.com/projects/LS_and_KD/index.html) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2104.00676](https://arxiv.org/abs/2104.00676) [cs.LG]** |
|           | (or **[arXiv:2104.00676v1](https://arxiv.org/abs/2104.00676v1) [cs.LG]** for this version) |





<h2 id="2021-04-02-2">2. Domain-specific MT for Low-resource Languages: The case of Bambara-French
</h2>

Title: [Domain-specific MT for Low-resource Languages: The case of Bambara-French](https://arxiv.org/abs/2104.00041)

Authors: [Allahsera Auguste Tapo](https://arxiv.org/search/cs?searchtype=author&query=Tapo%2C+A+A), [Michael Leventhal](https://arxiv.org/search/cs?searchtype=author&query=Leventhal%2C+M), [Sarah Luger](https://arxiv.org/search/cs?searchtype=author&query=Luger%2C+S), [Christopher M. Homan](https://arxiv.org/search/cs?searchtype=author&query=Homan%2C+C+M), [Marcos Zampieri](https://arxiv.org/search/cs?searchtype=author&query=Zampieri%2C+M)

> Translating to and from low-resource languages is a challenge for machine translation (MT) systems due to a lack of parallel data. In this paper we address the issue of domain-specific MT for Bambara, an under-resourced Mande language spoken in Mali. We present the first domain-specific parallel dataset for MT of Bambara into and from French. We discuss challenges in working with small quantities of domain-specific data for a low-resource language and we present the results of machine learning experiments on this data.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2104.00041](https://arxiv.org/abs/2104.00041) [cs.CL]** |
|           | (or **[arXiv:2104.00041v1](https://arxiv.org/abs/2104.00041v1) [cs.CL]** for this version) |





<h2 id="2021-04-02-3">3. Zero-Shot Language Transfer vs Iterative Back Translation for Unsupervised Machine Translation
</h2>

Title: [Zero-Shot Language Transfer vs Iterative Back Translation for Unsupervised Machine Translation](https://arxiv.org/abs/2104.00106)

Authors: [Aviral Joshi](https://arxiv.org/search/cs?searchtype=author&query=Joshi%2C+A), [Chengzhi Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+C), [Har Simrat Singh](https://arxiv.org/search/cs?searchtype=author&query=Singh%2C+H+S)

> This work focuses on comparing different solutions for machine translation on low resource language pairs, namely, with zero-shot transfer learning and unsupervised machine translation. We discuss how the data size affects the performance of both unsupervised MT and transfer learning. Additionally we also look at how the domain of the data affects the result of unsupervised MT. The code to all the experiments performed in this project are accessible on Github.

| Comments: | 7 pages, 2 figures, 4 tables                                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2104.00106](https://arxiv.org/abs/2104.00106) [cs.CL]** |
|           | (or **[arXiv:2104.00106v1](https://arxiv.org/abs/2104.00106v1) [cs.CL]** for this version) |





<h2 id="2021-04-02-4">4. Detecting over/under-translation errors for determining adequacy in human translations
</h2>

Title: [Detecting over/under-translation errors for determining adequacy in human translations](https://arxiv.org/abs/2104.00267)

Authors: [Prabhakar Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+P), [Ridha Juneja](https://arxiv.org/search/cs?searchtype=author&query=Juneja%2C+R), [Anil Nelakanti](https://arxiv.org/search/cs?searchtype=author&query=Nelakanti%2C+A), [Tamojit Chatterjee](https://arxiv.org/search/cs?searchtype=author&query=Chatterjee%2C+T)

> We present a novel approach to detecting over and under translations (OT/UT) as part of adequacy error checks in translation evaluation. We do not restrict ourselves to machine translation (MT) outputs and specifically target applications with human generated translation pipeline. The goal of our system is to identify OT/UT errors from human translated video subtitles with high error recall. We achieve this without reference translations by learning a model on synthesized training data. We compare various classification networks that we trained on embeddings from pre-trained language model with our best hybrid network of GRU + CNN achieving 89.3% accuracy on high-quality human-annotated evaluation data in 8 languages.

| Comments: | 6 pages, 5 tables                                            |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2104.00267](https://arxiv.org/abs/2104.00267) [cs.CL]** |
|           | (or **[arXiv:2104.00267v1](https://arxiv.org/abs/2104.00267v1) [cs.CL]** for this version) |





<h2 id="2021-04-02-5">5. Many-to-English Machine Translation Tools, Data, and Pretrained Models
</h2>

Title: [Many-to-English Machine Translation Tools, Data, and Pretrained Models](https://arxiv.org/abs/2104.00290)

Authors: [Thamme Gowda](https://arxiv.org/search/cs?searchtype=author&query=Gowda%2C+T), [Zhao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Chris A Mattmann](https://arxiv.org/search/cs?searchtype=author&query=Mattmann%2C+C+A), [Jonathan May](https://arxiv.org/search/cs?searchtype=author&query=May%2C+J)

> While there are more than 7000 languages in the world, most translation research efforts have targeted a few high-resource languages. Commercial translation systems support only one hundred languages or fewer, and do not make these models available for transfer to low resource languages. In this work, we present useful tools for machine translation research: MTData, NLCodec, and RTG. We demonstrate their usefulness by creating a multilingual neural machine translation model capable of translating from 500 source languages to English. We make this multilingual model readily downloadable and usable as a service, or as a parent model for transfer-learning to even lower-resource languages.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2104.00290](https://arxiv.org/abs/2104.00290) [cs.CL]** |
|           | (or **[arXiv:2104.00290v1](https://arxiv.org/abs/2104.00290v1) [cs.CL]** for this version) |





<h2 id="2021-04-02-6">6. Low-Resource Neural Machine Translation for South-Eastern African Languages
</h2>

Title: [Low-Resource Neural Machine Translation for South-Eastern African Languages](https://arxiv.org/abs/2104.00366)

Authors: [Evander Nyoni](https://arxiv.org/search/cs?searchtype=author&query=Nyoni%2C+E), [Bruce A. Bassett](https://arxiv.org/search/cs?searchtype=author&query=Bassett%2C+B+A)

> Low-resource African languages have not fully benefited from the progress in neural machine translation because of a lack of data. Motivated by this challenge we compare zero-shot learning, transfer learning and multilingual learning on three Bantu languages (Shona, isiXhosa and isiZulu) and English. Our main target is English-to-isiZulu translation for which we have just 30,000 sentence pairs, 28% of the average size of our other corpora. We show the importance of language similarity on the performance of English-to-isiZulu transfer learning based on English-to-isiXhosa and English-to-Shona parent models whose BLEU scores differ by 5.2. We then demonstrate that multilingual learning surpasses both transfer learning and zero-shot learning on our dataset, with BLEU score improvements relative to the baseline English-to-isiZulu model of 9.9, 6.1 and 2.0 respectively. Our best model also improves the previous SOTA BLEU score by more than 10.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2104.00366](https://arxiv.org/abs/2104.00366) [cs.CL]** |
|           | (or **[arXiv:2104.00366v1](https://arxiv.org/abs/2104.00366v1) [cs.CL]** for this version) |





<h2 id="2021-04-02-7">7. WakaVT: A Sequential Variational Transformer for Waka Generation
</h2>

Title: [WakaVT: A Sequential Variational Transformer for Waka Generation](https://arxiv.org/abs/2104.00426)

Authors: [Yuka Takeishi](https://arxiv.org/search/cs?searchtype=author&query=Takeishi%2C+Y), [Mingxuan Niu](https://arxiv.org/search/cs?searchtype=author&query=Niu%2C+M), [Jing Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+J), [Zhong Jin](https://arxiv.org/search/cs?searchtype=author&query=Jin%2C+Z), [Xinyu Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+X)

> Poetry generation has long been a challenge for artificial intelligence. In the scope of Japanese poetry generation, many researchers have paid attention to Haiku generation, but few have focused on Waka generation. To further explore the creative potential of natural language generation systems in Japanese poetry creation, we propose a novel Waka generation model, WakaVT, which automatically produces Waka poems given user-specified keywords. Firstly, an additive mask-based approach is presented to satisfy the form constraint. Secondly, the structures of Transformer and variational autoencoder are integrated to enhance the quality of generated content. Specifically, to obtain novelty and diversity, WakaVT employs a sequence of latent variables, which effectively captures word-level variability in Waka data. To improve linguistic quality in terms of fluency, coherence, and meaningfulness, we further propose the fused multilevel self-attention mechanism, which properly models the hierarchical linguistic structure of Waka. To the best of our knowledge, we are the first to investigate Waka generation with models based on Transformer and/or variational autoencoder. Both objective and subjective evaluation results demonstrate that our model outperforms baselines significantly.

| Comments: | This paper has been submitted to Neural Processing Letters   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2104.00426](https://arxiv.org/abs/2104.00426) [cs.CL]** |
|           | (or **[arXiv:2104.00426v1](https://arxiv.org/abs/2104.00426v1) [cs.CL]** for this version) |





<h2 id="2021-04-02-8">8. Sampling and Filtering of Neural Machine Translation Distillation Data
</h2>

Title: [Sampling and Filtering of Neural Machine Translation Distillation Data](https://arxiv.org/abs/2104.00664)

Authors: [Vilém Zouhar](https://arxiv.org/search/cs?searchtype=author&query=Zouhar%2C+V)

> In most of neural machine translation distillation or stealing scenarios, the goal is to preserve the performance of the target model (teacher). The highest-scoring hypothesis of the teacher model is commonly used to train a new model (student). If reference translations are also available, then better hypotheses (with respect to the references) can be upsampled and poor hypotheses either removed or undersampled.
> This paper explores the importance sampling method landscape (pruning, hypothesis upsampling and undersampling, deduplication and their combination) with English to Czech and English to German MT models using standard MT evaluation metrics. We show that careful upsampling and combination with the original data leads to better performance when compared to training only on the original or synthesized data or their direct combination.

| Comments: | 6 pages (without references); to be published in NAACL-SRW   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2104.00664](https://arxiv.org/abs/2104.00664) [cs.CL]** |
|           | (or **[arXiv:2104.00664v1](https://arxiv.org/abs/2104.00664v1) [cs.CL]** for this version) |







# 2021-04-01

[Return to Index](#Index)



<h2 id="2021-04-01-1">1. An Exploration of Data Augmentation Techniques for Improving English to Tigrinya Translation
</h2>

Title: [An Exploration of Data Augmentation Techniques for Improving English to Tigrinya Translation](https://arxiv.org/abs/2103.16789)

Authors:[Lidia Kidane](https://arxiv.org/search/cs?searchtype=author&query=Kidane%2C+L), [Sachin Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+S), [Yulia Tsvetkov](https://arxiv.org/search/cs?searchtype=author&query=Tsvetkov%2C+Y)

> It has been shown that the performance of neural machine translation (NMT) drops starkly in low-resource conditions, often requiring large amounts of auxiliary data to achieve competitive results. An effective method of generating auxiliary data is back-translation of target language sentences. In this work, we present a case study of Tigrinya where we investigate several back-translation methods to generate synthetic source sentences. We find that in low-resource conditions, back-translation by pivoting through a higher-resource language related to the target language proves most effective resulting in substantial improvements over baselines.

| Comments: | Accepted at AfricaNLP Workshop, EACL 2021                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2103.16789](https://arxiv.org/abs/2103.16789) [cs.CL]** |
|           | (or **[arXiv:2103.16789v1](https://arxiv.org/abs/2103.16789v1) [cs.CL]** for this version) |





<h2 id="2021-04-01-2">2. Few-shot learning through contextual data augmentation
</h2>

Title: [Few-shot learning through contextual data augmentation](https://arxiv.org/abs/2103.16911)

Authors:[Farid Arthaud](https://arxiv.org/search/cs?searchtype=author&query=Arthaud%2C+F), [Rachel Bawden](https://arxiv.org/search/cs?searchtype=author&query=Bawden%2C+R), [Alexandra Birch](https://arxiv.org/search/cs?searchtype=author&query=Birch%2C+A)

> Machine translation (MT) models used in industries with constantly changing topics, such as translation or news agencies, need to adapt to new data to maintain their performance over time. Our aim is to teach a pre-trained MT model to translate previously unseen words accurately, based on very few examples. We propose (i) an experimental setup allowing us to simulate novel vocabulary appearing in human-submitted translations, and (ii) corresponding evaluation metrics to compare our approaches. We extend a data augmentation approach using a pre-trained language model to create training examples with similar contexts for novel words. We compare different fine-tuning and data augmentation approaches and show that adaptation on the scale of one to five examples is possible. Combining data augmentation with randomly selected training sentences leads to the highest BLEU score and accuracy improvements. Impressively, with only 1 to 5 examples, our model reports better accuracy scores than a reference system trained with on average 313 parallel examples.

| Comments: | 14 pages includince 3 of appendices                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2103.16911](https://arxiv.org/abs/2103.16911) [cs.CL]** |
|           | (or **[arXiv:2103.16911v1](https://arxiv.org/abs/2103.16911v1) [cs.CL]** for this version) |





<h2 id="2021-04-01-3">3. UA-GEC: Grammatical Error Correction and Fluency Corpus for the Ukrainian Language
</h2>

Title: [UA-GEC: Grammatical Error Correction and Fluency Corpus for the Ukrainian Language](https://arxiv.org/abs/2103.16997)

Authors:[Oleksiy Syvokon](https://arxiv.org/search/cs?searchtype=author&query=Syvokon%2C+O), [Olena Nahorna](https://arxiv.org/search/cs?searchtype=author&query=Nahorna%2C+O)

> We present a corpus professionally annotated for grammatical error correction (GEC) and fluency edits in the Ukrainian language. To the best of our knowledge, this is the first GEC corpus for the Ukrainian language. We collected texts with errors (20,715 sentences) from a diverse pool of contributors, including both native and non-native speakers. The data cover a wide variety of writing domains, from text chats and essays to formal writing. Professional proofreaders corrected and annotated the corpus for errors relating to fluency, grammar, punctuation, and spelling. This corpus can be used for developing and evaluating GEC systems in Ukrainian. More generally, it can be used for researching multilingual and low-resource NLP, morphologically rich languages, document-level GEC, and fluency correction. The corpus is publicly available at [this https URL](https://github.com/grammarly/ua-gec)

| Comments: | See [this https URL](https://github.com/grammarly/ua-gec) for the dataset. Version 2 of the data is in progress |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2103.16997](https://arxiv.org/abs/2103.16997) [cs.CL]** |
|           | (or **[arXiv:2103.16997v1](https://arxiv.org/abs/2103.16997v1) [cs.CL]** for this version) |





<h2 id="2021-04-01-4">4. Divide and Rule: Training Context-Aware Multi-Encoder Translation Models with Little Resources
</h2>

Title: [Divide and Rule: Training Context-Aware Multi-Encoder Translation Models with Little Resources](https://arxiv.org/abs/2103.17151)

Authors:[Lorenzo Lupo](https://arxiv.org/search/cs?searchtype=author&query=Lupo%2C+L), [Marco Dinarelli](https://arxiv.org/search/cs?searchtype=author&query=Dinarelli%2C+M), [Laurent Besacier](https://arxiv.org/search/cs?searchtype=author&query=Besacier%2C+L)

> Multi-encoder models are a broad family of context-aware Neural Machine Translation (NMT) systems that aim to improve translation quality by encoding document-level contextual information alongside the current sentence. The context encoding is undertaken by contextual parameters, trained on document-level data. In this work, we show that training these parameters takes large amount of data, since the contextual training signal is sparse. We propose an efficient alternative, based on splitting sentence pairs, that allows to enrich the training signal of a set of parallel sentences by breaking intra-sentential syntactic links, and thus frequently pushing the model to search the context for disambiguating clues. We evaluate our approach with BLEU and contrastive test sets, showing that it allows multi-encoder models to achieve comparable performances to a setting where they are trained with ×10 document-level data. We also show that our approach is a viable option to context-aware NMT for language pairs with zero document-level parallel data.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2103.17151](https://arxiv.org/abs/2103.17151) [cs.CL]** |
|           | (or **[arXiv:2103.17151v1](https://arxiv.org/abs/2103.17151v1) [cs.CL]** for this version) |





<h2 id="2021-04-01-5">5. Leveraging Neural Machine Translation for Word Alignment
</h2>

Title: [Leveraging Neural Machine Translation for Word Alignment](https://arxiv.org/abs/2103.17250)

Authors:[Vilém Zouhar](https://arxiv.org/search/cs?searchtype=author&query=Zouhar%2C+V), [Daria Pylypenko](https://arxiv.org/search/cs?searchtype=author&query=Pylypenko%2C+D)

> The most common tools for word-alignment rely on a large amount of parallel sentences, which are then usually processed according to one of the IBM model algorithms. The training data is, however, the same as for machine translation (MT) systems, especially for neural MT (NMT), which itself is able to produce word-alignments using the trained attention heads. This is convenient because word-alignment is theoretically a viable byproduct of any attention-based NMT, which is also able to provide decoder scores for a translated sentence pair.
> We summarize different approaches on how word-alignment can be extracted from alignment scores and then explore ways in which scores can be extracted from NMT, focusing on inferring the word-alignment scores based on output sentence and token probabilities. We compare this to the extraction of alignment scores from attention. We conclude with aggregating all of the sources of alignment scores into a simple feed-forward network which achieves the best results when combined alignment extractors are used.

| Comments: | 16 pages (without references). To be published in PBML 116   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2103.17250](https://arxiv.org/abs/2103.17250) [cs.CL]** |
|           | (or **[arXiv:2103.17250v1](https://arxiv.org/abs/2103.17250v1) [cs.CL]** for this version) |






