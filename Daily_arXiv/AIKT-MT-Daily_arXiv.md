# Daily arXiv: Machine Translation - Mar., 2020

# Index

- [2020-03-09](#2020-03-09)
  - [1. Distill, Adapt, Distill: Training Small, In-Domain Models for Neural Machine Translation](#2020-03-09-1)
  - [2. What the [MASK]? Making Sense of Language-Specific BERT Models](#2020-03-09-2)
  - [3. Distributional semantic modeling: a revised technique to train term/word vector space models applying the ontology-related approach](#2020-03-09-3)
- [2020-03-06](#2020-03-06)
  - [1. Phase transitions in a decentralized graph-based approach to human language](#2020-03-06-1)
  - [2. BERT as a Teacher: Contextual Embeddings for Sequence-Level Reward](#2020-03-06-2)
  - [3. Zero-Shot Cross-Lingual Transfer with Meta Learning](#2020-03-06-3)
  - [4. An Empirical Accuracy Law for Sequential Machine Translation: the Case of Google Translate](#2020-03-06-4)
- [2020-03-05](#2020-03-05)
  - [1. Evaluating Low-Resource Machine Translation between Chinese and Vietnamese with Back-Translation](#2020-03-05-1)
- [2020-03-04](#2020-03-04)
  - [1. Understanding the Prediction Mechanism of Sentiments by XAI Visualization](#2020-03-04-1)
  - [2. XGPT: Cross-modal Generative Pre-Training for Image Captioning](#2020-03-04-2)
- [2020-03-02](#2020-03-02)
  - [1. Robust Unsupervised Neural Machine Translation with Adversarial Training](#2020-03-02-1)
  - [2. Modeling Future Cost for Neural Machine Translation](#2020-03-02-2)
  - [3. TextBrewer: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing](#2020-03-02-3)
  - [4. Do all Roads Lead to Rome? Understanding the Role of Initialization in Iterative Back-Translation](#2020-03-02-4)
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



# 2020-03-09

[Return to Index](#Index)



<h2 id="2020-03-09-1">1. Distill, Adapt, Distill: Training Small, In-Domain Models for Neural Machine Translation</h2>

Title: [Distill, Adapt, Distill: Training Small, In-Domain Models for Neural Machine Translation](https://arxiv.org/abs/2003.02877)

Authors:[Mitchell A. Gordon](https://arxiv.org/search/cs?searchtype=author&query=Gordon%2C+M+A), [Kevin Duh](https://arxiv.org/search/cs?searchtype=author&query=Duh%2C+K)

*(Submitted on 5 Mar 2020)*

> Abstract: We explore best practices for training small, memory efficient machine translation models with sequence-level knowledge distillation in the domain adaptation setting. While both domain adaptation and knowledge distillation are widely-used, their interaction remains little understood. Our large-scale empirical results in machine translation (on three language pairs with three domains each) suggest distilling twice for best performance: once using general-domain data and again using in-domain data with an adapted teacher.

| Subjects: | Computation and Language (cs.CL)                             |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2003.02877](https://arxiv.org/abs/2003.02877) [cs.CL] |
|           | (or [arXiv:2003.02877v1](https://arxiv.org/abs/2003.02877v1) [cs.CL] for this version) |



<h2 id="2020-03-09-2">2. What the [MASK]? Making Sense of Language-Specific BERT Models</h2>

Title: [What the [MASK]? Making Sense of Language-Specific BERT Models](https://arxiv.org/abs/2003.02912)

Authors: Authors:[Debora Nozza](https://arxiv.org/search/cs?searchtype=author&query=Nozza%2C+D), [Federico Bianchi](https://arxiv.org/search/cs?searchtype=author&query=Bianchi%2C+F), [Dirk Hovy](https://arxiv.org/search/cs?searchtype=author&query=Hovy%2C+D)

*(Submitted on 5 Mar 2020)*

> Abstract: Recently, Natural Language Processing (NLP) has witnessed an impressive progress in many areas, due to the advent of novel, pretrained contextual representation models. In particular, Devlin et al. (2019) proposed a model, called BERT (Bidirectional Encoder Representations from Transformers), which enables researchers to obtain state-of-the art performance on numerous NLP tasks by fine-tuning the representations on their data set and task, without the need for developing and training highly-specific architectures. The authors also released multilingual BERT (mBERT), a model trained on a corpus of 104 languages, which can serve as a universal language model. This model obtained impressive results on a zero-shot cross-lingual natural inference task. Driven by the potential of BERT models, the NLP community has started to investigate and generate an abundant number of BERT models that are trained on a particular language, and tested on a specific data domain and task. This allows us to evaluate the true potential of mBERT as a universal language model, by comparing it to the performance of these more specific models. This paper presents the current state of the art in language-specific BERT models, providing an overall picture with respect to different dimensions (i.e. architectures, data domains, and tasks). Our aim is to provide an immediate and straightforward overview of the commonalities and differences between Language-Specific (language-specific) BERT models and mBERT. We also provide an interactive and constantly updated website that can be used to explore the information we have collected, at [this https URL](https://bertlang.unibocconi.it/).

| Subjects: | Computation and Language (cs.CL)                             |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2003.02912](https://arxiv.org/abs/2003.02912) [cs.CL] |
|           | (or [arXiv:2003.02912v1](https://arxiv.org/abs/2003.02912v1) [cs.CL] for this version) |



<h2 id="2020-03-09-3">3. Distributional semantic modeling: a revised technique to train term/word vector space models applying the ontology-related approach</h2>

Title: [Distributional semantic modeling: a revised technique to train term/word vector space models applying the ontology-related approach](https://arxiv.org/abs/2003.03350)

Authors: Authors:[Oleksandr Palagin](https://arxiv.org/search/cs?searchtype=author&query=Palagin%2C+O), [Vitalii Velychko](https://arxiv.org/search/cs?searchtype=author&query=Velychko%2C+V), [Kyrylo Malakhov](https://arxiv.org/search/cs?searchtype=author&query=Malakhov%2C+K), [Oleksandr Shchurov](https://arxiv.org/search/cs?searchtype=author&query=Shchurov%2C+O)

*(Submitted on 6 Mar 2020)*

> Abstract: We design a new technique for the distributional semantic modeling with a neural network-based approach to learn distributed term representations (or term embeddings) - term vector space models as a result, inspired by the recent ontology-related approach (using different types of contextual knowledge such as syntactic knowledge, terminological knowledge, semantic knowledge, etc.) to the identification of terms (term extraction) and relations between them (relation extraction) called semantic pre-processing technology - SPT. Our method relies on automatic term extraction from the natural language texts and subsequent formation of the problem-oriented or application-oriented (also deeply annotated) text corpora where the fundamental entity is the term (includes non-compositional and compositional terms). This gives us an opportunity to changeover from distributed word representations (or word embeddings) to distributed term representations (or term embeddings). This transition will allow to generate more accurate semantic maps of different subject domains (also, of relations between input terms - it is useful to explore clusters and oppositions, or to test your hypotheses about them). The semantic map can be represented as a graph using Vec2graph - a Python library for visualizing word embeddings (term embeddings in our case) as dynamic and interactive graphs. The Vec2graph library coupled with term embeddings will not only improve accuracy in solving standard NLP tasks, but also update the conventional concept of automated ontology development. The main practical result of our work is the development kit (set of toolkits represented as web service APIs and web application), which provides all necessary routines for the basic linguistic pre-processing and the semantic pre-processing of the natural language texts in Ukrainian for future training of term vector space models.

| Comments: | In English, 9 pages, 2 figures. Not published yet. Prepared for special issue (UkrPROG 2020 conference) of the scientific journal "Problems in programming" (Founder: National Academy of Sciences of Ukraine, Institute of Software Systems of NAS Ukraine) |
| --------- | ------------------------------------------------------------ |
| Subjects: | Computation and Language (cs.CL); Artificial Intelligence (cs.AI) |
| Cite as:  | [arXiv:2003.03350](https://arxiv.org/abs/2003.03350) [cs.CL] |
|           | (or [arXiv:2003.03350v1](https://arxiv.org/abs/2003.03350v1) [cs.CL] for this version) |





# 2020-03-06

[Return to Index](#Index)



<h2 id="2020-03-06-1">1. Phase transitions in a decentralized graph-based approach to human language</h2>

Title: [Phase transitions in a decentralized graph-based approach to human language](https://arxiv.org/abs/2003.02639)

Authors: [Javier Vera](https://arxiv.org/search/physics?searchtype=author&query=Vera%2C+J), [Felipe Urbina](https://arxiv.org/search/physics?searchtype=author&query=Urbina%2C+F), [Wenceslao Palma](https://arxiv.org/search/physics?searchtype=author&query=Palma%2C+W)

*(Submitted on 4 Mar 2020)*

> Zipf's law establishes a scaling behavior for word-frequencies in large text corpora. The appearance of Zipfian properties in human language has been previously explained as an optimization problem for the interests of speakers and hearers. On the other hand, human-like vocabularies can be viewed as bipartite graphs. The aim here is double: within a bipartite-graph approach to human vocabularies, to propose a decentralized language game model for the formation of Zipfian properties. To do this, we define a language game, in which a population of artificial agents is involved in idealized linguistic interactions. Numerical simulations show the appearance of a phase transition from an initially disordered state to three possible phases for language formation. Our results suggest that Zipfian properties in language seem to arise partly from decentralized linguistic interactions between agents endowed with bipartite word-meaning mappings.

| Subjects: | **Physics and Society (physics.soc-ph)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2003.02639](https://arxiv.org/abs/2003.02639) [physics.soc-ph] |
|           | (or [arXiv:2003.02639v1](https://arxiv.org/abs/2003.02639v1) [physics.soc-ph] for this version) |





<h2 id="2020-03-06-2">2. BERT as a Teacher: Contextual Embeddings for Sequence-Level Reward</h2>

Title: [BERT as a Teacher: Contextual Embeddings for Sequence-Level Reward](https://arxiv.org/abs/2003.02738)

Authors: [Florian Schmidt](https://arxiv.org/search/cs?searchtype=author&query=Schmidt%2C+F), [Thomas Hofmann](https://arxiv.org/search/cs?searchtype=author&query=Hofmann%2C+T)

*(Submitted on 5 Mar 2020)*

> Measuring the quality of a generated sequence against a set of references is a central problem in many learning frameworks, be it to compute a score, to assign a reward, or to perform discrimination. Despite great advances in model architectures, metrics that scale independently of the number of references are still based on n-gram estimates. We show that the underlying operations, counting words and comparing counts, can be lifted to embedding words and comparing embeddings. An in-depth analysis of BERT embeddings shows empirically that contextual embeddings can be employed to capture the required dependencies while maintaining the necessary scalability through appropriate pruning and smoothing techniques. We cast unconditional generation as a reinforcement learning problem and show that our reward function indeed provides a more effective learning signal than n-gram reward in this challenging setting.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2003.02738](https://arxiv.org/abs/2003.02738) [cs.LG] |
|           | (or [arXiv:2003.02738v1](https://arxiv.org/abs/2003.02738v1) [cs.LG] for this version) |





<h2 id="2020-03-06-3">3. Zero-Shot Cross-Lingual Transfer with Meta Learning</h2>

Title: [Zero-Shot Cross-Lingual Transfer with Meta Learning](https://arxiv.org/abs/2003.02739)

Authors: [Farhad Nooralahzadeh](https://arxiv.org/search/cs?searchtype=author&query=Nooralahzadeh%2C+F), [Giannis Bekoulis](https://arxiv.org/search/cs?searchtype=author&query=Bekoulis%2C+G), [Johannes Bjerva](https://arxiv.org/search/cs?searchtype=author&query=Bjerva%2C+J), [Isabelle Augenstein](https://arxiv.org/search/cs?searchtype=author&query=Augenstein%2C+I)

*(Submitted on 5 Mar 2020)*

> Learning what to share between tasks has been a topic of high importance recently, as strategic sharing of knowledge has been shown to improve the performance of downstream tasks. The same applies to sharing between languages, and is especially important when considering the fact that most languages in the world suffer from being under-resourced. In this paper, we consider the setting of training models on multiple different languages at the same time, when little or no data is available for languages other than English. We show that this challenging setup can be approached using meta-learning, where, in addition to training a source language model, another model learns to select which training instances are the most beneficial. We experiment using standard supervised, zero-shot cross-lingual, as well as few-shot cross-lingual settings for different natural language understanding tasks (natural language inference, question answering). Our extensive experimental setup demonstrates the consistent effectiveness of meta-learning, on a total 16 languages. We improve upon state-of-the-art on zero-shot and few-shot NLI and QA tasks on the XNLI and X-WikiRe datasets, respectively. We further conduct a comprehensive analysis which indicates that correlation of typological features between languages can further explain when parameter sharing learned via meta learning is beneficial.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2003.02739](https://arxiv.org/abs/2003.02739) [cs.CL] |
|           | (or [arXiv:2003.02739v1](https://arxiv.org/abs/2003.02739v1) [cs.CL] for this version) |





<h2 id="2020-03-06-4">4. An Empirical Accuracy Law for Sequential Machine Translation: the Case of Google Translate</h2>

Title: [An Empirical Accuracy Law for Sequential Machine Translation: the Case of Google Translate](https://arxiv.org/abs/2003.02817)

Authors: [Lucas Nunes Sequeira](https://arxiv.org/search/cs?searchtype=author&query=Sequeira%2C+L+N), [Bruno Moreschi](https://arxiv.org/search/cs?searchtype=author&query=Moreschi%2C+B), [Fabio Gagliardi Cozman](https://arxiv.org/search/cs?searchtype=author&query=Cozman%2C+F+G), [Bernardo Fontes](https://arxiv.org/search/cs?searchtype=author&query=Fontes%2C+B)

*(Submitted on 5 Mar 2020)*

> We have established, through empirical testing, a law that relates the number of translating hops to translation accuracy in sequential machine translation in Google Translate. Both accuracy and size decrease with the number of hops; the former displays a decrease closely following a power law. Such a law allows one to predict the behavior of translation chains that may be built as society increasingly depends on automated devices.

| Comments: | 11 pages, 8 figures (mostly graphs), a few mathematical functions and samples of the experiment |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Machine Learning (stat.ML) |
| Cite as:  | [arXiv:2003.02817](https://arxiv.org/abs/2003.02817) [cs.CL] |
|           | (or [arXiv:2003.02817v1](https://arxiv.org/abs/2003.02817v1) [cs.CL] for this version) |





# 2020-03-05

[Return to Index](#Index)



<h2 id="2020-03-05-1">1. Evaluating Low-Resource Machine Translation between Chinese and Vietnamese with Back-Translation</h2>

Title: [Evaluating Low-Resource Machine Translation between Chinese and Vietnamese with Back-Translation](https://arxiv.org/abs/2003.02197)

Authors: [Hongzheng Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+H), [Heyan Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+H)

*(Submitted on 4 Mar 2020)*

> Back translation (BT) has been widely used and become one of standard techniques for data augmentation in Neural Machine Translation (NMT), BT has proven to be helpful for improving the performance of translation effectively, especially for low-resource scenarios. While most works related to BT mainly focus on European languages, few of them study languages in other areas around the world. In this paper, we investigate the impacts of BT on Asia language translations between the extremely low-resource Chinese and Vietnamese language pair. We evaluate and compare the effects of different sizes of synthetic data on both NMT and Statistical Machine Translation (SMT) models for Chinese to Vietnamese and Vietnamese to Chinese, with character-based and word-based settings. Some conclusions from previous works are partially confirmed and we also draw some other interesting findings and conclusions, which are beneficial to understand BT further.

| Comments: | 10 pages, 5 tables and 4 figures                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:2003.02197](https://arxiv.org/abs/2003.02197) [cs.CL] |
|           | (or [arXiv:2003.02197v1](https://arxiv.org/abs/2003.02197v1) [cs.CL] for this version) |





# 2020-03-04

[Return to Index](#Index)



<h2 id="2020-03-04-1">1. Understanding the Prediction Mechanism of Sentiments by XAI Visualization</h2>

Title: [Understanding the Prediction Mechanism of Sentiments by XAI Visualization](https://arxiv.org/abs/2003.01425)

Authors: [Chaehan So](https://arxiv.org/search/cs?searchtype=author&query=So%2C+C)

*(Submitted on 3 Mar 2020)*

> People often rely on online reviews to make purchase decisions. The present work aimed to gain an understanding of a machine learning model's prediction mechanism by visualizing the effect of sentiments extracted from online hotel reviews with explainable AI (XAI) methodology. Study 1 used the extracted sentiments as features to predict the review ratings by five machine learning algorithms (knn, CART decision trees, support vector machines, random forests, gradient boosting machines) and identified random forests as best algorithm. Study 2 analyzed the random forests model by feature importance and revealed the sentiments joy, disgust, positive and negative as the most predictive features. Furthermore, the visualization of additive variable attributions and their prediction distribution showed correct prediction in direction and effect size for the 5-star rating but partially wrong direction and insufficient effect size for the 1-star rating. These prediction details were corroborated by a what-if analysis for the four top features. In conclusion, the prediction mechanism of a machine learning model can be uncovered by visualization of particular observations. Comparing instances of contrasting ground truth values can draw a differential picture of the prediction mechanism and inform decisions for model improvement.

| Comments: | This is the author's prefinal version be published in conference proceedings: 4th International Conference on Natural Language Processing and Information Retrieval, Sejong, South Korea, 26-28 June, 2020, ACM |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Human-Computer Interaction (cs.HC)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | [arXiv:2003.01425](https://arxiv.org/abs/2003.01425) [cs.HC] |
|           | (or [arXiv:2003.01425v1](https://arxiv.org/abs/2003.01425v1) [cs.HC] for this version) |



<h2 id="2020-03-04-2">2. XGPT: Cross-modal Generative Pre-Training for Image Captioning</h2>

Title: [XGPT: Cross-modal Generative Pre-Training for Image Captioning](https://arxiv.org/abs/2003.01473)

Authors: [Qiaolin Xia](https://arxiv.org/search/cs?searchtype=author&query=Xia%2C+Q), [Haoyang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+H), [Nan Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan%2C+N), [Dongdong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+D), [Lei Ji](https://arxiv.org/search/cs?searchtype=author&query=Ji%2C+L), [Zhifang Sui](https://arxiv.org/search/cs?searchtype=author&query=Sui%2C+Z), [Edward Cui](https://arxiv.org/search/cs?searchtype=author&query=Cui%2C+E), [Taroon Bharti](https://arxiv.org/search/cs?searchtype=author&query=Bharti%2C+T), [Ming Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+M)

*(Submitted on 3 Mar 2020)*

> While many BERT-based cross-modal pre-trained models produce excellent results on downstream understanding tasks like image-text retrieval and VQA, they cannot be applied to generation tasks directly. In this paper, we propose XGPT, a new method of Cross-modal Generative Pre-Training for Image Captioning that is designed to pre-train text-to-image caption generators through three novel generation tasks, including Image-conditioned Masked Language Modeling (IMLM), Image-conditioned Denoising Autoencoding (IDA), and Text-conditioned Image Feature Generation (TIFG). As a result, the pre-trained XGPT can be fine-tuned without any task-specific architecture modifications to create state-of-the-art models for image captioning. Experiments show that XGPT obtains new state-of-the-art results on the benchmark datasets, including COCO Captions and Flickr30k Captions. We also use XGPT to generate new image captions as data augmentation for the image retrieval task and achieve significant improvement on all recall metrics.

| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2003.01473](https://arxiv.org/abs/2003.01473) [cs.CL] |
|           | (or [arXiv:2003.01473v1](https://arxiv.org/abs/2003.01473v1) [cs.CL] for this version) |





# 2020-03-02

[Return to Index](#Index)



<h2 id="2020-03-02-1">1. Robust Unsupervised Neural Machine Translation with Adversarial Training</h2>

Title: [Robust Unsupervised Neural Machine Translation with Adversarial Training](https://arxiv.org/abs/2002.12549)

Authors: [Haipeng Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+H), [Rui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+R), [Kehai Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+K), [Masao Utiyama](https://arxiv.org/search/cs?searchtype=author&query=Utiyama%2C+M), [Eiichiro Sumita](https://arxiv.org/search/cs?searchtype=author&query=Sumita%2C+E), [Tiejun Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+T)

*(Submitted on 28 Feb 2020)*

> Unsupervised neural machine translation (UNMT) has recently attracted great interest in the machine translation community, achieving only slightly worse results than supervised neural machine translation. However, in real-world scenarios, there usually exists minor noise in the input sentence and the neural translation system is sensitive to the small perturbations in the input, leading to poor performance. In this paper, we first define two types of noises and empirically show the effect of these noisy data on UNMT performance. Moreover, we propose adversarial training methods to improve the robustness of UNMT in the noisy scenario. To the best of our knowledge, this paper is the first work to explore the robustness of UNMT. Experimental results on several language pairs show that our proposed methods substantially outperform conventional UNMT systems in the noisy scenario.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2002.12549](https://arxiv.org/abs/2002.12549) [cs.CL] |
|           | (or [arXiv:2002.12549v1](https://arxiv.org/abs/2002.12549v1) [cs.CL] for this version) |





<h2 id="2020-03-02-2">2. Modeling Future Cost for Neural Machine Translation</h2>

Title: [Modeling Future Cost for Neural Machine Translation](https://arxiv.org/abs/2002.12558)

Authors: [Chaoqun Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan%2C+C), [Kehai Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+K), [Rui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+R), [Masao Utiyama](https://arxiv.org/search/cs?searchtype=author&query=Utiyama%2C+M), [Eiichiro Sumita](https://arxiv.org/search/cs?searchtype=author&query=Sumita%2C+E), [Conghui Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+C), [Tiejun Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+T)

*(Submitted on 28 Feb 2020)*

> Existing neural machine translation (NMT) systems utilize sequence-to-sequence neural networks to generate target translation word by word, and then make the generated word at each time-step and the counterpart in the references as consistent as possible. However, the trained translation model tends to focus on ensuring the accuracy of the generated target word at the current time-step and does not consider its future cost which means the expected cost of generating the subsequent target translation (i.e., the next target word). To respond to this issue, we propose a simple and effective method to model the future cost of each target word for NMT systems. In detail, a time-dependent future cost is estimated based on the current generated target word and its contextual information to boost the training of the NMT model. Furthermore, the learned future context representation at the current time-step is used to help the generation of the next target word in the decoding. Experimental results on three widely-used translation datasets, including the WMT14 German-to-English, WMT14 English-to-French, and WMT17 Chinese-to-English, show that the proposed approach achieves significant improvements over strong Transformer-based NMT baseline.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2002.12558](https://arxiv.org/abs/2002.12558) [cs.CL] |
|           | (or [arXiv:2002.12558v1](https://arxiv.org/abs/2002.12558v1) [cs.CL] for this version) |





<h2 id="2020-03-02-3">3. TextBrewer: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing</h2>

Title: [TextBrewer: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing](https://arxiv.org/abs/2002.12620)

Authors: [Ziqing Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Yiming Cui](https://arxiv.org/search/cs?searchtype=author&query=Cui%2C+Y), [Zhipeng Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Z), [Wanxiang Che](https://arxiv.org/search/cs?searchtype=author&query=Che%2C+W), [Ting Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+T), [Shijin Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Guoping Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+G)

*(Submitted on 28 Feb 2020)*

> In this paper, we introduce TextBrewer, an open-source knowledge distillation toolkit designed for natural language processing. It works with different neural network models and supports various kinds of tasks, such as text classification, reading comprehension, sequence labeling. TextBrewer provides a simple and uniform workflow that enables quick setup of distillation experiments with highly flexible configurations. It offers a set of predefined distillation methods and can be extended with custom code. As a case study, we use TextBrewer to distill BERT on several typical NLP tasks. With simple configuration, we achieve results that are comparable with or even higher than the state-of-the-art performance. Our toolkit is available through: [this http URL](http://textbrewer.hfl-rc.com/)

| Comments: | 8 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Neural and Evolutionary Computing (cs.NE) |
| Cite as:  | [arXiv:2002.12620](https://arxiv.org/abs/2002.12620) [cs.CL] |
|           | (or [arXiv:2002.12620v1](https://arxiv.org/abs/2002.12620v1) [cs.CL] for this version) |





<h2 id="2020-03-02-4">4. Do all Roads Lead to Rome? Understanding the Role of Initialization in Iterative Back-Translation</h2>

Title: [Do all Roads Lead to Rome? Understanding the Role of Initialization in Iterative Back-Translation](https://arxiv.org/abs/2002.12867)

Authors: [Mikel Artetxe](https://arxiv.org/search/cs?searchtype=author&query=Artetxe%2C+M), [Gorka Labaka](https://arxiv.org/search/cs?searchtype=author&query=Labaka%2C+G), [Noe Casas](https://arxiv.org/search/cs?searchtype=author&query=Casas%2C+N), [Eneko Agirre](https://arxiv.org/search/cs?searchtype=author&query=Agirre%2C+E)

*(Submitted on 28 Feb 2020)*

> Back-translation provides a simple yet effective approach to exploit monolingual corpora in Neural Machine Translation (NMT). Its iterative variant, where two opposite NMT models are jointly trained by alternately using a synthetic parallel corpus generated by the reverse model, plays a central role in unsupervised machine translation. In order to start producing sound translations and provide a meaningful training signal to each other, existing approaches rely on either a separate machine translation system to warm up the iterative procedure, or some form of pre-training to initialize the weights of the model. In this paper, we analyze the role that such initialization plays in iterative back-translation. Is the behavior of the final system heavily dependent on it? Or does iterative back-translation converge to a similar solution given any reasonable initialization? Through a series of empirical experiments over a diverse set of warmup systems, we show that, although the quality of the initial system does affect final performance, its effect is relatively small, as iterative back-translation has a strong tendency to convergence to a similar solution. As such, the margin of improvement left for the initialization method is narrow, suggesting that future research should focus more on improving the iterative mechanism itself.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2002.12867](https://arxiv.org/abs/2002.12867) [cs.CL] |
|           | (or [arXiv:2002.12867v1](https://arxiv.org/abs/2002.12867v1) [cs.CL] for this version) |



