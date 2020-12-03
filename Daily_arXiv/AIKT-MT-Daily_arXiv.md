# Daily arXiv: Machine Translation - December, 2020

# Index


- [2020-12-03](#2020-12-03)

  - [1. Evaluating Explanations: How much do explanations from the teacher aid students?](#2020-12-03-1)
  - [2. How Can We Know When Language Models Know?](#2020-12-03-2)
  - [3. Interactive Teaching for Conversational AI](#2020-12-03-3)
- [2020-12-02](#2020-12-02)

  - [1. Modifying Memories in Transformer Models](#2020-12-02-1)
  - [2. An Enhanced Knowledge Injection Model for Commonsense Generation](#2020-12-02-2)
  - [3. CPM: A Large-scale Generative Chinese Pre-trained Language Model](#2020-12-02-3)
  - [4. Extracting Synonyms from Bilingual Dictionaries](#2020-12-02-4)
  - [5. Intrinsic analysis for dual word embedding space models](#2020-12-02-5)
- [2020-12-01](#2020-12-01)
  - [1. EdgeBERT: Optimizing On-Chip Inference for Multi-Task NLP](#2020-12-01-1)
  - [2. Understanding How BERT Learns to Identify Edits](#2020-12-01-2)
  - [3. Using Multiple Subwords to Improve English-Esperanto Automated Literary Translation Quality](#2020-12-01-3)
  - [4. Intrinsic Knowledge Evaluation on Chinese Language Models](#2020-12-01-4)
  - [5. Dynamic Curriculum Learning for Low-Resource Neural Machine Translation](#2020-12-01-5)
  - [6. A Simple and Effective Approach to Robust Unsupervised Bilingual Dictionary Induction](#2020-12-01-6)
  - [7. Machine Translation of Novels in the Age of Transformer](#2020-12-01-7)
  - [8. Multimodal Pretraining Unmasked: Unifying the Vision and Language BERTs](#2020-12-01-8)
- [Other Columns](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2020-12-03

[Return to Index](#Index)



<h2 id="2020-12-03-1">1. Evaluating Explanations: How much do explanations from the teacher aid students?</h2>

Title: [Evaluating Explanations: How much do explanations from the teacher aid students?](https://arxiv.org/abs/2012.00893)

Authors: [Danish Pruthi](https://arxiv.org/search/cs?searchtype=author&query=Pruthi%2C+D), [Bhuwan Dhingra](https://arxiv.org/search/cs?searchtype=author&query=Dhingra%2C+B), [Livio Baldini Soares](https://arxiv.org/search/cs?searchtype=author&query=Soares%2C+L+B), [Michael Collins](https://arxiv.org/search/cs?searchtype=author&query=Collins%2C+M), [Zachary C. Lipton](https://arxiv.org/search/cs?searchtype=author&query=Lipton%2C+Z+C), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G), [William W. Cohen](https://arxiv.org/search/cs?searchtype=author&query=Cohen%2C+W+W)

> While many methods purport to explain predictions by highlighting salient features, what precise aims these explanations serve and how to evaluate their utility are often unstated. In this work, we formalize the value of explanations using a student-teacher paradigm that measures the extent to which explanations improve student models in learning to simulate the teacher model on unseen examples for which explanations are unavailable. Student models incorporate explanations in training (but not prediction) procedures. Unlike many prior proposals to evaluate explanations, our approach cannot be easily gamed, enabling principled, scalable, and automatic evaluation of attributions. Using our framework, we compare multiple attribution methods and observe consistent and quantitative differences amongst them across multiple learning strategies.

| Comments: | Preprint                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2012.00893](https://arxiv.org/abs/2012.00893) [cs.CL]** |
|           | (or **[arXiv:2012.00893v1](https://arxiv.org/abs/2012.00893v1) [cs.CL]** for this version) |





<h2 id="2020-12-03-2">2. How Can We Know When Language Models Know?</h2>

Title: [How Can We Know When Language Models Know?](https://arxiv.org/abs/2012.00955)

Authors: [Zhengbao Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+Z), [Jun Araki](https://arxiv.org/search/cs?searchtype=author&query=Araki%2C+J), [Haibo Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+H), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G)

> Recent works have shown that language models (LM) capture different types of knowledge regarding facts or common sense. However, because no model is perfect, they still fail to provide appropriate answers in many cases. In this paper, we ask the question "how can we know when language models know, with confidence, the answer to a particular query?" We examine this question from the point of view of calibration, the property of a probabilistic model's predicted probabilities actually being well correlated with the probability of correctness. We first examine a state-of-the-art generative QA model, T5, and examine whether its probabilities are well calibrated, finding the answer is a relatively emphatic no. We then examine methods to calibrate such models to make their confidence scores correlate better with the likelihood of correctness through fine-tuning, post-hoc probability modification, or adjustment of the predicted outputs or inputs. Experiments on a diverse range of datasets demonstrate the effectiveness of our methods. We also perform analysis to study the strengths and limitations of these methods, shedding light on further improvements that may be made in methods for calibrating LMs.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.00955](https://arxiv.org/abs/2012.00955) [cs.CL]** |
|           | (or **[arXiv:2012.00955v1](https://arxiv.org/abs/2012.00955v1) [cs.CL]** for this version) |





<h2 id="2020-12-03-3">3. Interactive Teaching for Conversational AI</h2>

Title: [Interactive Teaching for Conversational AI](https://arxiv.org/abs/2012.00958)

Authors: [Qing Ping](https://arxiv.org/search/cs?searchtype=author&query=Ping%2C+Q), [Feiyang Niu](https://arxiv.org/search/cs?searchtype=author&query=Niu%2C+F), [Govind Thattai](https://arxiv.org/search/cs?searchtype=author&query=Thattai%2C+G), [Joel Chengottusseriyil](https://arxiv.org/search/cs?searchtype=author&query=Chengottusseriyil%2C+J), [Qiaozi Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+Q), [Aishwarya Reganti](https://arxiv.org/search/cs?searchtype=author&query=Reganti%2C+A), [Prashanth Rajagopal](https://arxiv.org/search/cs?searchtype=author&query=Rajagopal%2C+P), [Gokhan Tur](https://arxiv.org/search/cs?searchtype=author&query=Tur%2C+G), [Dilek Hakkani-Tur](https://arxiv.org/search/cs?searchtype=author&query=Hakkani-Tur%2C+D), [Prem Nataraja](https://arxiv.org/search/cs?searchtype=author&query=Nataraja%2C+P)

> Current conversational AI systems aim to understand a set of pre-designed requests and execute related actions, which limits them to evolve naturally and adapt based on human interactions. Motivated by how children learn their first language interacting with adults, this paper describes a new Teachable AI system that is capable of learning new language nuggets called concepts, directly from end users using live interactive teaching sessions. The proposed setup uses three models to: a) Identify gaps in understanding automatically during live conversational interactions, b) Learn the respective interpretations of such unknown concepts from live interactions with users, and c) Manage a classroom sub-dialogue specifically tailored for interactive teaching sessions. We propose state-of-the-art transformer based neural architectures of models, fine-tuned on top of pre-trained models, and show accuracy improvements on the respective components. We demonstrate that this method is very promising in leading way to build more adaptive and personalized language understanding models.

| Comments: | Accepted at Human in the Loop Dialogue Systems Workshop @NeurIPS 2020 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2012.00958](https://arxiv.org/abs/2012.00958) [cs.CL]** |
|           | (or **[arXiv:2012.00958v1](https://arxiv.org/abs/2012.00958v1) [cs.CL]** for this version) |







# 2020-12-02

[Return to Index](#Index)



<h2 id="2020-12-02-1">1. Modifying Memories in Transformer Models</h2>

Title: [Modifying Memories in Transformer Models](https://arxiv.org/abs/2012.00363)

Authors: [Chen Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+C), [Ankit Singh Rawat](https://arxiv.org/search/cs?searchtype=author&query=Rawat%2C+A+S), [Manzil Zaheer](https://arxiv.org/search/cs?searchtype=author&query=Zaheer%2C+M), [Srinadh Bhojanapalli](https://arxiv.org/search/cs?searchtype=author&query=Bhojanapalli%2C+S), [Daliang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+D), [Felix Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+F), [Sanjiv Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+S)

> Large Transformer models have achieved impressive performance in many natural language tasks. In particular, Transformer based language models have been shown to have great capabilities in encoding factual knowledge in their vast amount of parameters. While the tasks of improving the memorization and generalization of Transformers have been widely studied, it is not well known how to make transformers forget specific old facts and memorize new ones. In this paper, we propose a new task of \emph{explicitly modifying specific factual knowledge in Transformer models while ensuring the model performance does not degrade on the unmodified facts}. This task is useful in many scenarios, such as updating stale knowledge, protecting privacy, and eliminating unintended biases stored in the models. We benchmarked several approaches that provide natural baseline performances on this task. This leads to the discovery of key components of a Transformer model that are especially effective for knowledge modifications. The work also provides insights into the role that different training phases (such as pretraining and fine-tuning) play towards memorization and knowledge modification.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.00363](https://arxiv.org/abs/2012.00363) [cs.CL]** |
|           | (or **[arXiv:2012.00363v1](https://arxiv.org/abs/2012.00363v1) [cs.CL]** for this version) |





<h2 id="2020-12-02-2">2. An Enhanced Knowledge Injection Model for Commonsense Generation</h2>

Title: [An Enhanced Knowledge Injection Model for Commonsense Generation](https://arxiv.org/abs/2012.00366)

Authors: [Zhihao Fan](https://arxiv.org/search/cs?searchtype=author&query=Fan%2C+Z), [Yeyun Gong](https://arxiv.org/search/cs?searchtype=author&query=Gong%2C+Y), [Zhongyu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+Z), [Siyuan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Yameng Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+Y), [Jian Jiao](https://arxiv.org/search/cs?searchtype=author&query=Jiao%2C+J), [Xuanjing Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+X), [Nan Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan%2C+N), [Ruofei Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+R)

> Commonsense generation aims at generating plausible everyday scenario description based on a set of provided concepts. Digging the relationship of concepts from scratch is non-trivial, therefore, we retrieve prototypes from external knowledge to assist the understanding of the scenario for better description generation. We integrate two additional modules, namely position indicator and scaling module, into the pretrained encoder-decoder model for prototype modeling to enhance the knowledge injection procedure. We conduct experiment on CommonGen benchmark, and experimental results show that our method significantly improves the performance on all the metrics.

| Comments: | Accepted to COLING 2020                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2012.00366](https://arxiv.org/abs/2012.00366) [cs.CL]** |
|           | (or **[arXiv:2012.00366v1](https://arxiv.org/abs/2012.00366v1) [cs.CL]** for this version) |





<h2 id="2020-12-02-3">3. CPM: A Large-scale Generative Chinese Pre-trained Language Model</h2>

Title: [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413)

Authors: [Zhengyan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Xu Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+X), [Hao Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+H), [Pei Ke](https://arxiv.org/search/cs?searchtype=author&query=Ke%2C+P), [Yuxian Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+Y), [Deming Ye](https://arxiv.org/search/cs?searchtype=author&query=Ye%2C+D), [Yujia Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+Y), [Yusheng Su](https://arxiv.org/search/cs?searchtype=author&query=Su%2C+Y), [Haozhe Ji](https://arxiv.org/search/cs?searchtype=author&query=Ji%2C+H), [Jian Guan](https://arxiv.org/search/cs?searchtype=author&query=Guan%2C+J), [Fanchao Qi](https://arxiv.org/search/cs?searchtype=author&query=Qi%2C+F), [Xiaozhi Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Yanan Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+Y), [Guoyang Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+G), [Huanqi Cao](https://arxiv.org/search/cs?searchtype=author&query=Cao%2C+H), [Shengqi Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+S), [Daixuan Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+D), [Zhenbo Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+Z), [Zhiyuan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Minlie Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+M), [Wentao Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+W), [Jie Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+J), [Juanzi Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J), [Xiaoyan Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+X), [Maosong Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+M)

> Pre-trained Language Models (PLMs) have proven to be beneficial for various downstream NLP tasks. Recently, GPT-3, with 175 billion parameters and 570GB training data, drew a lot of attention due to the capacity of few-shot (even zero-shot) learning. However, applying GPT-3 to address Chinese NLP tasks is still challenging, as the training corpus of GPT-3 is primarily English, and the parameters are not publicly available. In this technical report, we release the Chinese Pre-trained Language Model (CPM) with generative pre-training on large-scale Chinese training data. To the best of our knowledge, CPM, with 2.6 billion parameters and 100GB Chinese training data, is the largest Chinese pre-trained language model, which could facilitate several downstream Chinese NLP tasks, such as conversation, essay generation, cloze test, and language understanding. Extensive experiments demonstrate that CPM achieves strong performance on many NLP tasks in the settings of few-shot (even zero-shot) learning. The code and parameters are available at [this https URL](https://github.com/TsinghuaAI/CPM-Generate).

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.00413](https://arxiv.org/abs/2012.00413) [cs.CL]** |
|           | (or **[arXiv:2012.00413v1](https://arxiv.org/abs/2012.00413v1) [cs.CL]** for this version) |





<h2 id="2020-12-02-4">4. Extracting Synonyms from Bilingual Dictionaries</h2>

Title: [Extracting Synonyms from Bilingual Dictionaries](https://arxiv.org/abs/2012.00600)

Authors: [Mustafa Jarrar](https://arxiv.org/search/cs?searchtype=author&query=Jarrar%2C+M), [Eman Karajah](https://arxiv.org/search/cs?searchtype=author&query=Karajah%2C+E), [Muhammad Khalifa](https://arxiv.org/search/cs?searchtype=author&query=Khalifa%2C+M), [Khaled Shaalan](https://arxiv.org/search/cs?searchtype=author&query=Shaalan%2C+K)

> We present our progress in developing a novel algorithm to extract synonyms from bilingual dictionaries. Identification and usage of synonyms play a significant role in improving the performance of information access applications. The idea is to construct a translation graph from translation pairs, then to extract and consolidate cyclic paths to form bilingual sets of synonyms. The initial evaluation of this algorithm illustrates promising results in extracting Arabic-English bilingual synonyms. In the evaluation, we first converted the synsets in the Arabic WordNet into translation pairs (i.e., losing word-sense memberships). Next, we applied our algorithm to rebuild these synsets. We compared the original and extracted synsets obtaining an F-Measure of 82.3% and 82.1% for Arabic and English synsets extraction, respectively.

| Comments: | In Proceedings - 11th International Global Wordnet Conference (GWC2021). Global Wordnet Association (2021) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Information Retrieval (cs.IR) |
| Cite as:  | **[arXiv:2012.00600](https://arxiv.org/abs/2012.00600) [cs.CL]** |
|           | (or **[arXiv:2012.00600v1](https://arxiv.org/abs/2012.00600v1) [cs.CL]** for this version) |





<h2 id="2020-12-02-5">5. Intrinsic analysis for dual word embedding space models</h2>

Title: [Intrinsic analysis for dual word embedding space models](https://arxiv.org/abs/2012.00728)

Authors: [Mohit Mayank](https://arxiv.org/search/cs?searchtype=author&query=Mayank%2C+M)

> Recent word embeddings techniques represent words in a continuous vector space, moving away from the atomic and sparse representations of the past. Each such technique can further create multiple varieties of embeddings based on different settings of hyper-parameters like embedding dimension size, context window size and training method. One additional variety appears when we especially consider the Dual embedding space techniques which generate not one but two-word embeddings as output. This gives rise to an interesting question - "is there one or a combination of the two word embeddings variety, which works better for a specific task?". This paper tries to answer this question by considering all of these variations. Herein, we compare two classical embedding methods belonging to two different methodologies - Word2Vec from window-based and Glove from count-based. For an extensive evaluation after considering all variations, a total of 84 different models were compared against semantic, association and analogy evaluations tasks which are made up of 9 open-source linguistics datasets. The final Word2vec reports showcase the preference of non-default model for 2 out of 3 tasks. In case of Glove, non-default models outperform in all 3 evaluation tasks.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.00728](https://arxiv.org/abs/2012.00728) [cs.CL]** |
|           | (or **[arXiv:2012.00728v1](https://arxiv.org/abs/2012.00728v1) [cs.CL]** for this version) |





# 2020-12-01

[Return to Index](#Index)



<h2 id="2020-12-01-1">1. EdgeBERT: Optimizing On-Chip Inference for Multi-Task NLP</h2>

Title: [EdgeBERT: Optimizing On-Chip Inference for Multi-Task NLP](https://arxiv.org/abs/2011.14203)

Authors: [Thierry Tambe](https://arxiv.org/search/cs?searchtype=author&query=Tambe%2C+T), [Coleman Hooper](https://arxiv.org/search/cs?searchtype=author&query=Hooper%2C+C), [Lillian Pentecost](https://arxiv.org/search/cs?searchtype=author&query=Pentecost%2C+L), [En-Yu Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+E), [Marco Donato](https://arxiv.org/search/cs?searchtype=author&query=Donato%2C+M), [Victor Sanh](https://arxiv.org/search/cs?searchtype=author&query=Sanh%2C+V), [Alexander M. Rush](https://arxiv.org/search/cs?searchtype=author&query=Rush%2C+A+M), [David Brooks](https://arxiv.org/search/cs?searchtype=author&query=Brooks%2C+D), [Gu-Yeon Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+G)

> Transformer-based language models such as BERT provide significant accuracy improvement to a multitude of natural language processing (NLP) tasks. However, their hefty computational and memory demands make them challenging to deploy to resource-constrained edge platforms with strict latency requirements.
> We present EdgeBERT an in-depth and principled algorithm and hardware design methodology to achieve minimal latency and energy consumption on multi-task NLP inference. Compared to the ALBERT baseline, we achieve up to 2.4x and 13.4x inference latency and memory savings, respectively, with less than 1%-pt drop in accuracy on several GLUE benchmarks by employing a calibrated combination of 1) entropy-based early stopping, 2) adaptive attention span, 3) movement and magnitude pruning, and 4) floating-point quantization.
> Furthermore, in order to maximize the benefits of these algorithms in always-on and intermediate edge computing settings, we specialize a scalable hardware architecture wherein floating-point bit encodings of the shareable multi-task embedding parameters are stored in high-density non-volatile memory. Altogether, EdgeBERT enables fully on-chip inference acceleration of NLP workloads with 5.2x, and 157x lower energy than that of an un-optimized accelerator and CUDA adaptations on an Nvidia Jetson Tegra X2 mobile GPU, respectively.

| Comments: | 11 pages plus references                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Hardware Architecture (cs.AR)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2011.14203](https://arxiv.org/abs/2011.14203) [cs.AR]** |
|           | (or **[arXiv:2011.14203v1](https://arxiv.org/abs/2011.14203v1) [cs.AR]** for this version) |





<h2 id="2020-12-01-2">2. Understanding How BERT Learns to Identify Edits</h2>

Title: [Understanding How BERT Learns to Identify Edits](https://arxiv.org/abs/2011.14039)

Authors: [Samuel Stevens](https://arxiv.org/search/cs?searchtype=author&query=Stevens%2C+S), [Yu Su](https://arxiv.org/search/cs?searchtype=author&query=Su%2C+Y)

> Pre-trained transformer language models such as BERT are ubiquitous in NLP research, leading to work on understanding how and why these models work. Attention mechanisms have been proposed as a means of interpretability with varying conclusions. We propose applying BERT-based models to a sequence classification task and using the data set's labeling schema to measure each model's interpretability. We find that classification performance scores do not always correlate with interpretability. Despite this, BERT's attention weights are interpretable for over 70% of examples.

| Comments: | 8 pages, 11 figures. A work in progress                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2011.14039](https://arxiv.org/abs/2011.14039) [cs.CL]** |
|           | (or **[arXiv:2011.14039v1](https://arxiv.org/abs/2011.14039v1) [cs.CL]** for this version) |







<h2 id="2020-12-01-3">3. Using Multiple Subwords to Improve English-Esperanto Automated Literary Translation Quality</h2>

Title: [Using Multiple Subwords to Improve English-Esperanto Automated Literary Translation Quality](https://arxiv.org/abs/2011.14190)

Authors: [Alberto Poncelas](https://arxiv.org/search/cs?searchtype=author&query=Poncelas%2C+A), [Jan Buts](https://arxiv.org/search/cs?searchtype=author&query=Buts%2C+J), [James Hadley](https://arxiv.org/search/cs?searchtype=author&query=Hadley%2C+J), [Andy Way](https://arxiv.org/search/cs?searchtype=author&query=Way%2C+A)

> Building Machine Translation (MT) systems for low-resource languages remains challenging. For many language pairs, parallel data are not widely available, and in such cases MT models do not achieve results comparable to those seen with high-resource languages.
> When data are scarce, it is of paramount importance to make optimal use of the limited material available. To that end, in this paper we propose employing the same parallel sentences multiple times, only changing the way the words are split each time. For this purpose we use several Byte Pair Encoding models, with various merge operations used in their configuration.
> In our experiments, we use this technique to expand the available data and improve an MT system involving a low-resource language pair, namely English-Esperanto.
> As an additional contribution, we made available a set of English-Esperanto parallel data in the literary domain.

| Subjects:          | **Computation and Language (cs.CL)**                         |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | The 3rd Workshop on Technologies for MT of Low Resource Languages (LoResMT 2020) |
| Cite as:           | **[arXiv:2011.14190](https://arxiv.org/abs/2011.14190) [cs.CL]** |
|                    | (or **[arXiv:2011.14190v1](https://arxiv.org/abs/2011.14190v1) [cs.CL]** for this version) |







<h2 id="2020-12-01-4">4. Intrinsic Knowledge Evaluation on Chinese Language Models</h2>

Title: [Intrinsic Knowledge Evaluation on Chinese Language Models](https://arxiv.org/abs/2011.14277)

Authors: [Zhiruo Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Z), [Renfen Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+R)

> Recent NLP tasks have benefited a lot from pre-trained language models (LM) since they are able to encode knowledge of various aspects. However, current LM evaluations focus on downstream performance, hence lack to comprehensively inspect in which aspect and to what extent have they encoded knowledge. This paper addresses both queries by proposing four tasks on syntactic, semantic, commonsense, and factual knowledge, aggregating to a total of 39,308 questions covering both linguistic and world knowledge in Chinese. Throughout experiments, our probes and knowledge data prove to be a reliable benchmark for evaluating pre-trained Chinese LMs. Our work is publicly available at [this https URL](https://github.com/ZhiruoWang/ChnEval).

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2011.14277](https://arxiv.org/abs/2011.14277) [cs.CL]** |
|           | (or **[arXiv:2011.14277v1](https://arxiv.org/abs/2011.14277v1) [cs.CL]** for this version) |







<h2 id="2020-12-01-5">5. Dynamic Curriculum Learning for Low-Resource Neural Machine Translation</h2>

Title: [Dynamic Curriculum Learning for Low-Resource Neural Machine Translation](https://arxiv.org/abs/2011.14608)

Authors: [Chen Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+C), [Bojie Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+B), [Yufan Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+Y), [Kai Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+K), [Zeyang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Z), [Shen Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Qi Ju](https://arxiv.org/search/cs?searchtype=author&query=Ju%2C+Q), [Tong Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+T), [Jingbo Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+J)

> Large amounts of data has made neural machine translation (NMT) a big success in recent years. But it is still a challenge if we train these models on small-scale corpora. In this case, the way of using data appears to be more important. Here, we investigate the effective use of training data for low-resource NMT. In particular, we propose a dynamic curriculum learning (DCL) method to reorder training samples in training. Unlike previous work, we do not use a static scoring function for reordering. Instead, the order of training samples is dynamically determined in two ways - loss decline and model competence. This eases training by highlighting easy samples that the current model has enough competence to learn. We test our DCL method in a Transformer-based system. Experimental results show that DCL outperforms several strong baselines on three low-resource machine translation benchmarks and different sized data of WMT' 16 En-De.

| Comments: | COLING 2020                                                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2011.14608](https://arxiv.org/abs/2011.14608) [cs.CL]** |
|           | (or **[arXiv:2011.14608v1](https://arxiv.org/abs/2011.14608v1) [cs.CL]** for this version) |







<h2 id="2020-12-01-6">6. A Simple and Effective Approach to Robust Unsupervised Bilingual Dictionary Induction</h2>

Title: [A Simple and Effective Approach to Robust Unsupervised Bilingual Dictionary Induction](https://arxiv.org/abs/2011.14874)

Authors: [Yanyang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Yingfeng Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+Y), [Ye Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+Y), [Quan Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+Q), [Huizhen Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+H), [Shujian Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Tong Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+T), [Jingbo Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+J)

> Unsupervised Bilingual Dictionary Induction methods based on the initialization and the self-learning have achieved great success in similar language pairs, e.g., English-Spanish. But they still fail and have an accuracy of 0% in many distant language pairs, e.g., English-Japanese. In this work, we show that this failure results from the gap between the actual initialization performance and the minimum initialization performance for the self-learning to succeed. We propose Iterative Dimension Reduction to bridge this gap. Our experiments show that this simple method does not hamper the performance of similar language pairs and achieves an accuracy of 13.64~55.53% between English and four distant languages, i.e., Chinese, Japanese, Vietnamese and Thai.

| Comments: | Accepted by COLING2020                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2011.14874](https://arxiv.org/abs/2011.14874) [cs.CL]** |
|           | (or **[arXiv:2011.14874v1](https://arxiv.org/abs/2011.14874v1) [cs.CL]** for this version) |







<h2 id="2020-12-01-7">7. Machine Translation of Novels in the Age of Transformer</h2>

Title: [Machine Translation of Novels in the Age of Transformer](https://arxiv.org/abs/2011.14979)

Authors: [Antonio Toral](https://arxiv.org/search/cs?searchtype=author&query=Toral%2C+A), [Antoni Oliver](https://arxiv.org/search/cs?searchtype=author&query=Oliver%2C+A), [Pau Ribas Ballestín](https://arxiv.org/search/cs?searchtype=author&query=Ballestín%2C+P+R)

> In this chapter we build a machine translation (MT) system tailored to the literary domain, specifically to novels, based on the state-of-the-art architecture in neural MT (NMT), the Transformer (Vaswani et al., 2017), for the translation direction English-to-Catalan. Subsequently, we assess to what extent such a system can be useful by evaluating its translations, by comparing this MT system against three other systems (two domain-specific systems under the recurrent and phrase-based paradigms and a popular generic on-line system) on three evaluations. The first evaluation is automatic and uses the most-widely used automatic evaluation metric, BLEU. The two remaining evaluations are manual and they assess, respectively, preference and amount of post-editing required to make the translation error-free. As expected, the domain-specific Transformer-based system outperformed the three other systems in all the three evaluations conducted, in all cases by a large margin.

| Comments: | Chapter published in the book Maschinelle Übersetzung für Übersetzungsprofis (pp. 276-295). Jörg Porsiel (Ed.), BDÜ Fachverlag, 2020. ISBN 978-3-946702-09-2 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2011.14979](https://arxiv.org/abs/2011.14979) [cs.CL]** |
|           | (or **[arXiv:2011.14979v1](https://arxiv.org/abs/2011.14979v1) [cs.CL]** for this version) |





<h2 id="2020-12-01-8">8. Multimodal Pretraining Unmasked: Unifying the Vision and Language BERTs</h2>

Title: [Multimodal Pretraining Unmasked: Unifying the Vision and Language BERTs](https://arxiv.org/abs/2011.15124)

Authors: [Emanuele Bugliarello](https://arxiv.org/search/cs?searchtype=author&query=Bugliarello%2C+E), [Ryan Cotterell](https://arxiv.org/search/cs?searchtype=author&query=Cotterell%2C+R), [Naoaki Okazaki](https://arxiv.org/search/cs?searchtype=author&query=Okazaki%2C+N), [Desmond Elliott](https://arxiv.org/search/cs?searchtype=author&query=Elliott%2C+D)

> Large-scale pretraining and task-specific fine-tuning is now the standard methodology for many tasks in computer vision and natural language processing. Recently, a multitude of methods have been proposed for pretraining vision and language BERTs to tackle challenges at the intersection of these two key areas of AI. These models can be categorized into either single-stream or dual-stream encoders. We study the differences between these two categories, and show how they can be unified under a single theoretical framework. We then conduct controlled experiments to discern the empirical differences between five V&L BERTs. Our experiments show that training data and hyperparameters are responsible for most of the differences between the reported results, but they also reveal that the embedding layer plays a crucial role in these massive models.

| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2011.15124](https://arxiv.org/abs/2011.15124) [cs.CL]** |
|           | (or **[arXiv:2011.15124v1](https://arxiv.org/abs/2011.15124v1) [cs.CL]** for this version) |



