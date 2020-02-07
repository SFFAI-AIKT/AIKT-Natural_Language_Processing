# Daily arXiv: Machine Translation - Jan., 2020

# Index


- [2020-02-06](#2020-02-06)

  - [1. Multilingual acoustic word embedding models for processing zero-resource languages](#2020-02-06-1)
  - [2. Irony Detection in a Multilingual Context](#2020-02-06-2)
- [2020-02-05](#2020-02-05)

  - [1. CoVoST: A Diverse Multilingual Speech-To-Text Translation Corpus](#2020-02-05-1)
- [2020-02-04](#2020-02-04)

  - [1. Unsupervised Bilingual Lexicon Induction Across Writing Systems](#2020-02-04-1)
  - [2. Citation Text Generation](#2020-02-04-2)
  - [3. Unsupervised Multilingual Alignment using Wasserstein Barycenter](#2020-02-04-3)
  - [4. Joint Contextual Modeling for ASR Correction and Language Understanding](#2020-02-04-4)
  - [5. FastWordBug: A Fast Method To Generate Adversarial Text Against NLP Applications](#2020-02-04-5)
  - [6. Massively Multilingual Document Alignment with Cross-lingual Sentence-Mover's Distance](#2020-02-04-6)
- [2020-02-03](#2020-02-03)

  - [1. Self-Adversarial Learning with Comparative Discrimination for Text Generation](#2020-02-03-1)
  - [2. Teaching Machines to Converse](#2020-02-03-2)
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



# 2020-02-06

[Return to Index](#Index)



<h2 id="2020-02-06-1">1. Multilingual acoustic word embedding models for processing zero-resource languages</h2>

Title: [Multilingual acoustic word embedding models for processing zero-resource languages](https://arxiv.org/abs/2002.02109)

Authors: [Herman Kamper](https://arxiv.org/search/cs?searchtype=author&query=Kamper%2C+H), [Yevgen Matusevych](https://arxiv.org/search/cs?searchtype=author&query=Matusevych%2C+Y), [Sharon Goldwater](https://arxiv.org/search/cs?searchtype=author&query=Goldwater%2C+S)

*(Submitted on 6 Feb 2020)*

> Acoustic word embeddings are fixed-dimensional representations of variable-length speech segments. In settings where unlabelled speech is the only available resource, such embeddings can be used in "zero-resource" speech search, indexing and discovery systems. Here we propose to train a single supervised embedding model on labelled data from multiple well-resourced languages and then apply it to unseen zero-resource languages. For this transfer learning approach, we consider two multilingual recurrent neural network models: a discriminative classifier trained on the joint vocabularies of all training languages, and a correspondence autoencoder trained to reconstruct word pairs. We test these using a word discrimination task on six target zero-resource languages. When trained on seven well-resourced languages, both models perform similarly and outperform unsupervised models trained on the zero-resource languages. With just a single training language, the second model works better, but performance depends more on the particular training--testing language pair.

| Comments: | 5 pages, 4 figures, 1 table; accepted to ICASSP 2020. arXiv admin note: text overlap with [arXiv:1811.00403](https://arxiv.org/abs/1811.00403) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Audio and Speech Processing (eess.AS) |
| Cite as:  | [arXiv:2002.02109](https://arxiv.org/abs/2002.02109) [cs.CL] |
|           | (or [arXiv:2002.02109v1](https://arxiv.org/abs/2002.02109v1) [cs.CL] for this version) |





<h2 id="2020-02-06-2">2. Irony Detection in a Multilingual Context</h2>

Title: [Irony Detection in a Multilingual Context](https://arxiv.org/abs/2002.02427)

Authors: [Bilal Ghanem](https://arxiv.org/search/cs?searchtype=author&query=Ghanem%2C+B), [Jihen Karoui](https://arxiv.org/search/cs?searchtype=author&query=Karoui%2C+J), [Farah Benamara](https://arxiv.org/search/cs?searchtype=author&query=Benamara%2C+F), [Paolo Rosso](https://arxiv.org/search/cs?searchtype=author&query=Rosso%2C+P), [Véronique Moriceau](https://arxiv.org/search/cs?searchtype=author&query=Moriceau%2C+V)

*(Submitted on 6 Feb 2020)*

> This paper proposes the first multilingual (French, English and Arabic) and multicultural (Indo-European languages vs. less culturally close languages) irony detection system. We employ both feature-based models and neural architectures using monolingual word representation. We compare the performance of these systems with state-of-the-art systems to identify their capabilities. We show that these monolingual models trained separately on different languages using multilingual word representation or text-based features can open the door to irony detection in languages that lack of annotated data for irony.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2002.02427](https://arxiv.org/abs/2002.02427) [cs.CL] |
|           | (or [arXiv:2002.02427v1](https://arxiv.org/abs/2002.02427v1) [cs.CL] for this version) |



# 2020-02-05

[Return to Index](#Index)



<h2 id="2020-02-05-1">1. CoVoST: A Diverse Multilingual Speech-To-Text Translation Corpus</h2>

Title: [CoVoST: A Diverse Multilingual Speech-To-Text Translation Corpus](https://arxiv.org/abs/2002.01320)

Authors: [Changhan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Juan Pino](https://arxiv.org/search/cs?searchtype=author&query=Pino%2C+J), [Anne Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+A), [Jiatao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+J)

*(Submitted on 4 Feb 2020)*

> Spoken language translation has recently witnessed a resurgence in popularity, thanks to the development of end-to-end models and the creation of new corpora, such as Augmented LibriSpeech and MuST-C. Existing datasets involve language pairs with English as a source language, involve very specific domains or are low resource. We introduce CoVoST, a multilingual speech-to-text translation corpus from 11 languages into English, diversified with over 11,000 speakers and over 60 accents. We describe the dataset creation methodology and provide empirical evidence of the quality of the data. We also provide initial benchmarks, including, to our knowledge, the first end-to-end many-to-one multilingual models for spoken language translation. CoVoST is released under CC0 license and free to use. We also provide additional evaluation data derived from Tatoeba under CC licenses.

| Comments: | Submitted to LREC 2020                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:2002.01320](https://arxiv.org/abs/2002.01320) [cs.CL] |
|           | (or [arXiv:2002.01320v1](https://arxiv.org/abs/2002.01320v1) [cs.CL] for this version) |



# 2020-02-04

[Return to Index](#Index)



<h2 id="2020-02-04-1">1. Unsupervised Bilingual Lexicon Induction Across Writing Systems</h2>

Title: [Unsupervised Bilingual Lexicon Induction Across Writing Systems](https://arxiv.org/abs/2002.00037)

Authors: [Parker Riley](https://arxiv.org/search/cs?searchtype=author&query=Riley%2C+P), [Daniel Gildea](https://arxiv.org/search/cs?searchtype=author&query=Gildea%2C+D)

*(Submitted on 31 Jan 2020)*

> Recent embedding-based methods in unsupervised bilingual lexicon induction have shown good results, but generally have not leveraged orthographic (spelling) information, which can be helpful for pairs of related languages. This work augments a state-of-the-art method with orthographic features, and extends prior work in this space by proposing methods that can learn and utilize orthographic correspondences even between languages with different scripts. We demonstrate this by experimenting on three language pairs with different scripts and varying degrees of lexical similarity.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2002.00037](https://arxiv.org/abs/2002.00037) [cs.CL] |
|           | (or [arXiv:2002.00037v1](https://arxiv.org/abs/2002.00037v1) [cs.CL] for this version) |





<h2 id="2020-02-04-2">2. Citation Text Generation</h2>

Title: [Citation Text Generation](https://arxiv.org/abs/2002.00317)

Authors: [Kelvin Luu](https://arxiv.org/search/cs?searchtype=author&query=Luu%2C+K), [Rik Koncel-Kedziorski](https://arxiv.org/search/cs?searchtype=author&query=Koncel-Kedziorski%2C+R), [Kyle Lo](https://arxiv.org/search/cs?searchtype=author&query=Lo%2C+K), [Isabel Cachola](https://arxiv.org/search/cs?searchtype=author&query=Cachola%2C+I), [Noah A. Smith](https://arxiv.org/search/cs?searchtype=author&query=Smith%2C+N+A)

*(Submitted on 2 Feb 2020)*

> We introduce the task of citation text generation: given a pair of scientific documents, explain their relationship in natural language text in the manner of a citation from one text to the other. This task encourages systems to learn rich relationships between scientific texts and to express them concretely in natural language. Models for citation text generation will require robust document understanding including the capacity to quickly adapt to new vocabulary and to reason about document content. We believe this challenging direction of research will benefit high-impact applications such as automatic literature review or scientific writing assistance systems. In this paper we establish the task of citation text generation with a standard evaluation corpus and explore several baseline models.

| Comments: | 10 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:2002.00317](https://arxiv.org/abs/2002.00317) [cs.CL] |
|           | (or [arXiv:2002.00317v1](https://arxiv.org/abs/2002.00317v1) [cs.CL] for this version) |





<h2 id="2020-02-04-3">3. Unsupervised Multilingual Alignment using Wasserstein Barycenter</h2>

Title: [Unsupervised Multilingual Alignment using Wasserstein Barycenter](https://arxiv.org/abs/2002.00743)

Authors: [Xin Lian](https://arxiv.org/search/cs?searchtype=author&query=Lian%2C+X), [Kshitij Jain](https://arxiv.org/search/cs?searchtype=author&query=Jain%2C+K), [Jakub Truszkowski](https://arxiv.org/search/cs?searchtype=author&query=Truszkowski%2C+J), [Pascal Poupart](https://arxiv.org/search/cs?searchtype=author&query=Poupart%2C+P), [Yaoliang Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+Y)

*(Submitted on 28 Jan 2020)*

> We study unsupervised multilingual alignment, the problem of finding word-to-word translations between multiple languages without using any parallel data. One popular strategy is to reduce multilingual alignment to the much simplified bilingual setting, by picking one of the input languages as the pivot language that we transit through. However, it is well-known that transiting through a poorly chosen pivot language (such as English) may severely degrade the translation quality, since the assumed transitive relations among all pairs of languages may not be enforced in the training process. Instead of going through a rather arbitrarily chosen pivot language, we propose to use the Wasserstein barycenter as a more informative ''mean'' language: it encapsulates information from all languages and minimizes all pairwise transportation costs. We evaluate our method on standard benchmarks and demonstrate state-of-the-art performances.

| Comments: | Work in progress; comments welcome!                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Machine Learning (stat.ML) |
| Cite as:  | [arXiv:2002.00743](https://arxiv.org/abs/2002.00743) [cs.CL] |
|           | (or [arXiv:2002.00743v1](https://arxiv.org/abs/2002.00743v1) [cs.CL] for this version) |





<h2 id="2020-02-04-4">4. Joint Contextual Modeling for ASR Correction and Language Understanding</h2>

Title: [Joint Contextual Modeling for ASR Correction and Language Understanding](https://arxiv.org/abs/2002.00750)

Authors: [Yue Weng](https://arxiv.org/search/cs?searchtype=author&query=Weng%2C+Y), [Sai Sumanth Miryala](https://arxiv.org/search/cs?searchtype=author&query=Miryala%2C+S+S), [Chandra Khatri](https://arxiv.org/search/cs?searchtype=author&query=Khatri%2C+C), [Runze Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+R), [Huaixiu Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+H), [Piero Molino](https://arxiv.org/search/cs?searchtype=author&query=Molino%2C+P), [Mahdi Namazifar](https://arxiv.org/search/cs?searchtype=author&query=Namazifar%2C+M), [Alexandros Papangelis](https://arxiv.org/search/cs?searchtype=author&query=Papangelis%2C+A), [Hugh Williams](https://arxiv.org/search/cs?searchtype=author&query=Williams%2C+H), [Franziska Bell](https://arxiv.org/search/cs?searchtype=author&query=Bell%2C+F), [Gokhan Tur](https://arxiv.org/search/cs?searchtype=author&query=Tur%2C+G)

*(Submitted on 28 Jan 2020)*

> The quality of automatic speech recognition (ASR) is critical to Dialogue Systems as ASR errors propagate to and directly impact downstream tasks such as language understanding (LU). In this paper, we propose multi-task neural approaches to perform contextual language correction on ASR outputs jointly with LU to improve the performance of both tasks simultaneously. To measure the effectiveness of this approach we used a public benchmark, the 2nd Dialogue State Tracking (DSTC2) corpus. As a baseline approach, we trained task-specific Statistical Language Models (SLM) and fine-tuned state-of-the-art Generalized Pre-training (GPT) Language Model to re-rank the n-best ASR hypotheses, followed by a model to identify the dialog act and slots. i) We further trained ranker models using GPT and Hierarchical CNN-RNN models with discriminatory losses to detect the best output given n-best hypotheses. We extended these ranker models to first select the best ASR output and then identify the dialogue act and slots in an end to end fashion. ii) We also proposed a novel joint ASR error correction and LU model, a word confusion pointer network (WCN-Ptr) with multi-head self-attention on top, which consumes the word confusions populated from the n-best. We show that the error rates of off the shelf ASR and following LU systems can be reduced significantly by 14% relative with joint models trained using small amounts of in-domain data.

| Comments: | Accepted at IEEE ICASSP 2020                                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | [arXiv:2002.00750](https://arxiv.org/abs/2002.00750) [cs.CL] |
|           | (or [arXiv:2002.00750v1](https://arxiv.org/abs/2002.00750v1) [cs.CL] for this version) |





<h2 id="2020-02-04-5">5. FastWordBug: A Fast Method To Generate Adversarial Text Against NLP Applications</h2>

Title: [FastWordBug: A Fast Method To Generate Adversarial Text Against NLP Applications](https://arxiv.org/abs/2002.00760)

Authors: [Dou Goodman](https://arxiv.org/search/cs?searchtype=author&query=Goodman%2C+D), [Lv Zhonghou](https://arxiv.org/search/cs?searchtype=author&query=Zhonghou%2C+L), [Wang minghua](https://arxiv.org/search/cs?searchtype=author&query=minghua%2C+W)

*(Submitted on 31 Jan 2020)*

> In this paper, we present a novel algorithm, FastWordBug, to efficiently generate small text perturbations in a black-box setting that forces a sentiment analysis or text classification mode to make an incorrect prediction. By combining the part of speech attributes of words, we propose a scoring method that can quickly identify important words that affect text classification. We evaluate FastWordBug on three real-world text datasets and two state-of-the-art machine learning models under black-box setting. The results show that our method can significantly reduce the accuracy of the model, and at the same time, we can call the model as little as possible, with the highest attack efficiency. We also attack two popular real-world cloud services of NLP, and the results show that our method works as well.

| Subjects: | **Computation and Language (cs.CL)**; Cryptography and Security (cs.CR) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2002.00760](https://arxiv.org/abs/2002.00760) [cs.CL] |
|           | (or [arXiv:2002.00760v1](https://arxiv.org/abs/2002.00760v1) [cs.CL] for this version) |





<h2 id="2020-02-04-6">6. Massively Multilingual Document Alignment with Cross-lingual Sentence-Mover's Distance</h2>

Title: [Massively Multilingual Document Alignment with Cross-lingual Sentence-Mover's Distance](https://arxiv.org/abs/2002.00761)

Authors: [Ahmed El-Kishky](https://arxiv.org/search/cs?searchtype=author&query=El-Kishky%2C+A), [Francisco Guzmán](https://arxiv.org/search/cs?searchtype=author&query=Guzmán%2C+F)

*(Submitted on 31 Jan 2020)*

> Cross-lingual document alignment aims to identify pairs of documents in two distinct languages that are of comparable content or translations of each other. Such aligned data can be used for a variety of NLP tasks from training cross-lingual representations to mining parallel bitexts for machine translation training. In this paper we develop an unsupervised scoring function that leverages cross-lingual sentence embeddings to compute the semantic distance between documents in different languages. These semantic distances are then used to guide a document alignment algorithm to properly pair cross-lingual web documents across a variety of low, mid, and high-resource language pairs. Recognizing that our proposed scoring function and other state of the art methods are computationally intractable for long web documents, we utilize a more tractable greedy algorithm that performs comparably. We experimentally demonstrate that our distance metric performs better alignment than current baselines outperforming them by 7% on high-resource language pairs, 15% on mid-resource language pairs, and 22% on low-resource language pairs

| Subjects: | **Computation and Language (cs.CL)**; Information Retrieval (cs.IR); Machine Learning (cs.LG); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2002.00761](https://arxiv.org/abs/2002.00761) [cs.CL] |
|           | (or [arXiv:2002.00761v1](https://arxiv.org/abs/2002.00761v1) [cs.CL] for this version) |



# 2020-02-03
[Return to Index](#Index)



<h2 id="2020-02-03-1">1. Self-Adversarial Learning with Comparative Discrimination for Text Generation</h2>

Title: [Self-Adversarial Learning with Comparative Discrimination for Text Generation](https://arxiv.org/abs/2001.11691)

Authors: [Wangchunshu Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+W), [Tao Ge](https://arxiv.org/search/cs?searchtype=author&query=Ge%2C+T), [Ke Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+K), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F), [Ming Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+M)

*(Submitted on 31 Jan 2020)*

> Conventional Generative Adversarial Networks (GANs) for text generation tend to have issues of reward sparsity and mode collapse that affect the quality and diversity of generated samples. To address the issues, we propose a novel self-adversarial learning (SAL) paradigm for improving GANs' performance in text generation. In contrast to standard GANs that use a binary classifier as its discriminator to predict whether a sample is real or generated, SAL employs a comparative discriminator which is a pairwise classifier for comparing the text quality between a pair of samples. During training, SAL rewards the generator when its currently generated sentence is found to be better than its previously generated samples. This self-improvement reward mechanism allows the model to receive credits more easily and avoid collapsing towards the limited number of real samples, which not only helps alleviate the reward sparsity issue but also reduces the risk of mode collapse. Experiments on text generation benchmark datasets show that our proposed approach substantially improves both the quality and the diversity, and yields more stable performance compared to the previous GANs for text generation.

| Comments: | to be published in ICLR 2020                                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | [arXiv:2001.11691](https://arxiv.org/abs/2001.11691) [cs.CL] |
|           | (or [arXiv:2001.11691v1](https://arxiv.org/abs/2001.11691v1) [cs.CL] for this version) |





<h2 id="2020-02-03-2">2. Teaching Machines to Converse</h2>

Title: [Teaching Machines to Converse](https://arxiv.org/abs/2001.11701)

Authors: [Jiwei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J)

*(Submitted on 31 Jan 2020)*

> The ability of a machine to communicate with humans has long been associated with the general success of AI. This dates back to Alan Turing's epoch-making work in the early 1950s, which proposes that a machine's intelligence can be tested by how well it, the machine, can fool a human into believing that the machine is a human through dialogue conversations. Many systems learn generation rules from a minimal set of authored rules or labels on top of hand-coded rules or templates, and thus are both expensive and difficult to extend to open-domain scenarios. Recently, the emergence of neural network models the potential to solve many of the problems in dialogue learning that earlier systems cannot tackle: the end-to-end neural frameworks offer the promise of scalability and language-independence, together with the ability to track the dialogue state and then mapping between states and dialogue actions in a way not possible with conventional systems. On the other hand, neural systems bring about new challenges: they tend to output dull and generic responses; they lack a consistent or a coherent persona; they are usually optimized through single-turn conversations and are incapable of handling the long-term success of a conversation; and they are not able to take the advantage of the interactions with humans. This dissertation attempts to tackle these challenges: Contributions are two-fold: (1) we address new challenges presented by neural network models in open-domain dialogue generation systems; (2) we develop interactive question-answering dialogue systems by (a) giving the agent the ability to ask questions and (b) training a conversation agent through interactions with humans in an online fashion, where a bot improves through communicating with humans and learning from the mistakes that it makes.

| Comments: | phd thesis                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:2001.11701](https://arxiv.org/abs/2001.11701) [cs.CL] |
|           | (or [arXiv:2001.11701v1](https://arxiv.org/abs/2001.11701v1) [cs.CL] for this version) |


