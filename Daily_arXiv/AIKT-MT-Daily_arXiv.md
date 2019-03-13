# Daily arXiv: Machine Translation - Mar., 2019

### Index

- [2019-03-13](#2019-03-13)
  - [1. Context-Aware Learning for Neural Machine Translation](#2019-03-13-1)
- [2019-03-12](#2019-03-12)
  - [1. Partially Shuffling the Training Data to Improve Language Models](#2019-03-12-1)
  - [2. ETNLP: A Toolkit for Extraction, Evaluation and Visualization of Pre-trained Word Embeddings](#2019-03-12-2)
- [2019-03-11](#2019-03-11)
  - [1. Context-Aware Crosslingual Mapping](#2019-03-11-1)
  - [2. Filling Gender & Number Gaps in Neural Machine Translation with Black-box Context Injection](#2019-03-11-2)
- [2019-03-08](#2019-03-08)
  - [1. Integrating Artificial and Human Intelligence for Efficient Translation](#2019-03-08-1)
- [2019-03-04](#2019-03-04)
  - [1. Chinese-Japanese Unsupervised Neural Machine Translation Using Sub-character Level Information](#2019-03-04-1)
  - [2. Massively Multilingual Neural Machine Translation](#2019-03-04-2)
  - [3. Non-Parametric Adaptation for Neural Machine Translation](#2019-03-04-3)
  - [4. Reinforcement Learning based Curriculum Optimization for Neural Machine Translation](#2019-03-04-4)
- [2019-03-01](#2019-03-01)
  - [1. Efficient Contextual Representation Learning Without Softmax Layer](#2019-03-01-1)

* [2019-02](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-02.md)



# 2019-03-13

[Return to Index](#Index)

<h2 id="2019-03-13-1">1. Context-Aware Learning for Neural Machine Translation</h2> 

Title: [Context-Aware Learning for Neural Machine Translation](https://arxiv.org/abs/1903.04715)

Authors: [Sébastien Jean](https://arxiv.org/search/cs?searchtype=author&query=Jean%2C+S), [Kyunghyun Cho](https://arxiv.org/search/cs?searchtype=author&query=Cho%2C+K)

*(Submitted on 12 Mar 2019)*

> Interest in larger-context neural machine translation, including document-level and multi-modal translation, has been growing. Multiple works have proposed new network architectures or evaluation schemes, but potentially helpful context is still sometimes ignored by larger-context translation models. In this paper, we propose a novel learning algorithm that explicitly encourages a neural translation model to take into account additional context using a multilevel pair-wise ranking loss. We evaluate the proposed learning algorithm with a transformer-based larger-context translation system on document-level translation. By comparing performance using actual and random contexts, we show that a model trained with the proposed algorithm is more sensitive to the additional context.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1903.04715](https://arxiv.org/abs/1903.04715) [cs.CL] |
|           | (or **arXiv:1903.04715v1 [cs.CL]** for this version)         |



# 2019-03-12

[Return to Index](#Index)

<h2 id="2019-03-12-1">1. Partially Shuffling the Training Data to Improve Language Models</h2> 

Title: [Partially Shuffling the Training Data to Improve Language Models](https://arxiv.org/abs/1903.04167)

Authors: [Ofir Press](https://arxiv.org/search/cs?searchtype=author&query=Press%2C+O)

*(Submitted on 11 Mar 2019)*

> Although SGD requires shuffling the training data between epochs, currently none of the word-level language modeling systems do this. Naively shuffling all sentences in the training data would not permit the model to learn inter-sentence dependencies. Here we present a method that partially shuffles the training data between epochs. This method makes each batch random, while keeping most sentence ordering intact. It achieves new state of the art results on word-level language modeling on both the Penn Treebank and WikiText-2 datasets.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1903.04167](https://arxiv.org/abs/1903.04167) [cs.CL] |
|           | (or **arXiv:1903.04167v1 [cs.CL]** for this version)         |



<h2 id="2019-03-12-2">2. ETNLP: A Toolkit for Extraction, Evaluation and Visualization of Pre-trained Word Embeddings</h2> 

Title: [ETNLP: A Toolkit for Extraction, Evaluation and Visualization of Pre-trained Word Embeddings](https://arxiv.org/abs/1903.04433)

Authors: [Xuan-Son Vu](https://arxiv.org/search/cs?searchtype=author&query=Vu%2C+X), [Thanh Vu](https://arxiv.org/search/cs?searchtype=author&query=Vu%2C+T), [Son N. Tran](https://arxiv.org/search/cs?searchtype=author&query=Tran%2C+S+N), [Lili Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+L)

*(Submitted on 11 Mar 2019)*

> In this paper, we introduce a comprehensive toolkit, ETNLP, which can evaluate, extract, and visualize multiple sets of pre-trained word embeddings. First, for evaluation, ETNLP analyses the quality of pre-trained embeddings based on an input word analogy list. Second, for extraction ETNLP provides a subset of the embeddings to be used in the downstream NLP tasks. Finally, ETNLP has a visualization module which is for exploring the embedded words interactively. We demonstrate the effectiveness of ETNLP on our pre-trained word embeddings in Vietnamese. Specifically, we create a large Vietnamese word analogy list to evaluate the embeddings. We then utilize the pre-trained embeddings for the name entity recognition (NER) task in Vietnamese and achieve the new state-of-the-art results on a benchmark dataset for the NER task. A video demonstration of ETNLP is available at [this https URL](https://vimeo.com/317599106). The source code and data are available at https: //github.com/vietnlp/etnlp.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1903.04433](https://arxiv.org/abs/1903.04433) [cs.CL] |
|           | (or **arXiv:1903.04433v1 [cs.CL]** for this version)         |



# 2019-03-11

[Return to Index](#Index)

<h2 id="2019-03-11-1">1. Context-Aware Crosslingual Mapping</h2> 

Title: [Context-Aware Crosslingual Mapping](https://arxiv.org/abs/1903.03243)

Authors:[Hanan Aldarmaki](https://arxiv.org/search/cs?searchtype=author&query=Aldarmaki%2C+H), [Mona Diab](https://arxiv.org/search/cs?searchtype=author&query=Diab%2C+M)

*(Submitted on 8 Mar 2019)*

> Cross-lingual word vectors are typically obtained by fitting an orthogonal matrix that maps the entries of a bilingual dictionary from a source to a target vector space. Word vectors, however, are most commonly used for sentence or document-level representations that are calculated as the weighted average of word embeddings. In this paper, we propose an alternative to word-level mapping that better reflects sentence-level cross-lingual similarity. We incorporate context in the transformation matrix by directly mapping the averaged embeddings of aligned sentences in a parallel corpus. We also implement cross-lingual mapping of deep contextualized word embeddings using parallel sentences with word alignments. In our experiments, both approaches resulted in cross-lingual sentence embeddings that outperformed context-independent word mapping in sentence translation retrieval. Furthermore, the sentence-level transformation could be used for word-level mapping without loss in word translation quality.

| Comments: | NAACL-HLT 2019 (short paper). 5 pages, 1 figure              |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1903.03243](https://arxiv.org/abs/1903.03243) [cs.CL] |
|           | (or **arXiv:1903.03243v1 [cs.CL]** for this version)         |







<h2 id="2019-03-11-2">2. Filling Gender & Number Gaps in Neural Machine Translation with Black-box Context Injection</h2> 

Title: [Filling Gender & Number Gaps in Neural Machine Translation with Black-box Context Injection](https://arxiv.org/abs/1903.03467)

Authors: [Amit Moryossef](https://arxiv.org/search/cs?searchtype=author&query=Moryossef%2C+A), [Roee Aharoni](https://arxiv.org/search/cs?searchtype=author&query=Aharoni%2C+R), [Yoav Goldberg](https://arxiv.org/search/cs?searchtype=author&query=Goldberg%2C+Y)

*(Submitted on 8 Mar 2019)*

> When translating from a language that does not morphologically mark information such as gender and number into a language that does, translation systems must "guess" this missing information, often leading to incorrect translations in the given context. We propose a black-box approach for injecting the missing information to a pre-trained neural machine translation system, allowing to control the morphological variations in the generated translations without changing the underlying model or training data. We evaluate our method on an English to Hebrew translation task, and show that it is effective in injecting the gender and number information and that supplying the correct information improves the translation accuracy in up to 2.3 BLEU on a female-speaker test set for a state-of-the-art online black-box system. Finally, we perform a fine-grained syntactic analysis of the generated translations that shows the effectiveness of our method.

| Comments: | 6 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1903.03467](https://arxiv.org/abs/1903.03467) [cs.CL] |
|           | (or **arXiv:1903.03467v1 [cs.CL]** for this version)         |



# 2019-03-08

[Return to Index](#Index)

<h2 id="2019-03-08-1">1. Integrating Artificial and Human Intelligence for Efficient Translation</h2> 

Title: [Integrating Artificial and Human Intelligence for Efficient Translation](https://arxiv.org/abs/1903.02978)

Authors: [Nico Herbig](https://arxiv.org/search/cs?searchtype=author&query=Herbig%2C+N), [Santanu Pal](https://arxiv.org/search/cs?searchtype=author&query=Pal%2C+S), [Josef van Genabith](https://arxiv.org/search/cs?searchtype=author&query=van+Genabith%2C+J), [Antonio Krüger](https://arxiv.org/search/cs?searchtype=author&query=Kr%C3%BCger%2C+A)

*(Submitted on 7 Mar 2019)*

> Current advances in machine translation increase the need for translators to switch from traditional translation to post-editing of machine-translated text, a process that saves time and improves quality. Human and artificial intelligence need to be integrated in an efficient way to leverage the advantages of both for the translation task. This paper outlines approaches at this boundary of AI and HCI and discusses open research questions to further advance the field.

| Subjects: | **Human-Computer Interaction (cs.HC)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1903.02978](https://arxiv.org/abs/1903.02978) [cs.HC] |
|           | (or **arXiv:1903.02978v1 [cs.HC]** for this version)         |





# 2019-03-04

[Return to Index](#Index)

<h2 id="2019-03-04-1">1. Chinese-Japanese Unsupervised Neural Machine Translation Using Sub-character Level Information</h2> 

Title: [Chinese-Japanese Unsupervised Neural Machine Translation Using Sub-character Level Information](https://arxiv.org/abs/1903.00149)

Authors: [Longtu Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+L), [Mamoru Komachi](https://arxiv.org/search/cs?searchtype=author&query=Komachi%2C+M)

*(Submitted on 1 Mar 2019)*

> Unsupervised neural machine translation (UNMT) requires only monolingual data of similar language pairs during training and can produce bi-directional translation models with relatively good performance on alphabetic languages (Lample et al., 2018). However, no research has been done to logographic language pairs. This study focuses on Chinese-Japanese UNMT trained by data containing sub-character (ideograph or stroke) level information which is decomposed from character level data. BLEU scores of both character and sub-character level systems were compared against each other and the results showed that despite the effectiveness of UNMT on character level data, sub-character level data could further enhance the performance, in which the stroke level system outperformed the ideograph level system.

| Comments: | 5 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1903.00149](https://arxiv.org/abs/1903.00149) [cs.CL] |
|           | (or **arXiv:1903.00149v1 [cs.CL]** for this version)         |



<h2 id="2019-03-04-2">2. Massively Multilingual Neural Machine Translation</h2> 

Title: [Massively Multilingual Neural Machine Translation](https://arxiv.org/abs/1903.00089)

Authors: [Roee Aharoni](https://arxiv.org/search/cs?searchtype=author&query=Aharoni%2C+R), [Melvin Johnson](https://arxiv.org/search/cs?searchtype=author&query=Johnson%2C+M), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O)

*(Submitted on 28 Feb 2019)*

> Multilingual neural machine translation (NMT) enables training a single model that supports translation from multiple source languages into multiple target languages. In this paper, we push the limits of multilingual NMT in terms of number of languages being used. We perform extensive experiments in training massively multilingual NMT models, translating up to 102 languages to and from English within a single model. We explore different setups for training such models and analyze the trade-offs between translation quality and various modeling decisions. We report results on the publicly available TED talks multilingual corpus where we show that massively multilingual many-to-many models are effective in low resource settings, outperforming the previous state-of-the-art while supporting up to 59 languages. Our experiments on a large-scale dataset with 102 languages to and from English and up to one million examples per direction also show promising results, surpassing strong bilingual baselines and encouraging future work on massively multilingual NMT.

| Comments: | Accepted as a long paper in NAACL 2019                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1903.00089](https://arxiv.org/abs/1903.00089) [cs.CL] |
|           | (or **arXiv:1903.00089v1 [cs.CL]** for this version)         |



<h2 id="2019-03-04-3">3. Non-Parametric Adaptation for Neural Machine Translation</h2> 

Title: [Non-Parametric Adaptation for Neural Machine Translation](https://arxiv.org/abs/1903.00058)

Authors: [Ankur Bapna](https://arxiv.org/search/cs?searchtype=author&query=Bapna%2C+A), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O)

*(Submitted on 28 Feb 2019)*

> Neural Networks trained with gradient descent are known to be susceptible to catastrophic forgetting caused by parameter shift during the training process. In the context of Neural Machine Translation (NMT) this results in poor performance on heterogeneous datasets and on sub-tasks like rare phrase translation. On the other hand, non-parametric approaches are immune to forgetting, perfectly complementing the generalization ability of NMT. However, attempts to combine non-parametric or retrieval based approaches with NMT have only been successful on narrow domains, possibly due to over-reliance on sentence level retrieval. We propose a novel n-gram level retrieval approach that relies on local phrase level similarities, allowing us to retrieve neighbors that are useful for translation even when overall sentence similarity is low. We complement this with an expressive neural network, allowing our model to extract information from the noisy retrieved context. We evaluate our semi-parametric NMT approach on a heterogeneous dataset composed of WMT, IWSLT, JRC-Acquis and OpenSubtitles, and demonstrate gains on all 4 evaluation sets. The semi-parametric nature of our approach opens the door for non-parametric domain adaptation, demonstrating strong inference-time adaptation performance on new domains without the need for any parameter updates.

| Comments: | To appear at NAACL 2019                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Machine Learning (stat.ML) |
| Cite as:  | [arXiv:1903.00058](https://arxiv.org/abs/1903.00058) [cs.CL] |
|           | (or **arXiv:1903.00058v1 [cs.CL]** for this version)         |



<h2 id="2019-03-04-4">4. Reinforcement Learning based Curriculum Optimization for Neural Machine Translation</h2> 

Title: [Reinforcement Learning based Curriculum Optimization for Neural Machine Translation](https://arxiv.org/abs/1903.00041)

Authors: [Gaurav Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+G), [George Foster](https://arxiv.org/search/cs?searchtype=author&query=Foster%2C+G), [Colin Cherry](https://arxiv.org/search/cs?searchtype=author&query=Cherry%2C+C), [Maxim Krikun](https://arxiv.org/search/cs?searchtype=author&query=Krikun%2C+M)

*(Submitted on 28 Feb 2019)*

> We consider the problem of making efficient use of heterogeneous training data in neural machine translation (NMT). Specifically, given a training dataset with a sentence-level feature such as noise, we seek an optimal curriculum, or order for presenting examples to the system during training. Our curriculum framework allows examples to appear an arbitrary number of times, and thus generalizes data weighting, filtering, and fine-tuning schemes. Rather than relying on prior knowledge to design a curriculum, we use reinforcement learning to learn one automatically, jointly with the NMT system, in the course of a single training run. We show that this approach can beat uniform and filtering baselines on Paracrawl and WMT English-to-French datasets by up to +3.4 BLEU, and match the performance of a hand-designed, state-of-the-art curriculum.

| Comments: | NAACL 2019 short paper. Reviewer comments not yet addressed  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1903.00041](https://arxiv.org/abs/1903.00041) [cs.CL] |
|           | (or **arXiv:1903.00041v1 [cs.CL]** for this version)         |



# 2019-03-01

[Return to Index](#Index)

<h2 id="2019-03-01-1">1. Efficient Contextual Representation Learning Without Softmax Layer</h2> 

Title: [Efficient Contextual Representation Learning Without Softmax Layer](https://arxiv.org/pdf/1902.11269)

Authors: [Liunian Harold Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L+H), [Patrick H. Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+P+H), [Cho-Jui Hsieh](https://arxiv.org/search/cs?searchtype=author&query=Hsieh%2C+C), [Kai-Wei Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang%2C+K)

Note: Contextual Representaion

*(Submitted on 28 Feb 2019)*

> Contextual representation models have achieved great success in improving various downstream tasks. However, these language-model-based encoders are difficult to train due to the large parameter sizes and high computational complexity. By carefully examining the training procedure, we find that the softmax layer (the output layer) causes significant inefficiency due to the large vocabulary size. Therefore, we redesign the learning objective and propose an efficient framework for training contextual representation models. Specifically, the proposed approach bypasses the softmax layer by performing language modeling with dimension reduction, and allows the models to leverage pre-trained word embeddings. Our framework reduces the time spent on the output layer to a negligible level, eliminates almost all the trainable parameters of the softmax layer and performs language modeling without truncating the vocabulary. When applied to ELMo, our method achieves a 4 times speedup and eliminates 80% trainable parameters while achieving competitive performance on downstream tasks.

| Comments: | Work in progress                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1902.11269](https://arxiv.org/abs/1902.11269) [cs.CL] |
|           | (or **arXiv:1902.11269v1 [cs.CL]** for this version)         |



