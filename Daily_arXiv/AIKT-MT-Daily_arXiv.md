# Daily arXiv: Machine Translation - Jan., 2020

# Index

- [2020-01-07](#2020-01-07)
  - [1. A Comprehensive Survey of Multilingual Neural Machine Translation](#2020-01-07-1)
  - [2. Morphological Word Segmentation on Agglutinative Languages for Neural Machine Translation](#2020-01-07-2)
  - [3. Exploring Benefits of Transfer Learning in Neural Machine Translation](#2020-01-07-3)
- [2020-01-06](#2020-01-06)
  - [1. Learning Accurate Integer Transformer Machine-Translation Models](#2020-01-06-1)
- [2020-01-03](#2020-01-03)
  - [1. A Voice Interactive Multilingual Student Support System using IBM Watson](#2020-01-03-1)
- [2020-01-01](2020-01-01)
  - [1. TextScanner: Reading Characters in Order for Robust Scene Text Recognition](#2020-01-01-1)
  - [2. Teaching a New Dog Old Tricks: Resurrecting Multilingual Retrieval Using Zero-shot Learning](#2020-01-01-2)
  - [3. Robust Cross-lingual Embeddings from Parallel Sentences](#2020-01-01-3)
  - [4. "Hinglish" Language -- Modeling a Messy Code-Mixed Language](#2020-01-01-4)
  - [5. Amharic-Arabic Neural Machine Translation](#2020-01-01-5)
  - [6. LayoutLM: Pre-training of Text and Layout for Document Image Understanding](#2020-01-01-6)
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



# 2020-01-07

[Return to Index](#Index)



<h2 id="2020-01-07-1">1. A Comprehensive Survey of Multilingual Neural Machine Translation</h2>

Title: [A Comprehensive Survey of Multilingual Neural Machine Translation](https://arxiv.org/abs/2001.01115)

Authors: [Raj Dabre](https://arxiv.org/search/cs?searchtype=author&query=Dabre%2C+R), [Chenhui Chu](https://arxiv.org/search/cs?searchtype=author&query=Chu%2C+C), [Anoop Kunchukuttan](https://arxiv.org/search/cs?searchtype=author&query=Kunchukuttan%2C+A)

*(Submitted on 4 Jan 2020)*

> We present a survey on multilingual neural machine translation (MNMT), which has gained a lot of traction in the recent years. MNMT has been useful in improving translation quality as a result of translation knowledge transfer (transfer learning). MNMT is more promising and interesting than its statistical machine translation counterpart because end-to-end modeling and distributed representations open new avenues for research on machine translation. Many approaches have been proposed in order to exploit multilingual parallel corpora for improving translation quality. However, the lack of a comprehensive survey makes it difficult to determine which approaches are promising and hence deserve further exploration. In this paper, we present an in-depth survey of existing literature on MNMT. We first categorize various approaches based on their central use-case and then further categorize them based on resource scenarios, underlying modeling principles, core-issues and challenges. Wherever possible we address the strengths and weaknesses of several techniques by comparing them with each other. We also discuss the future directions that MNMT research might take. This paper is aimed towards both, beginners and experts in NMT. We hope this paper will serve as a starting point as well as a source of new ideas for researchers and engineers interested in MNMT.

| Comments: | This is an extended version of our survey paper on multilingual NMT. The previous version [[arXiv:1905.05395](https://arxiv.org/abs/1905.05395)] is rather condensed and is useful for speed-reading whereas this version is more beginner friendly. Under review at the computing surveys journal. We have intentionally decided to maintain both short and long versions of our survey paper for different reader groups |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | [arXiv:2001.01115](https://arxiv.org/abs/2001.01115) [cs.CL] |
|           | (or [arXiv:2001.01115v1](https://arxiv.org/abs/2001.01115v1) [cs.CL] for this version) |





<h2 id="2020-01-07-2">2. Morphological Word Segmentation on Agglutinative Languages for Neural Machine Translation</h2>

Title: [Morphological Word Segmentation on Agglutinative Languages for Neural Machine Translation](https://arxiv.org/abs/2001.01589)

Authors: [Yirong Pan](https://arxiv.org/search/cs?searchtype=author&query=Pan%2C+Y), [Xiao Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+X), [Yating Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Y), [Rui Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+R)

*(Submitted on 2 Jan 2020)*

> Neural machine translation (NMT) has achieved impressive performance on machine translation task in recent years. However, in consideration of efficiency, a limited-size vocabulary that only contains the top-N highest frequency words are employed for model training, which leads to many rare and unknown words. It is rather difficult when translating from the low-resource and morphologically-rich agglutinative languages, which have complex morphology and large vocabulary. In this paper, we propose a morphological word segmentation method on the source-side for NMT that incorporates morphology knowledge to preserve the linguistic and semantic information in the word structure while reducing the vocabulary size at training time. It can be utilized as a preprocessing tool to segment the words in agglutinative languages for other natural language processing (NLP) tasks. Experimental results show that our morphologically motivated word segmentation method is better suitable for the NMT model, which achieves significant improvements on Turkish-English and Uyghur-Chinese machine translation tasks on account of reducing data sparseness and language complexity.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2001.01589](https://arxiv.org/abs/2001.01589) [cs.CL] |
|           | (or [arXiv:2001.01589v1](https://arxiv.org/abs/2001.01589v1) [cs.CL] for this version) |





<h2 id="2020-01-07-3">3. Exploring Benefits of Transfer Learning in Neural Machine Translation</h2>

Title: [Exploring Benefits of Transfer Learning in Neural Machine Translation](https://arxiv.org/abs/2001.01622)

Authors: [Tom Kocmi](https://arxiv.org/search/cs?searchtype=author&query=Kocmi%2C+T)

*(Submitted on 6 Jan 2020)*

> Neural machine translation is known to require large numbers of parallel training sentences, which generally prevent it from excelling on low-resource language pairs. This thesis explores the use of cross-lingual transfer learning on neural networks as a way of solving the problem with the lack of resources. We propose several transfer learning approaches to reuse a model pretrained on a high-resource language pair. We pay particular attention to the simplicity of the techniques. We study two scenarios: (a) when we reuse the high-resource model without any prior modifications to its training process and (b) when we can prepare the first-stage high-resource model for transfer learning in advance. For the former scenario, we present a proof-of-concept method by reusing a model trained by other researchers. In the latter scenario, we present a method which reaches even larger improvements in translation performance. Apart from proposed techniques, we focus on an in-depth analysis of transfer learning techniques and try to shed some light on transfer learning improvements. We show how our techniques address specific problems of low-resource languages and are suitable even in high-resource transfer learning. We evaluate the potential drawbacks and behavior by studying transfer learning in various situations, for example, under artificially damaged training corpora, or with fixed various model parts.

| Comments: | Defended PhD thesis                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Neural and Evolutionary Computing (cs.NE) |
| Cite as:  | [arXiv:2001.01622](https://arxiv.org/abs/2001.01622) [cs.CL] |
|           | (or [arXiv:2001.01622v1](https://arxiv.org/abs/2001.01622v1) [cs.CL] for this version) |





# 2020-01-06

[Return to Index](#Index)



<h2 id="2020-01-06-1">1. Learning Accurate Integer Transformer Machine-Translation Models</h2>
Title: [Learning Accurate Integer Transformer Machine-Translation Models](https://arxiv.org/abs/2001.00926)

Authors: [Ephrem Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+E)

*(Submitted on 3 Jan 2020)*

> We describe a method for training accurate Transformer machine-translation models to run inference using 8-bit integer (INT8) hardware matrix multipliers, as opposed to the more costly single-precision floating-point (FP32) hardware. Unlike previous work, which converted only 85 Transformer matrix multiplications to INT8, leaving 48 out of 133 of them in FP32 because of unacceptable accuracy loss, we convert them all to INT8 without compromising accuracy. Tested on the newstest2014 English-to-German translation task, our INT8 Transformer Base and Transformer Big models yield BLEU scores that are 99.3% to 100% relative to those of the corresponding FP32 models. Our approach converts all matrix-multiplication tensors from an existing FP32 model into INT8 tensors by automatically making range-precision trade-offs during training. To demonstrate the robustness of this approach, we also include results from INT6 Transformer models.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2001.00926](https://arxiv.org/abs/2001.00926) [cs.LG] |
|           | (or [arXiv:2001.00926v1](https://arxiv.org/abs/2001.00926v1) [cs.LG] for this version) |





# 2020-01-03

[Return to Index](#Index)




<h2 id="2020-01-03-1">1. A Voice Interactive Multilingual Student Support System using IBM Watson</h2>
Title: [A Voice Interactive Multilingual Student Support System using IBM Watson](https://arxiv.org/abs/2001.00471)

Authors: [Kennedy Ralston](https://arxiv.org/search/cs?searchtype=author&query=Ralston%2C+K), [Yuhao Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Haruna Isah](https://arxiv.org/search/cs?searchtype=author&query=Isah%2C+H), [Farhana Zulkernine](https://arxiv.org/search/cs?searchtype=author&query=Zulkernine%2C+F)

(Submitted on 20 Dec 2019)

> Systems powered by artificial intelligence are being developed to be more user-friendly by communicating with users in a progressively human-like conversational way. Chatbots, also known as dialogue systems, interactive conversational agents, or virtual agents are an example of such systems used in a wide variety of applications ranging from customer support in the business domain to companionship in the healthcare sector. It is becoming increasingly important to develop chatbots that can best respond to the personalized needs of their users so that they can be as helpful to the user as possible in a real human way. This paper investigates and compares three popular existing chatbots API offerings and then propose and develop a voice interactive and multilingual chatbot that can effectively respond to users mood, tone, and language using IBM Watson Assistant, Tone Analyzer, and Language Translator. The chatbot was evaluated using a use case that was targeted at responding to users needs regarding exam stress based on university students survey data generated using Google Forms. The results of measuring the chatbot effectiveness at analyzing responses regarding exam stress indicate that the chatbot responding appropriately to the user queries regarding how they are feeling about exams 76.5%. The chatbot could also be adapted for use in other application areas such as student info-centers, government kiosks, and mental health support systems.

| Comments: | 6 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Human-Computer Interaction (cs.HC)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Information Retrieval (cs.IR) |
| Cite as:  | [arXiv:2001.00471](https://arxiv.org/abs/2001.00471) [cs.HC] |
|           | (or [arXiv:2001.00471v1](https://arxiv.org/abs/2001.00471v1) [cs.HC] for this version) |



# 2020-01-01

[Return to Index](#Index)



<h2 id="2020-01-01-1">1. TextScanner: Reading Characters in Order for Robust Scene Text Recognition</h2>
Title: [TextScanner: Reading Characters in Order for Robust Scene Text Recognition](https://arxiv.org/abs/1912.12422)

Authors: [Zhaoyi Wan](https://arxiv.org/search/cs?searchtype=author&query=Wan%2C+Z), [Mingling He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+M), [Haoran Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+H), [Xiang Bai](https://arxiv.org/search/cs?searchtype=author&query=Bai%2C+X), [Cong Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao%2C+C)

*(Submitted on 28 Dec 2019)*

> Driven by deep learning and the large volume of data, scene text recognition has evolved rapidly in recent years. Formerly, RNN-attention based methods have dominated this field, but suffer from the problem of \textit{attention drift} in certain situations. Lately, semantic segmentation based algorithms have proven effective at recognizing text of different forms (horizontal, oriented and curved). However, these methods may produce spurious characters or miss genuine characters, as they rely heavily on a thresholding procedure operated on segmentation maps. To tackle these challenges, we propose in this paper an alternative approach, called TextScanner, for scene text recognition. TextScanner bears three characteristics: (1) Basically, it belongs to the semantic segmentation family, as it generates pixel-wise, multi-channel segmentation maps for character class, position and order; (2) Meanwhile, akin to RNN-attention based methods, it also adopts RNN for context modeling; (3) Moreover, it performs paralleled prediction for character position and class, and ensures that characters are transcripted in correct order. The experiments on standard benchmark datasets demonstrate that TextScanner outperforms the state-of-the-art methods. Moreover, TextScanner shows its superiority in recognizing more difficult text such Chinese transcripts and aligning with target characters.

| Comments: | Accepted by AAAI-2020                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1912.12422](https://arxiv.org/abs/1912.12422) [cs.CV] |
|           | (or [arXiv:1912.12422v1](https://arxiv.org/abs/1912.12422v1) [cs.CV] for this version) |





<h2 id="2020-01-01-2">2. Teaching a New Dog Old Tricks: Resurrecting Multilingual Retrieval Using Zero-shot Learning</h2>
Title: [Teaching a New Dog Old Tricks: Resurrecting Multilingual Retrieval Using Zero-shot Learning](https://arxiv.org/abs/1912.13080)

Authors: [Sean MacAvaney](https://arxiv.org/search/cs?searchtype=author&query=MacAvaney%2C+S), [Luca Soldaini](https://arxiv.org/search/cs?searchtype=author&query=Soldaini%2C+L), [Nazli Goharian](https://arxiv.org/search/cs?searchtype=author&query=Goharian%2C+N)

*(Submitted on 30 Dec 2019)*

> While billions of non-English speaking users rely on search engines every day, the problem of ad-hoc information retrieval is rarely studied for non-English languages. This is primarily due to a lack of data set that are suitable to train ranking algorithms. In this paper, we tackle the lack of data by leveraging pre-trained multilingual language models to transfer a retrieval system trained on English collections to non-English queries and documents. Our model is evaluated in a zero-shot setting, meaning that we use them to predict relevance scores for query-document pairs in languages never seen during training. Our results show that the proposed approach can significantly outperform unsupervised retrieval techniques for Arabic, Chinese Mandarin, and Spanish. We also show that augmenting the English training collection with some examples from the target language can sometimes improve performance.

| Comments: | ECIR 2020 (short)                                            |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Information Retrieval (cs.IR)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1912.13080](https://arxiv.org/abs/1912.13080) [cs.IR] |
|           | (or [arXiv:1912.13080v1](https://arxiv.org/abs/1912.13080v1) [cs.IR] for this version) |





<h2 id="2020-01-01-3">3. Robust Cross-lingual Embeddings from Parallel Sentences</h2>
Title: [Robust Cross-lingual Embeddings from Parallel Sentences](https://arxiv.org/abs/1912.12481)

Authors: [Ali Sabet](https://arxiv.org/search/cs?searchtype=author&query=Sabet%2C+A), [Prakhar Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+P), [Jean-Baptiste Cordonnier](https://arxiv.org/search/cs?searchtype=author&query=Cordonnier%2C+J), [Robert West](https://arxiv.org/search/cs?searchtype=author&query=West%2C+R), [Martin Jaggi](https://arxiv.org/search/cs?searchtype=author&query=Jaggi%2C+M)

*(Submitted on 28 Dec 2019)*

> Recent advances in cross-lingual word embeddings have primarily relied on mapping-based methods, which project pretrained word embeddings from different languages into a shared space through a linear transformation. However, these approaches assume word embedding spaces are isomorphic between different languages, which has been shown not to hold in practice (Søgaard et al., 2018), and fundamentally limits their performance. This motivates investigating joint learning methods which can overcome this impediment, by simultaneously learning embeddings across languages via a cross-lingual term in the training objective. Given the abundance of parallel data available (Tiedemann, 2012), we propose a bilingual extension of the CBOW method which leverages sentence-aligned corpora to obtain robust cross-lingual word and sentence representations. Our approach significantly improves cross-lingual sentence retrieval performance over all other approaches, as well as convincingly outscores mapping methods while maintaining parity with jointly trained methods on word-translation. It also achieves parity with a deep RNN method on a zero-shot cross-lingual document classification task, requiring far fewer computational resources for training and inference. As an additional advantage, our bilingual method also improves the quality of monolingual word vectors despite training on much smaller datasets. We make our code and models publicly available.

| Subjects: | **Computation and Language (cs.CL)**; Information Retrieval (cs.IR); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1912.12481](https://arxiv.org/abs/1912.12481) [cs.CL] |
|           | (or [arXiv:1912.12481v1](https://arxiv.org/abs/1912.12481v1) [cs.CL] for this version) |





<h2 id="2020-01-01-4">4. "Hinglish" Language -- Modeling a Messy Code-Mixed Language</h2>
Title: ["Hinglish" Language -- Modeling a Messy Code-Mixed Language](https://arxiv.org/abs/1912.13109)

Authors: [Vivek Kumar Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+V+K)

*(Submitted on 30 Dec 2019)*

> With a sharp rise in fluency and users of "Hinglish" in linguistically diverse country, India, it has increasingly become important to analyze social content written in this language in platforms such as Twitter, Reddit, Facebook. This project focuses on using deep learning techniques to tackle a classification problem in categorizing social content written in Hindi-English into Abusive, Hate-Inducing and Not offensive categories. We utilize bi-directional sequence models with easy text augmentation techniques such as synonym replacement, random insertion, random swap, and random deletion to produce a state of the art classifier that outperforms the previous work done on analyzing this dataset.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1912.13109](https://arxiv.org/abs/1912.13109) [cs.CL] |
|           | (or [arXiv:1912.13109v1](https://arxiv.org/abs/1912.13109v1) [cs.CL] for this version) |





<h2 id="2020-01-01-5">5. Amharic-Arabic Neural Machine Translation</h2>
Title: [Amharic-Arabic Neural Machine Translation](https://arxiv.org/abs/1912.13161)

Authors: [Ibrahim Gashaw](https://arxiv.org/search/cs?searchtype=author&query=Gashaw%2C+I), [H L Shashirekha](https://arxiv.org/search/cs?searchtype=author&query=Shashirekha%2C+H+L)

*(Submitted on 26 Dec 2019)*

> Many automatic translation works have been addressed between major European language pairs, by taking advantage of large scale parallel corpora, but very few research works are conducted on the Amharic-Arabic language pair due to its parallel data scarcity. Two Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) based Neural Machine Translation (NMT) models are developed using Attention-based Encoder-Decoder architecture which is adapted from the open-source OpenNMT system. In order to perform the experiment, a small parallel Quranic text corpus is constructed by modifying the existing monolingual Arabic text and its equivalent translation of Amharic language text corpora available on Tanzile. LSTM and GRU based NMT models and Google Translation system are compared and found that LSTM based OpenNMT outperforms GRU based OpenNMT and Google Translation system, with a BLEU score of 12%, 11%, and 6% respectively.

| Comments: | 15 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1912.13161](https://arxiv.org/abs/1912.13161) [cs.CL] |
|           | (or [arXiv:1912.13161v1](https://arxiv.org/abs/1912.13161v1) [cs.CL] for this version) |





<h2 id="2020-01-01-6">6. LayoutLM: Pre-training of Text and Layout for Document Image Understanding</h2>
Title: [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318)

Authors: [Yiheng Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+Y), [Minghao Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+M), [Lei Cui](https://arxiv.org/search/cs?searchtype=author&query=Cui%2C+L), [Shaohan Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F), [Ming Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+M)

*(Submitted on 31 Dec 2019)*

> Pre-training techniques have been verified successfully in a variety of NLP tasks in recent years. Despite the wide spread of pre-training models for NLP applications, they almost focused on text-level manipulation, while neglecting the layout and style information that is vital for document image understanding. In this paper, we propose \textbf{LayoutLM} to jointly model the interaction between text and layout information across scanned document images, which is beneficial for a great number of real-world document image understanding tasks such as information extraction from scanned documents. We also leverage the image features to incorporate the style information of words in LayoutLM. To the best of our knowledge, this is the first time that text and layout are jointly learned in a single framework for document-level pre-training, leading to significant performance improvement in downstream tasks for document image understanding.

| Comments: | Work in progress                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1912.13318](https://arxiv.org/abs/1912.13318) [cs.CL] |
|           | (or [arXiv:1912.13318v1](https://arxiv.org/abs/1912.13318v1) [cs.CL] for this version) |



