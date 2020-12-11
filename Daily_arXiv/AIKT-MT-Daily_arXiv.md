# Daily arXiv: Machine Translation - December, 2020

# Index


- [2020-12-11](#2020-12-11)

  - [1. Rewriter-Evaluator Framework for Neural Machine Translation](#2020-12-11-1)
  - [2. As good as new. How to successfully recycle English GPT-2 to make models for other languages](#2020-12-11-2)
  - [3. Direct multimodal few-shot learning of speech and images](#2020-12-11-3)
  - [4. Exploring Pair-Wise NMT for Indian Languages](#2020-12-11-4)
- [2020-12-10](#2020-12-10)

  - [1. SongMASS: Automatic Song Writing with Pre-training and Alignment Constraint](#2020-12-10-1)
  - [2. Breeding Gender-aware Direct Speech Translation Systems](#2020-12-10-2)
  - [3. On Knowledge Distillation for Direct Speech Translation](#2020-12-10-3)
  - [4. Towards Zero-shot Cross-lingual Image Retrieval](#2020-12-10-4)
- [2020-12-09](#2020-12-09)

  - [1. Revisiting Iterative Back-Translation from the Perspective of Compositional Generalization](#2020-12-09-1)
  - [2. Globetrotter: Unsupervised Multilingual Translation from Visual Alignment](#2020-12-09-2)
- [2020-12-08](#2020-12-08)

  - [1. Cross-Modal Generalization: Learning in Low Resource Modalities via Meta-Alignment](#2020-12-08-1)
  - [2. MLS: A Large-Scale Multilingual Dataset for Speech Research](#2020-12-08-2)
  - [3. Reciprocal Supervised Learning Improves Neural Machine Translation](#2020-12-08-3)
  - [4. Document Graph for Neural Machine Translation](#2020-12-08-4)
  - [5. KgPLM: Knowledge-guided Language Model Pre-training via Generative and Discriminative Learning](#2020-12-08-5)
  - [6. PPKE: Knowledge Representation Learning by Path-based Pre-training](#2020-12-08-6)
- [2020-12-07](#2020-12-07)
- [1. A Correspondence Variational Autoencoder for Unsupervised Acoustic Word Embeddings](#2020-12-07-1)
  - [2. Self-Supervised VQA: Answering Visual Questions using Images and Captions](#2020-12-07-2)
  - [3. Accurate and Scalable Matching of Translators to Displaced Persons for Overcoming Language Barriers](#2020-12-07-3)
  - [4. A Benchmark Dataset for Understandable Medical Language Translation](#2020-12-07-4)
  - [5. Fine-tuning BERT for Low-Resource Natural Language Understanding via Active Learning](#2020-12-07-5)
- [2020-12-04](#2020-12-04)

  - [1. SemMT: A Semantic-based Testing Approach for Machine Translation Systems](#2020-12-04-1)
  - [2. Self-Explaining Structures Improve NLP Models](#2020-12-04-2)
  - [3. On Extending NLP Techniques from the Categorical to the Latent Space: KL Divergence, Zipf's Law, and Similarity Search](#2020-12-04-3)
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



# 2020-12-11

[Return to Index](#Index)



<h2 id="2020-12-11-1">1. Rewriter-Evaluator Framework for Neural Machine Translation</h2>

Title: [Rewriter-Evaluator Framework for Neural Machine Translation](https://arxiv.org/abs/2012.05414)

Authors: [Yangming Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Kaisheng Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao%2C+K)

> Encoder-decoder architecture has been widely used in neural machine translation (NMT). A few methods have been proposed to improve it with multiple passes of decoding. However, their full potential is limited by a lack of appropriate termination policy. To address this issue, we present a novel framework, Rewriter-Evaluator. It consists of a rewriter and an evaluator. Translating a source sentence involves multiple passes. At every pass, the rewriter produces a new translation to improve the past translation and the evaluator estimates the translation quality to decide whether to terminate the rewriting process. We also propose a prioritized gradient descent (PGD) method that facilitates training the rewriter and the evaluator jointly. Though incurring multiple passes of decoding, Rewriter-Evaluator with the proposed PGD method can be trained with similar time to that of training encoder-decoder models. We apply the proposed framework to improve the general NMT models (e.g., Transformer). We conduct extensive experiments on two translation tasks, Chinese-English and English-German, and show that the proposed framework notably improves the performances of NMT models and significantly outperforms previous baselines.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.05414](https://arxiv.org/abs/2012.05414) [cs.CL]** |
|           | (or **[arXiv:2012.05414v1](https://arxiv.org/abs/2012.05414v1) [cs.CL]** for this version) |





<h2 id="2020-12-11-2">2. As good as new. How to successfully recycle English GPT-2 to make models for other languages</h2>

Title: [As good as new. How to successfully recycle English GPT-2 to make models for other languages](https://arxiv.org/abs/2012.05628)

Authors: [Wietse de Vries](https://arxiv.org/search/cs?searchtype=author&query=de+Vries%2C+W), [Malvina Nissim](https://arxiv.org/search/cs?searchtype=author&query=Nissim%2C+M)

> Large generative language models have been very successful for English, but other languages lag behind due to data and computational limitations. We propose a method that may overcome these problems by adapting existing pre-trained language models to new languages. Specifically, we describe the adaptation of English GPT-2 to Italian and Dutch by retraining lexical embeddings without tuning the Transformer layers. As a result, we obtain lexical embeddings for Italian and Dutch that are aligned with the original English lexical embeddings and induce a bilingual lexicon from this alignment. Additionally, we show how to scale up complexity by transforming relearned lexical embeddings of GPT-2 small to the GPT-2 medium embedding space. This method minimises the amount of training and prevents losing information during adaptation that was learned by GPT-2. English GPT-2 models with relearned lexical embeddings can generate realistic sentences in Italian and Dutch, but on average these sentences are still identifiable as artificial by humans. Based on perplexity scores and human judgements, we find that generated sentences become more realistic with some additional full model finetuning, especially for Dutch. For Italian, we see that they are evaluated on par with sentences generated by a GPT-2 model fully trained from scratch. Our work can be conceived as a blueprint for training GPT-2s for other languages, and we provide a 'recipe' to do so.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.05628](https://arxiv.org/abs/2012.05628) [cs.CL]** |
|           | (or **[arXiv:2012.05628v1](https://arxiv.org/abs/2012.05628v1) [cs.CL]** for this version) |





<h2 id="2020-12-11-3">3. Direct multimodal few-shot learning of speech and images</h2>

Title: [Direct multimodal few-shot learning of speech and images](https://arxiv.org/abs/2012.05680)

Authors: [Leanne Nortje](https://arxiv.org/search/cs?searchtype=author&query=Nortje%2C+L), [Herman Kamper](https://arxiv.org/search/cs?searchtype=author&query=Kamper%2C+H)

> We propose direct multimodal few-shot models that learn a shared embedding space of spoken words and images from only a few paired examples. Imagine an agent is shown an image along with a spoken word describing the object in the picture, e.g. pen, book and eraser. After observing a few paired examples of each class, the model is asked to identify the "book" in a set of unseen pictures. Previous work used a two-step indirect approach relying on learned unimodal representations: speech-speech and image-image comparisons are performed across the support set of given speech-image pairs. We propose two direct models which instead learn a single multimodal space where inputs from different modalities are directly comparable: a multimodal triplet network (MTriplet) and a multimodal correspondence autoencoder (MCAE). To train these direct models, we mine speech-image pairs: the support set is used to pair up unlabelled in-domain speech and images. In a speech-to-image digit matching task, direct models outperform indirect models, with the MTriplet achieving the best multimodal five-shot accuracy. We show that the improvements are due to the combination of unsupervised and transfer learning in the direct models, and the absence of two-step compounding errors.

| Comments: | 3 figures, 2 tables                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2012.05680](https://arxiv.org/abs/2012.05680) [cs.CL]** |
|           | (or **[arXiv:2012.05680v1](https://arxiv.org/abs/2012.05680v1) [cs.CL]** for this version) |





<h2 id="2020-12-11-4">4. Exploring Pair-Wise NMT for Indian Languages</h2>

Title: [Exploring Pair-Wise NMT for Indian Languages](https://arxiv.org/abs/2012.05786)

Authors: [Kartheek Akella](https://arxiv.org/search/cs?searchtype=author&query=Akella%2C+K), [Sai Himal Allu](https://arxiv.org/search/cs?searchtype=author&query=Allu%2C+S+H), [Sridhar Suresh Ragupathi](https://arxiv.org/search/cs?searchtype=author&query=Ragupathi%2C+S+S), [Aman Singhal](https://arxiv.org/search/cs?searchtype=author&query=Singhal%2C+A), [Zeeshan Khan](https://arxiv.org/search/cs?searchtype=author&query=Khan%2C+Z), [Vinay P. Namboodiri](https://arxiv.org/search/cs?searchtype=author&query=Namboodiri%2C+V+P), [C V Jawahar](https://arxiv.org/search/cs?searchtype=author&query=Jawahar%2C+C+V)

> In this paper, we address the task of improving pair-wise machine translation for specific low resource Indian languages. Multilingual NMT models have demonstrated a reasonable amount of effectiveness on resource-poor languages. In this work, we show that the performance of these models can be significantly improved upon by using back-translation through a filtered back-translation process and subsequent fine-tuning on the limited pair-wise language corpora. The analysis in this paper suggests that this method can significantly improve a multilingual model's performance over its baseline, yielding state-of-the-art results for various Indian languages.

| Comments: | ICON 2020 Short paper                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2012.05786](https://arxiv.org/abs/2012.05786) [cs.CL]** |
|           | (or **[arXiv:2012.05786v1](https://arxiv.org/abs/2012.05786v1) [cs.CL]** for this version) |





# 2020-12-10

[Return to Index](#Index)



<h2 id="2020-12-10-1">1. SongMASS: Automatic Song Writing with Pre-training and Alignment Constraint</h2>

Title: [SongMASS: Automatic Song Writing with Pre-training and Alignment Constraint](https://arxiv.org/abs/2012.05168)

Authors: [Zhonghao Sheng](https://arxiv.org/search/cs?searchtype=author&query=Sheng%2C+Z), [Kaitao Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+K), [Xu Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+X), [Yi Ren](https://arxiv.org/search/cs?searchtype=author&query=Ren%2C+Y), [Wei Ye](https://arxiv.org/search/cs?searchtype=author&query=Ye%2C+W), [Shikun Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+S), [Tao Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+T)

> Automatic song writing aims to compose a song (lyric and/or melody) by machine, which is an interesting topic in both academia and industry. In automatic song writing, lyric-to-melody generation and melody-to-lyric generation are two important tasks, both of which usually suffer from the following challenges: 1) the paired lyric and melody data are limited, which affects the generation quality of the two tasks, considering a lot of paired training data are needed due to the weak correlation between lyric and melody; 2) Strict alignments are required between lyric and melody, which relies on specific alignment modeling. In this paper, we propose SongMASS to address the above challenges, which leverages masked sequence to sequence (MASS) pre-training and attention based alignment modeling for lyric-to-melody and melody-to-lyric generation. Specifically, 1) we extend the original sentence-level MASS pre-training to song level to better capture long contextual information in music, and use a separate encoder and decoder for each modality (lyric or melody); 2) we leverage sentence-level attention mask and token-level attention constraint during training to enhance the alignment between lyric and melody. During inference, we use a dynamic programming strategy to obtain the alignment between each word/syllable in lyric and note in melody. We pre-train SongMASS on unpaired lyric and melody datasets, and both objective and subjective evaluations demonstrate that SongMASS generates lyric and melody with significantly better quality than the baseline method without pre-training or alignment constraint.

| Subjects: | **Sound (cs.SD)**; Computation and Language (cs.CL); Audio and Speech Processing (eess.AS) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.05168](https://arxiv.org/abs/2012.05168) [cs.SD]** |
|           | (or **[arXiv:2012.05168v1](https://arxiv.org/abs/2012.05168v1) [cs.SD]** for this version) |





<h2 id="2020-12-10-2">2. Breeding Gender-aware Direct Speech Translation Systems</h2>

Title: [Breeding Gender-aware Direct Speech Translation Systems](https://arxiv.org/abs/2012.04955)

Authors: [Marco Gaido](https://arxiv.org/search/cs?searchtype=author&query=Gaido%2C+M), [Beatrice Savoldi](https://arxiv.org/search/cs?searchtype=author&query=Savoldi%2C+B), [Luisa Bentivogli](https://arxiv.org/search/cs?searchtype=author&query=Bentivogli%2C+L), [Matteo Negri](https://arxiv.org/search/cs?searchtype=author&query=Negri%2C+M), [Marco Turchi](https://arxiv.org/search/cs?searchtype=author&query=Turchi%2C+M)

> In automatic speech translation (ST), traditional cascade approaches involving separate transcription and translation steps are giving ground to increasingly competitive and more robust direct solutions. In particular, by translating speech audio data without intermediate transcription, direct ST models are able to leverage and preserve essential information present in the input (e.g. speaker's vocal characteristics) that is otherwise lost in the cascade framework. Although such ability proved to be useful for gender translation, direct ST is nonetheless affected by gender bias just like its cascade counterpart, as well as machine translation and numerous other natural language processing applications. Moreover, direct ST systems that exclusively rely on vocal biometric features as a gender cue can be unsuitable and potentially harmful for certain users. Going beyond speech signals, in this paper we compare different approaches to inform direct ST models about the speaker's gender and test their ability to handle gender translation from English into Italian and French. To this aim, we manually annotated large datasets with speakers' gender information and used them for experiments reflecting different possible real-world scenarios. Our results show that gender-aware direct ST solutions can significantly outperform strong - but gender-unaware - direct ST models. In particular, the translation of gender-marked words can increase up to 30 points in accuracy while preserving overall translation quality.

| Comments:          | Outstanding paper at COLING 2020                             |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**                         |
| Journal reference: | In Proceedings of the 28th International Conference on Computational Linguistics, Dec 2020, 3951-3964. Online |
| Cite as:           | **[arXiv:2012.04955](https://arxiv.org/abs/2012.04955) [cs.CL]** |
|                    | (or **[arXiv:2012.04955v1](https://arxiv.org/abs/2012.04955v1) [cs.CL]** for this version) |







<h2 id="2020-12-10-3">3. On Knowledge Distillation for Direct Speech Translation</h2>

Title: [On Knowledge Distillation for Direct Speech Translation](https://arxiv.org/abs/2012.04964)

Authors: [Marco Gaido](https://arxiv.org/search/cs?searchtype=author&query=Gaido%2C+M), [Mattia A. Di Gangi](https://arxiv.org/search/cs?searchtype=author&query=Di+Gangi%2C+M+A), [Matteo Negri](https://arxiv.org/search/cs?searchtype=author&query=Negri%2C+M), [Marco Turchi](https://arxiv.org/search/cs?searchtype=author&query=Turchi%2C+M)

> Direct speech translation (ST) has shown to be a complex task requiring knowledge transfer from its sub-tasks: automatic speech recognition (ASR) and machine translation (MT). For MT, one of the most promising techniques to transfer knowledge is knowledge distillation. In this paper, we compare the different solutions to distill knowledge in a sequence-to-sequence task like ST. Moreover, we analyze eventual drawbacks of this approach and how to alleviate them maintaining the benefits in terms of translation quality.

| Comments: | Accepted at CLiC-IT 2020                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2012.04964](https://arxiv.org/abs/2012.04964) [cs.CL]** |
|           | (or **[arXiv:2012.04964v1](https://arxiv.org/abs/2012.04964v1) [cs.CL]** for this version) |







<h2 id="2020-12-10-4">4. Towards Zero-shot Cross-lingual Image Retrieval</h2>

Title: [Towards Zero-shot Cross-lingual Image Retrieval](https://arxiv.org/abs/2012.05107)

Authors: [Pranav Aggarwal](https://arxiv.org/search/cs?searchtype=author&query=Aggarwal%2C+P), [Ajinkya Kale](https://arxiv.org/search/cs?searchtype=author&query=Kale%2C+A)

> There has been a recent spike in interest in multi-modal Language and Vision problems. On the language side, most of these models primarily focus on English since most multi-modal datasets are monolingual. We try to bridge this gap with a zero-shot approach for learning multi-modal representations using cross-lingual pre-training on the text side. We present a simple yet practical approach for building a cross-lingual image retrieval model which trains on a monolingual training dataset but can be used in a zero-shot cross-lingual fashion during inference. We also introduce a new objective function which tightens the text embedding clusters by pushing dissimilar texts from each other. Finally, we introduce a new 1K multi-lingual MSCOCO2014 caption test dataset (XTD10) in 7 languages that we collected using a crowdsourcing platform. We use this as the test set for evaluating zero-shot model performance across languages. XTD10 dataset is made publicly available here: [this https URL](https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10)

| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.05107](https://arxiv.org/abs/2012.05107) [cs.CL]** |
|           | (or **[arXiv:2012.05107v1](https://arxiv.org/abs/2012.05107v1) [cs.CL]** for this version) |







# 2020-12-09

[Return to Index](#Index)



<h2 id="2020-12-09-1">1. Revisiting Iterative Back-Translation from the Perspective of Compositional Generalization</h2>

Title: [Revisiting Iterative Back-Translation from the Perspective of Compositional Generalization](https://arxiv.org/abs/2012.04276)

Authors: [Yinuo Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+Y), [Hualei Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+H), [Zeqi Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+Z), [Bei Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+B), [Jian-Guang Lou](https://arxiv.org/search/cs?searchtype=author&query=Lou%2C+J), [Dongmei Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+D)

> Human intelligence exhibits compositional generalization (i.e., the capacity to understand and produce unseen combinations of seen components), but current neural seq2seq models lack such ability. In this paper, we revisit iterative back-translation, a simple yet effective semi-supervised method, to investigate whether and how it can improve compositional generalization. In this work: (1) We first empirically show that iterative back-translation substantially improves the performance on compositional generalization benchmarks (CFQ and SCAN). (2) To understand why iterative back-translation is useful, we carefully examine the performance gains and find that iterative back-translation can increasingly correct errors in pseudo-parallel data. (3) To further encourage this mechanism, we propose curriculum iterative back-translation, which better improves the quality of pseudo-parallel data, thus further improving the performance.

| Comments: | accepted in AAAI 2021                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2012.04276](https://arxiv.org/abs/2012.04276) [cs.CL]** |
|           | (or **[arXiv:2012.04276v1](https://arxiv.org/abs/2012.04276v1) [cs.CL]** for this version) |





<h2 id="2020-12-09-2">2. Globetrotter: Unsupervised Multilingual Translation from Visual Alignment</h2>

Title: [Globetrotter: Unsupervised Multilingual Translation from Visual Alignment](https://arxiv.org/abs/2012.04631)

Authors: [Dídac Surís](https://arxiv.org/search/cs?searchtype=author&query=Surís%2C+D), [Dave Epstein](https://arxiv.org/search/cs?searchtype=author&query=Epstein%2C+D), [Carl Vondrick](https://arxiv.org/search/cs?searchtype=author&query=Vondrick%2C+C)

> Multi-language machine translation without parallel corpora is challenging because there is no explicit supervision between languages. Existing unsupervised methods typically rely on topological properties of the language representations. We introduce a framework that instead uses the visual modality to align multiple languages, using images as the bridge between them. We estimate the cross-modal alignment between language and images, and use this estimate to guide the learning of cross-lingual representations. Our language representations are trained jointly in one model with a single stage. Experiments with fifty-two languages show that our method outperforms baselines on unsupervised word-level and sentence-level translation using retrieval.

| Comments: | 19 pages, 9 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2012.04631](https://arxiv.org/abs/2012.04631) [cs.CL]** |
|           | (or **[arXiv:2012.04631v1](https://arxiv.org/abs/2012.04631v1) [cs.CL]** for this version) |





# 2020-12-08

[Return to Index](#Index)



<h2 id="2020-12-08-1">1. Cross-Modal Generalization: Learning in Low Resource Modalities via Meta-Alignment</h2>

Title: [Cross-Modal Generalization: Learning in Low Resource Modalities via Meta-Alignment](https://arxiv.org/abs/2012.02813)

Authors: [Paul Pu Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+P+P), [Peter Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+P), [Liu Ziyin](https://arxiv.org/search/cs?searchtype=author&query=Ziyin%2C+L), [Louis-Philippe Morency](https://arxiv.org/search/cs?searchtype=author&query=Morency%2C+L), [Ruslan Salakhutdinov](https://arxiv.org/search/cs?searchtype=author&query=Salakhutdinov%2C+R)

> The natural world is abundant with concepts expressed via visual, acoustic, tactile, and linguistic modalities. Much of the existing progress in multimodal learning, however, focuses primarily on problems where the same set of modalities are present at train and test time, which makes learning in low-resource modalities particularly difficult. In this work, we propose algorithms for cross-modal generalization: a learning paradigm to train a model that can (1) quickly perform new tasks in a target modality (i.e. meta-learning) and (2) doing so while being trained on a different source modality. We study a key research question: how can we ensure generalization across modalities despite using separate encoders for different source and target modalities? Our solution is based on meta-alignment, a novel method to align representation spaces using strongly and weakly paired cross-modal data while ensuring quick generalization to new tasks across different modalities. We study this problem on 3 classification tasks: text to image, image to audio, and text to speech. Our results demonstrate strong performance even when the new target modality has only a few (1-10) labeled samples and in the presence of noisy labels, a scenario particularly prevalent in low-resource modalities.

| Subjects: | **Machine Learning (cs.LG)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.02813](https://arxiv.org/abs/2012.02813) [cs.LG]** |
|           | (or **[arXiv:2012.02813v1](https://arxiv.org/abs/2012.02813v1) [cs.LG]** for this version) |





<h2 id="2020-12-08-2">2. MLS: A Large-Scale Multilingual Dataset for Speech Research</h2>

Title: [MLS: A Large-Scale Multilingual Dataset for Speech Research](https://arxiv.org/abs/2012.03411)

Authors: [Vineel Pratap](https://arxiv.org/search/eess?searchtype=author&query=Pratap%2C+V), [Qiantong Xu](https://arxiv.org/search/eess?searchtype=author&query=Xu%2C+Q), [Anuroop Sriram](https://arxiv.org/search/eess?searchtype=author&query=Sriram%2C+A), [Gabriel Synnaeve](https://arxiv.org/search/eess?searchtype=author&query=Synnaeve%2C+G), [Ronan Collobert](https://arxiv.org/search/eess?searchtype=author&query=Collobert%2C+R)

> This paper introduces Multilingual LibriSpeech (MLS) dataset, a large multilingual corpus suitable for speech research. The dataset is derived from read audiobooks from LibriVox and consists of 8 languages, including about 44.5K hours of English and a total of about 6K hours for other languages. Additionally, we provide Language Models (LM) and baseline Automatic Speech Recognition (ASR) models and for all the languages in our dataset. We believe such a large transcribed dataset will open new avenues in ASR and Text-To-Speech (TTS) research. The dataset will be made freely available for anyone at [this http URL](http://www.openslr.org/).

| Subjects:          | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL); Sound (cs.SD) |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | Interspeech 2020                                             |
| DOI:               | [10.21437/Interspeech.2020-2826](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.21437%2FInterspeech.2020-2826&v=ec9c71bb) |
| Cite as:           | **[arXiv:2012.03411](https://arxiv.org/abs/2012.03411) [eess.AS]** |
|                    | (or **[arXiv:2012.03411v1](https://arxiv.org/abs/2012.03411v1) [eess.AS]** for this version) |







<h2 id="2020-12-08-3">3. Reciprocal Supervised Learning Improves Neural Machine Translation</h2>

Title: [Reciprocal Supervised Learning Improves Neural Machine Translation](https://arxiv.org/abs/2012.02975)

Authors: [Minkai Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+M), [Mingxuan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+M), [Zhouhan Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+Z), [Hao Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+H), [Weinan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+W), [Lei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L)

> Despite the recent success on image classification, self-training has only achieved limited gains on structured prediction tasks such as neural machine translation (NMT). This is mainly due to the compositionality of the target space, where the far-away prediction hypotheses lead to the notorious reinforced mistake problem. In this paper, we revisit the utilization of multiple diverse models and present a simple yet effective approach named Reciprocal-Supervised Learning (RSL). RSL first exploits individual models to generate pseudo parallel data, and then cooperatively trains each model on the combined synthetic corpus. RSL leverages the fact that different parameterized models have different inductive biases, and better predictions can be made by jointly exploiting the agreement among each other. Unlike the previous knowledge distillation methods built upon a much stronger teacher, RSL is capable of boosting the accuracy of one model by introducing other comparable or even weaker models. RSL can also be viewed as a more efficient alternative to ensemble. Extensive experiments demonstrate the superior performance of RSL on several benchmarks with significant margins.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.02975](https://arxiv.org/abs/2012.02975) [cs.CL]** |
|           | (or **[arXiv:2012.02975v1](https://arxiv.org/abs/2012.02975v1) [cs.CL]** for this version) |







<h2 id="2020-12-08-4">4. Document Graph for Neural Machine Translation</h2>

Title: [Document Graph for Neural Machine Translation](https://arxiv.org/abs/2012.03477)

Authors: [Mingzhou Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+M), [Liangyou Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Derek. F. Wai](https://arxiv.org/search/cs?searchtype=author&query=Wai%2C+D+F), [Qun Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Q), [Lidia S. Chao](https://arxiv.org/search/cs?searchtype=author&query=Chao%2C+L+S)

> Previous works have shown that contextual information can improve the performance of neural machine translation (NMT). However, most existing document-level NMT methods failed to leverage contexts beyond a few set of previous sentences. How to make use of the whole document as global contexts is still a challenge. To address this issue, we hypothesize that a document can be represented as a graph that connects relevant contexts regardless of their distances. We employ several types of relations, including adjacency, syntactic dependency, lexical consistency, and coreference, to construct the document graph. Then, we incorporate both source and target graphs into the conventional Transformer architecture with graph convolutional networks. Experiments on various NMT benchmarks, including IWSLT English-French, Chinese-English, WMT English-German and Opensubtitle English-Russian, demonstrate that using document graphs can significantly improve the translation quality.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.03477](https://arxiv.org/abs/2012.03477) [cs.CL]** |
|           | (or **[arXiv:2012.03477v1](https://arxiv.org/abs/2012.03477v1) [cs.CL]** for this version) |







<h2 id="2020-12-08-5">5. KgPLM: Knowledge-guided Language Model Pre-training via Generative and Discriminative Learning</h2>

Title: [KgPLM: Knowledge-guided Language Model Pre-training via Generative and Discriminative Learning](https://arxiv.org/abs/2012.03551)

Authors: [Bin He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+B), [Xin Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+X), [Jinghui Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+J), [Qun Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Q)

> Recent studies on pre-trained language models have demonstrated their ability to capture factual knowledge and applications in knowledge-aware downstream tasks. In this work, we present a language model pre-training framework guided by factual knowledge completion and verification, and use the generative and discriminative approaches cooperatively to learn the model. Particularly, we investigate two learning schemes, named two-tower scheme and pipeline scheme, in training the generator and discriminator with shared parameter. Experimental results on LAMA, a set of zero-shot cloze-style question answering tasks, show that our model contains richer factual knowledge than the conventional pre-trained language models. Furthermore, when fine-tuned and evaluated on the MRQA shared tasks which consists of several machine reading comprehension datasets, our model achieves the state-of-the-art performance, and gains large improvements on NewsQA (+1.26 F1) and TriviaQA (+1.56 F1) over RoBERTa.

| Comments: | 10 pages, 3 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2012.03551](https://arxiv.org/abs/2012.03551) [cs.CL]** |
|           | (or **[arXiv:2012.03551v1](https://arxiv.org/abs/2012.03551v1) [cs.CL]** for this version) |





<h2 id="2020-12-08-6">6. PPKE: Knowledge Representation Learning by Path-based Pre-training</h2>

Title: [PPKE: Knowledge Representation Learning by Path-based Pre-training](https://arxiv.org/abs/2012.03573)

Authors: [Bin He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+B), [Di Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+D), [Jing Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+J), [Jinghui Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+J), [Xin Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+X), [Qun Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Q)

> Entities may have complex interactions in a knowledge graph (KG), such as multi-step relationships, which can be viewed as graph contextual information of the entities. Traditional knowledge representation learning (KRL) methods usually treat a single triple as a training unit, and neglect most of the graph contextual information exists in the topological structure of KGs. In this study, we propose a Path-based Pre-training model to learn Knowledge Embeddings, called PPKE, which aims to integrate more graph contextual information between entities into the KRL model. Experiments demonstrate that our model achieves state-of-the-art results on several benchmark datasets for link prediction and relation prediction tasks, indicating that our model provides a feasible way to take advantage of graph contextual information in KGs.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.03573](https://arxiv.org/abs/2012.03573) [cs.CL]** |
|           | (or **[arXiv:2012.03573v1](https://arxiv.org/abs/2012.03573v1) [cs.CL]** for this version) |







# 2020-12-07

[Return to Index](#Index)



<h2 id="2020-12-07-1">1. A Correspondence Variational Autoencoder for Unsupervised Acoustic Word Embeddings</h2>

Title: [A Correspondence Variational Autoencoder for Unsupervised Acoustic Word Embeddings](https://arxiv.org/abs/2012.02221)

Authors: [Puyuan Peng](https://arxiv.org/search/eess?searchtype=author&query=Peng%2C+P), [Herman Kamper](https://arxiv.org/search/eess?searchtype=author&query=Kamper%2C+H), [Karen Livescu](https://arxiv.org/search/eess?searchtype=author&query=Livescu%2C+K)

> We propose a new unsupervised model for mapping a variable-duration speech segment to a fixed-dimensional representation. The resulting acoustic word embeddings can form the basis of search, discovery, and indexing systems for low- and zero-resource languages. Our model, which we refer to as a maximal sampling correspondence variational autoencoder (MCVAE), is a recurrent neural network (RNN) trained with a novel self-supervised correspondence loss that encourages consistency between embeddings of different instances of the same word. Our training scheme improves on previous correspondence training approaches through the use and comparison of multiple samples from the approximate posterior distribution. In the zero-resource setting, the MCVAE can be trained in an unsupervised way, without any ground-truth word pairs, by using the word-like segments discovered via an unsupervised term discovery system. In both this setting and a semi-supervised low-resource setting (with a limited set of ground-truth word pairs), the MCVAE outperforms previous state-of-the-art models, such as Siamese-, CAE- and VAE-based RNNs.

| Comments: | 10 pages, 6 figures, NeurIPS 2020 Workshop Self-Supervised Learning for Speech and Audio Processing |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL); Sound (cs.SD) |
| Cite as:  | **[arXiv:2012.02221](https://arxiv.org/abs/2012.02221) [eess.AS]** |
|           | (or **[arXiv:2012.02221v1](https://arxiv.org/abs/2012.02221v1) [eess.AS]** for this version) |





<h2 id="2020-12-07-2">2. Self-Supervised VQA: Answering Visual Questions using Images and Captions
</h2>

Title: [Self-Supervised VQA: Answering Visual Questions using Images and Captions](https://arxiv.org/abs/2012.02356)

Authors: [Pratyay Banerjee](https://arxiv.org/search/cs?searchtype=author&query=Banerjee%2C+P), [Tejas Gokhale](https://arxiv.org/search/cs?searchtype=author&query=Gokhale%2C+T), [Yezhou Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Y), [Chitta Baral](https://arxiv.org/search/cs?searchtype=author&query=Baral%2C+C)

> Methodologies for training VQA models assume the availability of datasets with human-annotated Image-Question-Answer(I-Q-A) triplets for training. This has led to a heavy reliance and overfitting on datasets and a lack of generalization to new types of questions and scenes. Moreover, these datasets exhibit annotator subjectivity, biases, and errors, along with linguistic priors, which percolate into VQA models trained on such samples. We study whether models can be trained without any human-annotated Q-A pairs, but only with images and associated text captions which are descriptive and less subjective. We present a method to train models with procedurally generated Q-A pairs from captions using techniques, such as templates and annotation frameworks like QASRL. As most VQA models rely on dense and costly object annotations extracted from object detectors, we propose spatial-pyramid image patches as a simple but effective alternative to object bounding boxes, and demonstrate that our method uses fewer human annotations. We benchmark on VQA-v2, GQA, and on VQA-CP which contains a softer version of label shift. Our methods surpass prior supervised methods on VQA-CP and are competitive with methods without object features in fully supervised setting.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.02356](https://arxiv.org/abs/2012.02356) [cs.CV]** |
|           | (or **[arXiv:2012.02356v1](https://arxiv.org/abs/2012.02356v1) [cs.CV]** for this version) |





<h2 id="2020-12-07-3">3. Accurate and Scalable Matching of Translators to Displaced Persons for Overcoming Language Barriers</h2>

Title: [Accurate and Scalable Matching of Translators to Displaced Persons for Overcoming Language Barriers](https://arxiv.org/abs/2012.02595)

Authors: [Divyansh Agarwal](https://arxiv.org/search/cs?searchtype=author&query=Agarwal%2C+D), [Yuta Baba](https://arxiv.org/search/cs?searchtype=author&query=Baba%2C+Y), [Pratik Sachdeva](https://arxiv.org/search/cs?searchtype=author&query=Sachdeva%2C+P), [Tanya Tandon](https://arxiv.org/search/cs?searchtype=author&query=Tandon%2C+T), [Thomas Vetterli](https://arxiv.org/search/cs?searchtype=author&query=Vetterli%2C+T), [Aziz Alghunaim](https://arxiv.org/search/cs?searchtype=author&query=Alghunaim%2C+A)

> Residents of developing countries are disproportionately susceptible to displacement as a result of humanitarian crises. During such crises, language barriers impede aid workers in providing services to those displaced. To build resilience, such services must be flexible and robust to a host of possible languages. \textit{Tarjimly} aims to overcome the barriers by providing a platform capable of matching bilingual volunteers to displaced persons or aid workers in need of translating. However, Tarjimly's large pool of translators comes with the challenge of selecting the right translator per request. In this paper, we describe a machine learning system that matches translator requests to volunteers at scale. We demonstrate that a simple logistic regression, operating on easily computable features, can accurately predict and rank translator response. In deployment, this lightweight system matches 82\% of requests with a median response time of 59 seconds, allowing aid workers to accelerate their services supporting displaced persons.

| Comments: | Presented at NeurIPS 2020 Workshop on Machine Learning for the Developing World |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computers and Society (cs.CY)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2012.02595](https://arxiv.org/abs/2012.02595) [cs.CY]** |
|           | (or **[arXiv:2012.02595v1](https://arxiv.org/abs/2012.02595v1) [cs.CY]** for this version) |





<h2 id="2020-12-07-4">4. A Benchmark Dataset for Understandable Medical Language Translation</h2>

Title: [A Benchmark Dataset for Understandable Medical Language Translation](https://arxiv.org/abs/2012.02420)

Authors: [Junyu Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+J), [Zifei Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+Z), [Hanzhong Ye](https://arxiv.org/search/cs?searchtype=author&query=Ye%2C+H), [Muchao Ye](https://arxiv.org/search/cs?searchtype=author&query=Ye%2C+M), [Yaqing Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Quanzeng You](https://arxiv.org/search/cs?searchtype=author&query=You%2C+Q), [Cao Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+C), [Fenglong Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+F)

> In this paper, we introduce MedLane -- a new human-annotated Medical Language translation dataset, to align professional medical sentences with layperson-understandable expressions. The dataset contains 12,801 training samples, 1,015 validation samples, and 1,016 testing samples. We then evaluate one naive and six deep learning-based approaches on the MedLane dataset, including directly copying, a statistical machine translation approach Moses, four neural machine translation approaches (i.e., the proposed PMBERT-MT model, Seq2Seq and its two variants), and a modified text summarization model PointerNet. To compare the results, we utilize eleven metrics, including three new measures specifically designed for this task. Finally, we discuss the limitations of MedLane and baselines, and point out possible research directions for this task.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.02420](https://arxiv.org/abs/2012.02420) [cs.CL]** |
|           | (or **[arXiv:2012.02420v1](https://arxiv.org/abs/2012.02420v1) [cs.CL]** for this version) |





<h2 id="2020-12-07-5">5. Fine-tuning BERT for Low-Resource Natural Language Understanding via Active Learning</h2>

Title: [Fine-tuning BERT for Low-Resource Natural Language Understanding via Active Learning](https://arxiv.org/abs/2012.02462)

Authors: [Daniel Grießhaber](https://arxiv.org/search/cs?searchtype=author&query=Grießhaber%2C+D), [Johannes Maucher](https://arxiv.org/search/cs?searchtype=author&query=Maucher%2C+J), [Ngoc Thang Vu](https://arxiv.org/search/cs?searchtype=author&query=Vu%2C+N+T)

> Recently, leveraging pre-trained Transformer based language models in down stream, task specific models has advanced state of the art results in natural language understanding tasks. However, only a little research has explored the suitability of this approach in low resource settings with less than 1,000 training data points. In this work, we explore fine-tuning methods of BERT -- a pre-trained Transformer based language model -- by utilizing pool-based active learning to speed up training while keeping the cost of labeling new data constant. Our experimental results on the GLUE data set show an advantage in model performance by maximizing the approximate knowledge gain of the model when querying from the pool of unlabeled data. Finally, we demonstrate and analyze the benefits of freezing layers of the language model during fine-tuning to reduce the number of trainable parameters, making it more suitable for low-resource settings.

| Comments: | COLING'2020                                                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2012.02462](https://arxiv.org/abs/2012.02462) [cs.CL]** |
|           | (or **[arXiv:2012.02462v1](https://arxiv.org/abs/2012.02462v1) [cs.CL]** for this version) |





# 2020-12-04

[Return to Index](#Index)



<h2 id="2020-12-04-1">1. SemMT: A Semantic-based Testing Approach for Machine Translation Systems</h2>

Title: [SemMT: A Semantic-based Testing Approach for Machine Translation Systems](https://arxiv.org/abs/2012.01815)

Authors: [Jialun Cao](https://arxiv.org/search/cs?searchtype=author&query=Cao%2C+J), [Meiziniu Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+M), [Yeting Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Ming Wen](https://arxiv.org/search/cs?searchtype=author&query=Wen%2C+M), [Shing-Chi Cheung](https://arxiv.org/search/cs?searchtype=author&query=Cheung%2C+S)

> Machine translation has wide applications in daily life. In mission-critical applications such as translating official documents, incorrect translation can have unpleasant or sometimes catastrophic consequences. This motivates recent research on testing methodologies for machine translation systems. Existing methodologies mostly rely on metamorphic relations designed at the textual level (e.g., Levenshtein distance) or syntactic level (e.g., the distance between grammar structures) to determine the correctness of translation results. However, these metamorphic relations do not consider whether the original and translated sentences have the same meaning (i.e., Semantic similarity). Therefore, in this paper, we propose SemMT, an automatic testing approach for machine translation systems based on semantic similarity checking. SemMT applies round-trip translation and measures the semantic similarity between the original and translated sentences. Our insight is that the semantics expressed by the logic and numeric constraint in sentences can be captured using regular expressions (or deterministic finite automata) where efficient equivalence/similarity checking algorithms are available. Leveraging the insight, we propose three semantic similarity metrics and implement them in SemMT. The experiment result reveals SemMT can achieve higher effectiveness compared with state-of-the-art works, achieving an increase of 21% and 23% on accuracy and F-Score, respectively. We also explore potential improvements that can be achieved when proper combinations of metrics are adopted. Finally, we discuss a solution to locate the suspicious trip in round-trip translation, which may shed lights on further exploration.

| Subjects: | **Software Engineering (cs.SE)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.01815](https://arxiv.org/abs/2012.01815) [cs.SE]** |
|           | (or **[arXiv:2012.01815v1](https://arxiv.org/abs/2012.01815v1) [cs.SE]** for this version) |





<h2 id="2020-12-04-2">2. Self-Explaining Structures Improve NLP Models</h2>

Title: [Self-Explaining Structures Improve NLP Models](https://arxiv.org/abs/2012.01786)

Authors: [Zijun Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+Z), [Chun Fan](https://arxiv.org/search/cs?searchtype=author&query=Fan%2C+C), [Qinghong Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+Q), [Xiaofei Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+X), [Yuxian Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+Y), [Fei Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+F), [Jiwei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J)

> Existing approaches to explaining deep learning models in NLP usually suffer from two major drawbacks: (1) the main model and the explaining model are decoupled: an additional probing or surrogate model is used to interpret an existing model, and thus existing explaining tools are not self-explainable; (2) the probing model is only able to explain a model's predictions by operating on low-level features by computing saliency scores for individual words but are clumsy at high-level text units such as phrases, sentences, or paragraphs. To deal with these two issues, in this paper, we propose a simple yet general and effective self-explaining framework for deep learning models in NLP. The key point of the proposed framework is to put an additional layer, as is called by the interpretation layer, on top of any existing NLP model. This layer aggregates the information for each text span, which is then associated with a specific weight, and their weighted combination is fed to the softmax function for the final prediction. The proposed model comes with the following merits: (1) span weights make the model self-explainable and do not require an additional probing model for interpretation; (2) the proposed model is general and can be adapted to any existing deep learning structures in NLP; (3) the weight associated with each text span provides direct importance scores for higher-level text units such as phrases and sentences. We for the first time show that interpretability does not come at the cost of performance: a neural model of self-explaining features obtains better performances than its counterpart without the self-explaining nature, achieving a new SOTA performance of 59.1 on SST-5 and a new SOTA performance of 92.3 on SNLI.

| Comments: | Code is available at [this https URL](https://github.com/ShannonAI/Self_Explaining_Structures_Improve_NLP_Models) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2012.01786](https://arxiv.org/abs/2012.01786) [cs.CL]** |
|           | (or **[arXiv:2012.01786v1](https://arxiv.org/abs/2012.01786v1) [cs.CL]** for this version) |





<h2 id="2020-12-04-3">3. On Extending NLP Techniques from the Categorical to the Latent Space: KL Divergence, Zipf's Law, and Similarity Search</h2>

Title: [On Extending NLP Techniques from the Categorical to the Latent Space: KL Divergence, Zipf's Law, and Similarity Search](https://arxiv.org/abs/2012.01941)

Authors: [Adam Hare](https://arxiv.org/search/cs?searchtype=author&query=Hare%2C+A), [Yu Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Yinan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Zhenming Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Christopher G. Brinton](https://arxiv.org/search/cs?searchtype=author&query=Brinton%2C+C+G)

> Despite the recent successes of deep learning in natural language processing (NLP), there remains widespread usage of and demand for techniques that do not rely on machine learning. The advantage of these techniques is their interpretability and low cost when compared to frequently opaque and expensive machine learning models. Although they may not be be as performant in all cases, they are often sufficient for common and relatively simple problems. In this paper, we aim to modernize these older methods while retaining their advantages by extending approaches from categorical or bag-of-words representations to word embeddings representations in the latent space. First, we show that entropy and Kullback-Leibler divergence can be efficiently estimated using word embeddings and use this estimation to compare text across several categories. Next, we recast the heavy-tailed distribution known as Zipf's law that is frequently observed in the categorical space to the latent space. Finally, we look to improve the Jaccard similarity measure for sentence suggestion by introducing a new method of identifying similar sentences based on the set cover problem. We compare the performance of this algorithm against several baselines including Word Mover's Distance and the Levenshtein distance.

| Comments: | 13 pages, 6 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2012.01941](https://arxiv.org/abs/2012.01941) [cs.CL]** |
|           | (or **[arXiv:2012.01941v1](https://arxiv.org/abs/2012.01941v1) [cs.CL]** for this version) |









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



