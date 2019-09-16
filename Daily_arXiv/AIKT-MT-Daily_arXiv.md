# Daily arXiv: Machine Translation - Sep., 2019

# Index

- [2019-09-16](#2019-09-16)
  - [1. CTRL: A Conditional Transformer Language Model for Controllable Generation](#2019-09-16-1)
  - [2. Sequence-to-sequence Pre-training with Data Augmentation for Sentence Rewriting](#2019-09-16-2)
  - [3. Neural Machine Translation with 4-Bit Precision and Beyond](#2019-09-16-3)
  - [4. A General Framework for Implicit and Explicit Debiasing of Distributional Word Vector Spaces](#2019-09-16-4)
- [2019-09-13](#2019-09-13)
  - [1. Entity Projection via Machine-Translation for Cross-Lingual NER](#2019-09-13-1)
  - [2. Problems with automating translation of movie/TV show subtitles](#2019-09-13-2)
  - [3. Speculative Beam Search for Simultaneous Translation](#2019-09-13-3)
  - [4. VizSeq: A Visual Analysis Toolkit for Text Generation Tasks](#2019-09-13-4)
  - [5. Neural Semantic Parsing in Low-Resource Settings with Back-Translation and Meta-Learning](#2019-09-13-5)
  - [6. Lost in Evaluation: Misleading Benchmarks for Bilingual Dictionary Induction](#2019-09-13-6)
- [2019-09-10](#2019-09-10)
  - [1. Improving Neural Machine Translation with Parent-Scaled Self-Attention](#2019-09-10-1)
  - [2. LAMAL: LAnguage Modeling Is All You Need for Lifelong Language Learning](#2019-09-10-2)
  - [3. Neural Machine Translation with Byte-Level Subwords](#2019-09-10-3)
  - [4. Combining SMT and NMT Back-Translated Data for Efficient NMT](#2019-09-10-4)
- [2019-09-09](#2019-09-09)
  - [1. Don't Forget the Long Tail! A Comprehensive Analysis of Morphological Generalization in Bilingual Lexicon Induction](#2019-09-09-1)
- [2019-09-06](#2019-09-06)
  - [1. Jointly Learning to Align and Translate with Transformer Models](#2019-09-06-1)
  - [2. Investigating Multilingual NMT Representations at Scale](#2019-09-06-2)
  - [3. Multi-Granularity Self-Attention for Neural Machine Translation](#2019-09-06-3)
  - [4. Source Dependency-Aware Transformer with Supervised Self-Attention](#2019-09-06-4)
  - [5. Accelerating Transformer Decoding via a Hybrid of Self-attention and Recurrent Neural Network](#2019-09-06-1)
  - [6. FlowSeq: Non-Autoregressive Conditional Sequence Generation with Generative Flow](#2019-09-06-6)
- [2019-09-05](#2019-09-05)
  - [1. The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives](#2019-09-05-1)
  - [2. Context-Aware Monolingual Repair for Neural Machine Translation](#2019-09-05-2)
  - [3. Simpler and Faster Learning of Adaptive Policies for Simultaneous Translation](#2019-09-05-3)
  - [4. Do We Really Need Fully Unsupervised Cross-Lingual Embeddings?](#2019-09-05-4)
  - [5. SAO WMT19 Test Suite: Machine Translation of Audit Reports](#2019-09-05-5)
- [2019-09-04](#2019-09-04)
  - [1. Hybrid Data-Model Parallel Training for Sequence-to-Sequence Recurrent Neural Network Machine Translation](#2019-09-04-1)
  - [2. Handling Syntactic Divergence in Low-resource Machine Translation](#2019-09-04-2)
  - [3. Evaluating Pronominal Anaphora in Machine Translation: An Evaluation Measure and a Test Suite](#2019-09-04-3)
  - [4. Improving Back-Translation with Uncertainty-based Confidence Estimation](#2019-09-04-4)
  - [5. Explicit Cross-lingual Pre-training for Unsupervised Machine Translation](#2019-09-04-5)
  - [6. Towards Understanding Neural Machine Translation with Word Importance](#2019-09-04-6)
  - [7. One Model to Learn Both: Zero Pronoun Prediction and Translation](#2019-09-04-7)
  - [8. Evaluating the Cross-Lingual Effectiveness of Massively Multilingual Neural Machine Translation](#2019-09-04-8)
  - [9. Improving Context-aware Neural Machine Translation with Target-side Context](#2019-09-04-9)
  - [10. Enhancing Context Modeling with a Query-Guided Capsule Network for Document-level Translation](#2019-09-04-10)
  - [11. Unicoder: A Universal Language Encoder by Pre-training with Multiple Cross-lingual Tasks](#2019-09-04-11)
  - [12. Multi-agent Learning for Neural Machine Translation](#2019-09-04-12)
  - [13. Bilingual is At Least Monolingual (BALM): A Novel Translation Algorithm that Encodes Monolingual Priors](#2019-09-04-13)
- [2019-09-02](#2019-09-02)
  - [1. Latent Part-of-Speech Sequences for Neural Machine Translation](#2019-09-02-1)
  - [2. Encoders Help You Disambiguate Word Senses in Neural Machine Translation](#2019-09-02-2)
* [2019-08](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-08.md)
* [2019-07](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-07.md)
* [2019-06](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-06.md)
* [2019-05](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-05.md)
* [2019-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-04.md)
* [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)



# 2019-09-16

[Return to Index](#Index)



<h2 id="2019-09-16-1">1. CTRL: A Conditional Transformer Language Model for Controllable Generation</h2> 

Title: [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858)

Authors: [Nitish Shirish Keskar](https://arxiv.org/search/cs?searchtype=author&query=Keskar%2C+N+S), [Bryan McCann](https://arxiv.org/search/cs?searchtype=author&query=McCann%2C+B), [Lav R. Varshney](https://arxiv.org/search/cs?searchtype=author&query=Varshney%2C+L+R), [Caiming Xiong](https://arxiv.org/search/cs?searchtype=author&query=Xiong%2C+C), [Richard Socher](https://arxiv.org/search/cs?searchtype=author&query=Socher%2C+R)

*(Submitted on 11 Sep 2019)*

> Large-scale language models show promising text generation capabilities, but users cannot easily control particular aspects of the generated text. We release CTRL, a 1.6 billion-parameter conditional transformer language model, trained to condition on control codes that govern style, content, and task-specific behavior. Control codes were derived from structure that naturally co-occurs with raw text, preserving the advantages of unsupervised learning while providing more explicit control over text generation. These codes also allow CTRL to predict which parts of the training data are most likely given a sequence. This provides a potential method for analyzing large amounts of data via model-based source attribution. We have released multiple full-sized, pretrained versions of CTRL at [this http URL](http://github.com/salesforce/ctrl).

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1909.05858 [cs.CL]**                         |
|           | (or **arXiv:1909.05858v1 [cs.CL]** for this version) |



<h2 id="2019-09-16-2">2. Sequence-to-sequence Pre-training with Data Augmentation for Sentence Rewriting</h2> 

Title: [Sequence-to-sequence Pre-training with Data Augmentation for Sentence Rewriting](https://arxiv.org/abs/1909.06002)

Authors: [Yi Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Tao Ge](https://arxiv.org/search/cs?searchtype=author&query=Ge%2C+T), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F), [Ming Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+M), [Xu Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+X)

*(Submitted on 13 Sep 2019)*

> We study sequence-to-sequence (seq2seq) pre-training with data augmentation for sentence rewriting. Instead of training a seq2seq model with gold training data and augmented data simultaneously, we separate them to train in different phases: pre-training with the augmented data and fine-tuning with the gold data. We also introduce multiple data augmentation methods to help model pre-training for sentence rewriting. We evaluate our approach in two typical well-defined sentence rewriting tasks: Grammatical Error Correction (GEC) and Formality Style Transfer (FST). Experiments demonstrate our approach can better utilize augmented data without hurting the model's trust in gold data and further improve the model's performance with our proposed data augmentation methods.
> Our approach substantially advances the state-of-the-art results in well-recognized sentence rewriting benchmarks over both GEC and FST. Specifically, it pushes the CoNLL-2014 benchmark's F0.5 score and JFLEG Test GLEU score to 62.61 and 63.54 in the restricted training setting, 66.77 and 65.22 respectively in the unrestricted setting, and advances GYAFC benchmark's BLEU to 74.24 (2.23 absolute improvement) in E&M domain and 77.97 (2.64 absolute improvement) in F&R domain.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1909.06002 [cs.CL]**                         |
|           | (or **arXiv:1909.06002v1 [cs.CL]** for this version) |





<h2 id="2019-09-16-3">3. Neural Machine Translation with 4-Bit Precision and Beyond</h2> 

Title: [Neural Machine Translation with 4-Bit Precision and Beyond](https://arxiv.org/abs/1909.06091)

Authors: [Alham Fikri Aji](https://arxiv.org/search/cs?searchtype=author&query=Aji%2C+A+F), [Kenneth Heafield](https://arxiv.org/search/cs?searchtype=author&query=Heafield%2C+K)

*(Submitted on 13 Sep 2019)*

> Neural Machine Translation (NMT) is resource intensive. We design a quantization procedure to compress fit NMT models better for devices with limited hardware capability. We use logarithmic quantization, instead of the more commonly used fixed-point quantization, based on the empirical fact that parameters distribution is not uniform. We find that biases do not take a lot of memory and show that biases can be left uncompressed to improve the overall quality without affecting the compression rate. We also propose to use an error-feedback mechanism during retraining, to preserve the compressed model as a stale gradient. We empirically show that NMT models based on Transformer or RNN architecture can be compressed up to 4-bit precision without any noticeable quality degradation. Models can be compressed up to binary precision, albeit with lower quality. RNN architecture seems to be more robust towards compression, compared to the Transformer.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1909.06091 [cs.CL]**                                 |
|           | (or **arXiv:1909.06091v1 [cs.CL]** for this version)         |





<h2 id="2019-09-16-4">4. A General Framework for Implicit and Explicit Debiasing of Distributional Word Vector Spaces</h2> 

Title: [A General Framework for Implicit and Explicit Debiasing of Distributional Word Vector Spaces](https://arxiv.org/abs/1909.06092)

Authors: [Anne Lauscher](https://arxiv.org/search/cs?searchtype=author&query=Lauscher%2C+A), [Goran Glavaš](https://arxiv.org/search/cs?searchtype=author&query=Glavaš%2C+G), [Simone Paolo Ponzetto](https://arxiv.org/search/cs?searchtype=author&query=Ponzetto%2C+S+P), [Ivan Vulić](https://arxiv.org/search/cs?searchtype=author&query=Vulić%2C+I)

*(Submitted on 13 Sep 2019)*

> Distributional word vectors have recently been shown to encode many of the human biases, most notably gender and racial biases, and models for attenuating such biases have consequently been proposed. However, existing models and studies (1) operate on under-specified and mutually differing bias definitions, (2) are tailored for a particular bias (e.g., gender bias) and (3) have been evaluated inconsistently and non-rigorously. In this work, we introduce a general framework for debiasing word embeddings. We operationalize the definition of a bias by discerning two types of bias specification: explicit and implicit. We then propose three debiasing models that operate on explicit or implicit bias specifications, and that can be composed towards more robust debiasing. Finally, we devise a full-fledged evaluation framework in which we couple existing bias metrics with newly proposed ones. Experimental findings across three embedding methods suggest that the proposed debiasing models are robust and widely applicable: they often completely remove the bias both implicitly and explicitly, without degradation of semantic information encoded in any of the input distributional spaces. Moreover, we successfully transfer debiasing models, by means of crosslingual embedding spaces, and remove or attenuate biases in distributional word vector spaces of languages that lack readily available bias specifications.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1909.06092 [cs.CL]**                                 |
|           | (or **arXiv:1909.06092v1 [cs.CL]** for this version)         |





# 2019-09-13

[Return to Index](#Index)



<h2 id="2019-09-13-1">1. Entity Projection via Machine-Translation for Cross-Lingual NER</h2> 
Title: [Entity Projection via Machine-Translation for Cross-Lingual NER](https://arxiv.org/abs/1909.05356)

Authors: [Alankar Jain](https://arxiv.org/search/cs?searchtype=author&query=Jain%2C+A), [Bhargavi Paranjape](https://arxiv.org/search/cs?searchtype=author&query=Paranjape%2C+B), [Zachary C. Lipton](https://arxiv.org/search/cs?searchtype=author&query=Lipton%2C+Z+C)

*(Submitted on 31 Aug 2019)*

> Although over 100 languages are supported by strong off-the-shelf machine translation systems, only a subset of them possess large annotated corpora for named entity recognition. Motivated by this fact, we leverage machine translation to improve annotation-projection approaches to cross-lingual named entity recognition. We propose a system that improves over prior entity-projection methods by: (a) leveraging machine translation systems twice: first for translating sentences and subsequently for translating entities; (b) matching entities based on orthographic and phonetic similarity; and (c) identifying matches based on distributional statistics derived from the dataset. Our approach improves upon current state-of-the-art methods for cross-lingual named entity recognition on 5 diverse languages by an average of 4.1 points. Further, our method achieves state-of-the-art F_1 scores for Armenian, outperforming even a monolingual model trained on Armenian source data.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1909.05356 [cs.CL]**                                 |
|           | (or **arXiv:1909.05356v1 [cs.CL]** for this version)         |





<h2 id="2019-09-13-2">2. Problems with automating translation of movie/TV show subtitles</h2> 
Title: [Problems with automating translation of movie/TV show subtitles](https://arxiv.org/abs/1909.05362)

Authors: [Prabhakar Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+P), [Mayank Sharma](https://arxiv.org/search/cs?searchtype=author&query=Sharma%2C+M), [Kartik Pitale](https://arxiv.org/search/cs?searchtype=author&query=Pitale%2C+K), [Keshav Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+K)

*(Submitted on 4 Sep 2019)*

> We present 27 problems encountered in automating the translation of movie/TV show subtitles. We categorize each problem in one of the three categories viz. problems directly related to textual translation, problems related to subtitle creation guidelines, and problems due to adaptability of machine translation (MT) engines. We also present the findings of a translation quality evaluation experiment where we share the frequency of 16 key problems. We show that the systems working at the frontiers of Natural Language Processing do not perform well for subtitles and require some post-processing solutions for redressal of these problems

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1909.05362 [cs.CL]**                                 |
|           | (or **arXiv:1909.05362v1 [cs.CL]** for this version)         |



<h2 id="2019-09-13-3">3. Speculative Beam Search for Simultaneous Translation</h2> 
Title: [Speculative Beam Search for Simultaneous Translation](https://arxiv.org/abs/1909.05421)

Authors: [Renjie Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+R), [Mingbo Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+M), [Baigong Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+B), [Liang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+L)

*(Submitted on 12 Sep 2019)*

> Beam search is universally used in full-sentence translation but its application to simultaneous translation remains non-trivial, where output words are committed on the fly. In particular, the recently proposed wait-k policy (Ma et al., 2019a) is a simple and effective method that (after an initial wait) commits one output word on receiving each input word, making beam search seemingly impossible. To address this challenge, we propose a speculative beam search algorithm that hallucinates several steps into the future in order to reach a more accurate decision, implicitly benefiting from a target language model. This makes beam search applicable for the first time to the generation of a single word in each step. Experiments over diverse language pairs show large improvements over previous work.

| Comments: | accepted by EMNLP 2019                               |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.05421 [cs.CL]**                         |
|           | (or **arXiv:1909.05421v1 [cs.CL]** for this version) |





<h2 id="2019-09-13-4">4. VizSeq: A Visual Analysis Toolkit for Text Generation Tasks</h2> 
Title: [VizSeq: A Visual Analysis Toolkit for Text Generation Tasks](https://arxiv.org/abs/1909.05424)

Authors: [Changhan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Anirudh Jain](https://arxiv.org/search/cs?searchtype=author&query=Jain%2C+A), [Danlu Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+D), [Jiatao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+J)

*(Submitted on 12 Sep 2019)*

> Automatic evaluation of text generation tasks (e.g. machine translation, text summarization, image captioning and video description) usually relies heavily on task-specific metrics, such as BLEU and ROUGE. They, however, are abstract numbers and are not perfectly aligned with human assessment. This suggests inspecting detailed examples as a complement to identify system error patterns. In this paper, we present VizSeq, a visual analysis toolkit for instance-level and corpus-level system evaluation on a wide variety of text generation tasks. It supports multimodal sources and multiple text references, providing visualization in Jupyter notebook or a web app interface. It can be used locally or deployed onto public servers for centralized data hosting and benchmarking. It covers most common n-gram based metrics accelerated with multiprocessing, and also provides latest embedding-based metrics such as BERTScore.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1909.05424 [cs.CL]**                         |
|           | (or **arXiv:1909.05424v1 [cs.CL]** for this version) |





<h2 id="2019-09-13-5">5. Neural Semantic Parsing in Low-Resource Settings with Back-Translation and Meta-Learning</h2> 
Title: [Neural Semantic Parsing in Low-Resource Settings with Back-Translation and Meta-Learning](https://arxiv.org/abs/1909.05438)

Authors: [Yibo Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+Y), [Duyu Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+D), [Nan Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan%2C+N), [Yeyun Gong](https://arxiv.org/search/cs?searchtype=author&query=Gong%2C+Y), [Xiaocheng Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+X), [Bing Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+B), [Daxin Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+D)

*(Submitted on 12 Sep 2019)*

> Neural semantic parsing has achieved impressive results in recent years, yet its success relies on the availability of large amounts of supervised data. Our goal is to learn a neural semantic parser when only prior knowledge about a limited number of simple rules is available, without access to either annotated programs or execution results. Our approach is initialized by rules, and improved in a back-translation paradigm using generated question-program pairs from the semantic parser and the question generator. A phrase table with frequent mapping patterns is automatically derived, also updated as training progresses, to measure the quality of generated instances. We train the model with model-agnostic meta-learning to guarantee the accuracy and stability on examples covered by rules, and meanwhile acquire the versatility to generalize well on examples uncovered by rules. Results on three benchmark datasets with different domains and programs show that our approach incrementally improves the accuracy. On WikiSQL, our best model is comparable to the SOTA system learned from denotations.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1909.05438 [cs.CL]**                         |
|           | (or **arXiv:1909.05438v1 [cs.CL]** for this version) |





<h2 id="2019-09-13-6">6. Lost in Evaluation: Misleading Benchmarks for Bilingual Dictionary Induction</h2> 
Title: [Lost in Evaluation: Misleading Benchmarks for Bilingual Dictionary Induction](https://arxiv.org/abs/1909.05708)

Authors: [Yova Kementchedjhieva](https://arxiv.org/search/cs?searchtype=author&query=Kementchedjhieva%2C+Y), [Mareike Hartmann](https://arxiv.org/search/cs?searchtype=author&query=Hartmann%2C+M), [Anders Søgaard](https://arxiv.org/search/cs?searchtype=author&query=Søgaard%2C+A)

*(Submitted on 12 Sep 2019)*

> The task of bilingual dictionary induction (BDI) is commonly used for intrinsic evaluation of cross-lingual word embeddings. The largest dataset for BDI was generated automatically, so its quality is dubious. We study the composition and quality of the test sets for five diverse languages from this dataset, with concerning findings: (1) a quarter of the data consists of proper nouns, which can be hardly indicative of BDI performance, and (2) there are pervasive gaps in the gold-standard targets. These issues appear to affect the ranking between cross-lingual embedding systems on individual languages, and the overall degree to which the systems differ in performance. With proper nouns removed from the data, the margin between the top two systems included in the study grows from 3.4% to 17.2%. Manual verification of the predictions, on the other hand, reveals that gaps in the gold standard targets artificially inflate the margin between the two systems on English to Bulgarian BDI from 0.1% to 6.7%. We thus suggest that future research either avoids drawing conclusions from quantitative results on this BDI dataset, or accompanies such evaluation with rigorous error analysis.

| Comments: | Accepted at EMNLP 2019                               |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.05708 [cs.CL]**                         |
|           | (or **arXiv:1909.05708v1 [cs.CL]** for this version) |



# 2019-09-12

[Return to Index](#Index)



<h2 id="2019-09-12-1">1. A Quantum Search Decoder for Natural Language Processing</h2> 
Title: [A Quantum Search Decoder for Natural Language Processing](https://arxiv.org/abs/1909.05023)

Authors:[Johannes Bausch](https://arxiv.org/search/quant-ph?searchtype=author&query=Bausch%2C+J), [Sathyawageeswar Subramanian](https://arxiv.org/search/quant-ph?searchtype=author&query=Subramanian%2C+S), [Stephen Piddock](https://arxiv.org/search/quant-ph?searchtype=author&query=Piddock%2C+S)

*(Submitted on 9 Sep 2019)*

> Probabilistic language models, e.g. those based on an LSTM, often face the problem of finding a high probability prediction from a sequence of random variables over a set of words. This is commonly addressed using a form of greedy decoding such as beam search, where a limited number of highest-likelihood paths (the beam width) of the decoder are kept, and at the end the maximum-likelihood path is chosen. The resulting algorithm has linear runtime in the beam width. However, the input is not necessarily distributed such that a high-likelihood input symbol at any given time step also leads to the global optimum. Limiting the beam width can thus result in a failure to recognise long-range dependencies. In practice, only an exponentially large beam width can guarantee that the global optimum is found: for an input of length n and average parser branching ratio R, the baseline classical algorithm needs to query the input on average Rn times. In this work, we construct a quantum algorithm to find the globally optimal parse with high constant success probability. Given the input to the decoder is distributed like a power-law with exponent k>0, our algorithm yields a runtime Rnf(R,k), where f≤1/2, and f→0 exponentially quickly for growing k. This implies that our algorithm always yields a super-Grover type speedup, i.e. it is more than quadratically faster than its classical counterpart. We further modify our procedure to recover a quantum beam search variant, which enables an even stronger empirical speedup, while sacrificing accuracy. Finally, we apply this quantum beam search decoder to Mozilla's implementation of Baidu's DeepSpeech neural net, which we show to exhibit such a power law word rank frequency, underpinning the applicability of our model.

| Comments:    | 36 pages, 9 figures                                          |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Quantum Physics (quant-ph)**; Computation and Language (cs.CL); Data Structures and Algorithms (cs.DS); Machine Learning (cs.LG) |
| MSC classes: | 68T50, 68Q12, 68T05                                          |
| Cite as:     | **arXiv:1909.05023 [quant-ph]**                              |
|              | (or **arXiv:1909.05023v1 [quant-ph]** for this version)      |





<h2 id="2019-09-12-2">2. MultiFiT: Efficient Multi-lingual Language Model Fine-tuning</h2> 
Title: [MultiFiT: Efficient Multi-lingual Language Model Fine-tuning](https://arxiv.org/abs/1909.04761)

Authors:[Julian Eisenschlos](https://arxiv.org/search/cs?searchtype=author&query=Eisenschlos%2C+J), [Sebastian Ruder](https://arxiv.org/search/cs?searchtype=author&query=Ruder%2C+S), [Piotr Czapla](https://arxiv.org/search/cs?searchtype=author&query=Czapla%2C+P), [Marcin Kardas](https://arxiv.org/search/cs?searchtype=author&query=Kardas%2C+M), [Sylvain Gugger](https://arxiv.org/search/cs?searchtype=author&query=Gugger%2C+S), [Jeremy Howard](https://arxiv.org/search/cs?searchtype=author&query=Howard%2C+J)

*(Submitted on 10 Sep 2019)*

> Pretrained language models are promising particularly for low-resource languages as they only require unlabelled data. However, training existing models requires huge amounts of compute, while pretrained cross-lingual models often underperform on low-resource languages. We propose Multi-lingual language model Fine-Tuning (MultiFiT) to enable practitioners to train and fine-tune language models efficiently in their own language. In addition, we propose a zero-shot method using an existing pretrained cross-lingual model. We evaluate our methods on two widely used cross-lingual classification datasets where they outperform models pretrained on orders of magnitude more data and compute. We release all models and code.

| Comments: | Proceedings of EMNLP-IJCNLP 2019                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1909.04761 [cs.CL]**                                 |
|           | (or **arXiv:1909.04761v1 [cs.CL]** for this version)         |





<h2 id="2019-09-12-3">3. Dynamic Fusion: Attentional Language Model for Neural Machine Translation</h2> 
Title: [Dynamic Fusion: Attentional Language Model for Neural Machine Translation](https://arxiv.org/abs/1909.04879)

Authors:[Michiki Kurosawa](https://arxiv.org/search/cs?searchtype=author&query=Kurosawa%2C+M), [Mamoru Komachi](https://arxiv.org/search/cs?searchtype=author&query=Komachi%2C+M)

*(Submitted on 11 Sep 2019)*

> Neural Machine Translation (NMT) can be used to generate fluent output. As such, language models have been investigated for incorporation with NMT. In prior investigations, two models have been used: a translation model and a language model. The translation model's predictions are weighted by the language model with a hand-crafted ratio in advance. However, these approaches fail to adopt the language model weighting with regard to the translation history. In another line of approach, language model prediction is incorporated into the translation model by jointly considering source and target information. However, this line of approach is limited because it largely ignores the adequacy of the translation output.
> Accordingly, this work employs two mechanisms, the translation model and the language model, with an attentive architecture to the language model as an auxiliary element of the translation model. Compared with previous work in English--Japanese machine translation using a language model, the experimental results obtained with the proposed Dynamic Fusion mechanism improve BLEU and Rank-based Intuitive Bilingual Evaluation Scores (RIBES) scores. Additionally, in the analyses of the attention and predictivity of the language model, the Dynamic Fusion mechanism allows predictive language modeling that conforms to the appropriate grammatical structure.

| Comments: | 13 pages; PACLING 2019                               |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.04879 [cs.CL]**                         |
|           | (or **arXiv:1909.04879v1 [cs.CL]** for this version) |





<h2 id="2019-09-12-4">4. Getting Gender Right in Neural Machine Translation</h2> 
Title: [Getting Gender Right in Neural Machine Translation](https://arxiv.org/abs/1909.05088)

Authors:[Eva Vanmassenhove](https://arxiv.org/search/cs?searchtype=author&query=Vanmassenhove%2C+E), [Christian Hardmeier](https://arxiv.org/search/cs?searchtype=author&query=Hardmeier%2C+C), [Andy Way](https://arxiv.org/search/cs?searchtype=author&query=Way%2C+A)

*(Submitted on 11 Sep 2019)*

> Speakers of different languages must attend to and encode strikingly different aspects of the world in order to use their language correctly (Sapir, 1921; Slobin, 1996). One such difference is related to the way gender is expressed in a language. Saying "I am happy" in English, does not encode any additional knowledge of the speaker that uttered the sentence. However, many other languages do have grammatical gender systems and so such knowledge would be encoded. In order to correctly translate such a sentence into, say, French, the inherent gender information needs to be retained/recovered. The same sentence would become either "Je suis heureux", for a male speaker or "Je suis heureuse" for a female one. Apart from morphological agreement, demographic factors (gender, age, etc.) also influence our use of language in terms of word choices or even on the level of syntactic constructions (Tannen, 1991; Pennebaker et al., 2003). We integrate gender information into NMT systems. Our contribution is two-fold: (1) the compilation of large datasets with speaker information for 20 language pairs, and (2) a simple set of experiments that incorporate gender information into NMT for multiple language pairs. Our experiments show that adding a gender feature to an NMT system significantly improves the translation quality for some language pairs.

| Comments:          | Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), October-November, 2018. Brussels, Belgium, pages 3003-3008, URL: [this https URL](https://www.aclweb.org/anthology/D18-1334), DOI: [10.18653/v1/D18-1334](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.18653%2Fv1%2FD18-1334&v=6f889533) |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**                         |
| Journal reference: | Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing |
| DOI:               | [10.18653/v1/D18-1334](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.18653%2Fv1%2FD18-1334&v=6f889533) |
| Cite as:           | **arXiv:1909.05088 [cs.CL]**                                 |
|                    | (or **arXiv:1909.05088v1 [cs.CL]** for this version)         |







# 2019-09-10

[Return to Index](#Index)



<h2 id="2019-09-10-1">1. Improving Neural Machine Translation with Parent-Scaled Self-Attention</h2> 
Title: [Improving Neural Machine Translation with Parent-Scaled Self-Attention](https://arxiv.org/abs/1909.03149)

Authors:[Emanuele Bugliarello](https://arxiv.org/search/cs?searchtype=author&query=Bugliarello%2C+E), [Naoaki Okazaki](https://arxiv.org/search/cs?searchtype=author&query=Okazaki%2C+N)

*(Submitted on 6 Sep 2019)*

> Most neural machine translation (NMT) models operate on source and target sentences, treating them as sequences of words and neglecting their syntactic structure. Recent studies have shown that embedding the syntax information of a source sentence in recurrent neural networks can improve their translation accuracy, especially for low-resource language pairs. However, state-of-the-art NMT models are based on self-attention networks (e.g., Transformer), in which it is still not clear how to best embed syntactic information. In this work, we explore different approaches to make such models syntactically aware. Moreover, we propose a novel method to incorporate syntactic information in the self-attention mechanism of the Transformer encoder by introducing attention heads that can attend to the dependency parent of each token. The proposed model is simple yet effective, requiring no additional parameter and improving the translation quality of the Transformer model especially for long sentences and low-resource scenarios. We show the efficacy of the proposed approach on NC11 English-German, WMT16 and WMT17 English-German, WMT18 English-Turkish, and WAT English-Japanese translation tasks.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1909.03149 [cs.CL]**                         |
|           | (or **arXiv:1909.03149v1 [cs.CL]** for this version) |





<h2 id="2019-09-10-2">2. LAMAL: LAnguage Modeling Is All You Need for Lifelong Language Learning</h2> 
Title: [LAMAL: LAnguage Modeling Is All You Need for Lifelong Language Learning](https://arxiv.org/abs/1909.03329)

Authors: [Fan-Keng Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+F), [Cheng-Hao Ho](https://arxiv.org/search/cs?searchtype=author&query=Ho%2C+C), [Hung-Yi Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+H)

*(Submitted on 7 Sep 2019)*

> Most research on lifelong learning (LLL) applies to images or games, but not language. Here, we introduce LAMAL, a simple yet effective method for LLL based on language modeling. LAMAL replays pseudo samples of previous tasks while requiring no extra memory or model capacity. To be specific, LAMAL is a language model learning to solve the task and generate training samples at the same time. At the beginning of training a new task, the model generates some pseudo samples of previous tasks to train alongside the data of the new task. The results show that LAMAL prevents catastrophic forgetting without any sign of intransigence and can solve up to five very different language tasks sequentially with only one model. Overall, LAMAL outperforms previous methods by a considerable margin and is only 2-3\% worse than multitasking which is usually considered as the upper bound of LLL. Our source code is available at [this https URL](https://github.com/xxx).

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1909.03329 [cs.CL]**                                 |
|           | (or **arXiv:1909.03329v1 [cs.CL]** for this version)         |





<h2 id="2019-09-10-3">3. Neural Machine Translation with Byte-Level Subwords</h2> 
Title: [Neural Machine Translation with Byte-Level Subwords](https://arxiv.org/abs/1909.03341)

Authors:[Changhan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Kyunghyun Cho](https://arxiv.org/search/cs?searchtype=author&query=Cho%2C+K), [Jiatao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+J)

*(Submitted on 7 Sep 2019)*

> Almost all existing machine translation models are built on top of character-based vocabularies: characters, subwords or words. Rare characters from noisy text or character-rich languages such as Japanese and Chinese however can unnecessarily take up vocabulary slots and limit its compactness. Representing text at the level of bytes and using the 256 byte set as vocabulary is a potential solution to this issue. High computational cost has however prevented it from being widely deployed or used in practice. In this paper, we investigate byte-level subwords, specifically byte-level BPE (BBPE), which is compacter than character vocabulary and has no out-of-vocabulary tokens, but is more efficient than using pure bytes only is. We claim that contextualizing BBPE embeddings is necessary, which can be implemented by a convolutional or recurrent layer. Our experiments show that BBPE has comparable performance to BPE while its size is only 1/8 of that for BPE. In the multilingual setting, BBPE maximizes vocabulary sharing across many languages and achieves better translation quality. Moreover, we show that BBPE enables transferring models between languages with non-overlapping character sets.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1909.03341 [cs.CL]**                         |
|           | (or **arXiv:1909.03341v1 [cs.CL]** for this version) |





<h2 id="2019-09-10-4">4. Combining SMT and NMT Back-Translated Data for Efficient NMT</h2> 
Title: [Combining SMT and NMT Back-Translated Data for Efficient NMT](https://arxiv.org/abs/1909.03750)

Authors:[Alberto Poncelas](https://arxiv.org/search/cs?searchtype=author&query=Poncelas%2C+A), [Maja Popovic](https://arxiv.org/search/cs?searchtype=author&query=Popovic%2C+M), [Dimitar Shterionov](https://arxiv.org/search/cs?searchtype=author&query=Shterionov%2C+D), [Gideon Maillette de Buy Wenniger](https://arxiv.org/search/cs?searchtype=author&query=de+Buy+Wenniger%2C+G+M), [Andy Way](https://arxiv.org/search/cs?searchtype=author&query=Way%2C+A)

*(Submitted on 9 Sep 2019)*

> Neural Machine Translation (NMT) models achieve their best performance when large sets of parallel data are used for training. Consequently, techniques for augmenting the training set have become popular recently. One of these methods is back-translation (Sennrich et al., 2016), which consists on generating synthetic sentences by translating a set of monolingual, target-language sentences using a Machine Translation (MT) model.
> Generally, NMT models are used for back-translation. In this work, we analyze the performance of models when the training data is extended with synthetic data using different MT approaches. In particular we investigate back-translated data generated not only by NMT but also by Statistical Machine Translation (SMT) models and combinations of both. The results reveal that the models achieve the best performances when the training set is augmented with back-translated data created by merging different MT approaches.

| Subjects:          | **Computation and Language (cs.CL)**                         |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | Proceedings of Recent Advances in Natural Language Processing (RANLP 2019). pages 922--931 |
| Cite as:           | **arXiv:1909.03750 [cs.CL]**                                 |
|                    | (or **arXiv:1909.03750v1 [cs.CL]** for this version)         |







# 2019-09-09

[Return to Index](#Index)



<h2 id="2019-09-09-1">1. Don't Forget the Long Tail! A Comprehensive Analysis of Morphological Generalization in Bilingual Lexicon Induction</h2> 
Title: [Don't Forget the Long Tail! A Comprehensive Analysis of Morphological Generalization in Bilingual Lexicon Induction](https://arxiv.org/abs/1909.02855)

Authors: [Paula Czarnowska](https://arxiv.org/search/cs?searchtype=author&query=Czarnowska%2C+P), [Sebastian Ruder](https://arxiv.org/search/cs?searchtype=author&query=Ruder%2C+S), [Edouard Grave](https://arxiv.org/search/cs?searchtype=author&query=Grave%2C+E), [Ryan Cotterell](https://arxiv.org/search/cs?searchtype=author&query=Cotterell%2C+R), [Ann Copestake](https://arxiv.org/search/cs?searchtype=author&query=Copestake%2C+A)

*(Submitted on 6 Sep 2019)*

> Human translators routinely have to translate rare inflections of words - due to the Zipfian distribution of words in a language. When translating from Spanish, a good translator would have no problem identifying the proper translation of a statistically rare inflection such as habláramos. Note the lexeme itself, hablar, is relatively common. In this work, we investigate whether state-of-the-art bilingual lexicon inducers are capable of learning this kind of generalization. We introduce 40 morphologically complete dictionaries in 10 languages and evaluate three of the state-of-the-art models on the task of translation of less frequent morphological forms. We demonstrate that the performance of state-of-the-art models drops considerably when evaluated on infrequent morphological inflections and then show that adding a simple morphological constraint at training time improves the performance, proving that the bilingual lexicon inducers can benefit from better encoding of morphology.

| Comments: | EMNLP 2019                                           |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.02855 [cs.CL]**                         |
|           | (or **arXiv:1909.02855v1 [cs.CL]** for this version) |



# 2019-09-06

[Return to Index](#Index)



<h2 id="2019-09-06-1">1. Jointly Learning to Align and Translate with Transformer Models</h2> 
Title: [Jointly Learning to Align and Translate with Transformer Models](https://arxiv.org/abs/1909.02074)

Authors: [Sarthak Garg](https://arxiv.org/search/cs?searchtype=author&query=Garg%2C+S), [Stephan Peitz](https://arxiv.org/search/cs?searchtype=author&query=Peitz%2C+S), [Udhyakumar Nallasamy](https://arxiv.org/search/cs?searchtype=author&query=Nallasamy%2C+U), [Matthias Paulik](https://arxiv.org/search/cs?searchtype=author&query=Paulik%2C+M)

*(Submitted on 4 Sep 2019)*

> The state of the art in machine translation (MT) is governed by neural approaches, which typically provide superior translation accuracy over statistical approaches. However, on the closely related task of word alignment, traditional statistical word alignment models often remain the go-to solution. In this paper, we present an approach to train a Transformer model to produce both accurate translations and alignments. We extract discrete alignments from the attention probabilities learnt during regular neural machine translation model training and leverage them in a multi-task framework to optimize towards translation and alignment objectives. We demonstrate that our approach produces competitive results compared to GIZA++ trained IBM alignment models without sacrificing translation accuracy and outperforms previous attempts on Transformer model based word alignment. Finally, by incorporating IBM model alignments into our multi-task training, we report significantly better alignment accuracies compared to GIZA++ on three publicly available data sets.

| Comments: | 10 pages, 2 figures. To appear at EMNLP 2019         |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.02074 [cs.CL]**                         |
|           | (or **arXiv:1909.02074v1 [cs.CL]** for this version) |





<h2 id="2019-09-06-2">2. Investigating Multilingual NMT Representations at Scale</h2> 
Title: [Investigating Multilingual NMT Representations at Scale](https://arxiv.org/abs/1909.02197)

Authors: [Sneha Reddy Kudugunta](https://arxiv.org/search/cs?searchtype=author&query=Kudugunta%2C+S+R), [Ankur Bapna](https://arxiv.org/search/cs?searchtype=author&query=Bapna%2C+A), [Isaac Caswell](https://arxiv.org/search/cs?searchtype=author&query=Caswell%2C+I), [Naveen Arivazhagan](https://arxiv.org/search/cs?searchtype=author&query=Arivazhagan%2C+N), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O)

*(Submitted on 5 Sep 2019)*

> Multilingual Neural Machine Translation (NMT) models have yielded large empirical success in transfer learning settings. However, these black-box representations are poorly understood, and their mode of transfer remains elusive. In this work, we attempt to understand massively multilingual NMT representations (with 103 languages) using Singular Value Canonical Correlation Analysis (SVCCA), a representation similarity framework that allows us to compare representations across different languages, layers and models. Our analysis validates several empirical results and long-standing intuitions, and unveils new observations regarding how representations evolve in a multilingual translation model. We draw three major conclusions from our analysis, with implications on cross-lingual transfer learning: (i) Encoder representations of different languages cluster based on linguistic similarity, (ii) Representations of a source language learned by the encoder are dependent on the target language, and vice-versa, and (iii) Representations of high resource and/or linguistically similar languages are more robust when fine-tuning on an arbitrary language pair, which is critical to determining how much cross-lingual transfer can be expected in a zero or few-shot setting. We further connect our findings with existing empirical observations in multilingual NMT and transfer learning.

| Comments: | Paper at EMNLP 2019                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1909.02197 [cs.CL]**                                 |
|           | (or **arXiv:1909.02197v1 [cs.CL]** for this version)         |





<h2 id="2019-09-06-3">3. Multi-Granularity Self-Attention for Neural Machine Translation</h2> 
Title: [Multi-Granularity Self-Attention for Neural Machine Translation](https://arxiv.org/abs/1909.02222)

Authors: [Jie Hao](https://arxiv.org/search/cs?searchtype=author&query=Hao%2C+J), [Xing Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Shuming Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+S), [Jinfeng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+J), [Zhaopeng Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+Z)

*(Submitted on 5 Sep 2019)*

> Current state-of-the-art neural machine translation (NMT) uses a deep multi-head self-attention network with no explicit phrase information. However, prior work on statistical machine translation has shown that extending the basic translation unit from words to phrases has produced substantial improvements, suggesting the possibility of improving NMT performance from explicit modeling of phrases. In this work, we present multi-granularity self-attention (Mg-Sa): a neural network that combines multi-head self-attention and phrase modeling. Specifically, we train several attention heads to attend to phrases in either n-gram or syntactic formalism. Moreover, we exploit interactions among phrases to enhance the strength of structure modeling - a commonly-cited weakness of self-attention. Experimental results on WMT14 English-to-German and NIST Chinese-to-English translation tasks show the proposed approach consistently improves performance. Targeted linguistic analysis reveals that Mg-Sa indeed captures useful phrase information at various levels of granularities.

| Comments: | EMNLP 2019                                           |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.02222 [cs.CL]**                         |
|           | (or **arXiv:1909.02222v1 [cs.CL]** for this version) |





<h2 id="2019-09-06-4">4. Source Dependency-Aware Transformer with Supervised Self-Attention</h2> 
Title: [Source Dependency-Aware Transformer with Supervised Self-Attention](https://arxiv.org/abs/1909.02273)

Authors: [Chengyi Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Shuangzhi Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+S), [Shujie Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+S)

*(Submitted on 5 Sep 2019)*

> Recently, Transformer has achieved the state-of-the-art performance on many machine translation tasks. However, without syntax knowledge explicitly considered in the encoder, incorrect context information that violates the syntax structure may be integrated into source hidden states, leading to erroneous translations. In this paper, we propose a novel method to incorporate source dependencies into the Transformer. Specifically, we adopt the source dependency tree and define two matrices to represent the dependency relations. Based on the matrices, two heads in the multi-head self-attention module are trained in a supervised manner and two extra cross entropy losses are introduced into the training objective function. Under this training objective, the model is trained to learn the source dependency relations directly. Without requiring pre-parsed input during inference, our model can generate better translations with the dependency-aware context information. Experiments on bi-directional Chinese-to-English, English-to-Japanese and English-to-German translation tasks show that our proposed method can significantly improve the Transformer baseline.

| Comments: | 6 pages                                              |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.02273 [cs.CL]**                         |
|           | (or **arXiv:1909.02273v1 [cs.CL]** for this version) |





<h2 id="2019-09-06-5">5. Accelerating Transformer Decoding via a Hybrid of Self-attention and Recurrent Neural Network</h2> 
Title: [Accelerating Transformer Decoding via a Hybrid of Self-attention and Recurrent Neural Network](https://arxiv.org/abs/1909.02279)

Authors: [Chengyi Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Shuangzhi Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+S), [Shujie Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+S)

*(Submitted on 5 Sep 2019)*

> Due to the highly parallelizable architecture, Transformer is faster to train than RNN-based models and popularly used in machine translation tasks. However, at inference time, each output word requires all the hidden states of the previously generated words, which limits the parallelization capability, and makes it much slower than RNN-based ones. In this paper, we systematically analyze the time cost of different components of both the Transformer and RNN-based model. Based on it, we propose a hybrid network of self-attention and RNN structures, in which, the highly parallelizable self-attention is utilized as the encoder, and the simpler RNN structure is used as the decoder. Our hybrid network can decode 4-times faster than the Transformer. In addition, with the help of knowledge distillation, our hybrid network achieves comparable translation quality to the original Transformer.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1909.02279 [cs.CL]**                         |
|           | (or **arXiv:1909.02279v1 [cs.CL]** for this version) |





<h2 id="2019-09-06-6">6. FlowSeq: Non-Autoregressive Conditional Sequence Generation with Generative Flow</h2> 
Title: [FlowSeq: Non-Autoregressive Conditional Sequence Generation with Generative Flow](https://arxiv.org/abs/1909.02480)

Authors: [Xuezhe Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+X), [Chunting Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+C), [Xian Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+X), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G), [Eduard Hovy](https://arxiv.org/search/cs?searchtype=author&query=Hovy%2C+E)

*(Submitted on 5 Sep 2019)*

> Most sequence-to-sequence (seq2seq) models are autoregressive; they generate each token by conditioning on previously generated tokens. In contrast, non-autoregressive seq2seq models generate all tokens in one pass, which leads to increased efficiency through parallel processing on hardware such as GPUs. However, directly modeling the joint distribution of all tokens simultaneously is challenging, and even with increasingly complex model structures accuracy lags significantly behind autoregressive models. In this paper, we propose a simple, efficient, and effective model for non-autoregressive sequence generation using latent variable models. Specifically, we turn to generative flow, an elegant technique to model complex distributions using neural networks, and design several layers of flow tailored for modeling the conditional density of sequential latent variables. We evaluate this model on three neural machine translation (NMT) benchmark datasets, achieving comparable performance with state-of-the-art non-autoregressive NMT models and almost constant decoding time w.r.t the sequence length.

| Comments: | Accepted by EMNLP 2019 (Long Paper)                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1909.02480 [cs.CL]**                                 |
|           | (or **arXiv:1909.02480v1 [cs.CL]** for this version)         |



# 2019-09-05

[Return to Index](#Index)



<h2 id="2019-09-05-1">1. The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives</h2> 
Title: [The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives](https://arxiv.org/abs/1909.01380)

Authors: [Elena Voita](https://arxiv.org/search/cs?searchtype=author&query=Voita%2C+E), [Rico Sennrich](https://arxiv.org/search/cs?searchtype=author&query=Sennrich%2C+R), [Ivan Titov](https://arxiv.org/search/cs?searchtype=author&query=Titov%2C+I)

*(Submitted on 3 Sep 2019)*

> We seek to understand how the representations of individual tokens and the structure of the learned feature space evolve between layers in deep neural networks under different learning objectives. We focus on the Transformers for our analysis as they have been shown effective on various tasks, including machine translation (MT), standard left-to-right language models (LM) and masked language modeling (MLM). Previous work used black-box probing tasks to show that the representations learned by the Transformer differ significantly depending on the objective. In this work, we use canonical correlation analysis and mutual information estimators to study how information flows across Transformer layers and how this process depends on the choice of learning objective. For example, as you go from bottom to top layers, information about the past in left-to-right language models gets vanished and predictions about the future get formed. In contrast, for MLM, representations initially acquire information about the context around the token, partially forgetting the token identity and producing a more generalized token representation. The token identity then gets recreated at the top MLM layers.

| Comments: | EMNLP 2019 (camera-ready)                            |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.01380 [cs.CL]**                         |
|           | (or **arXiv:1909.01380v1 [cs.CL]** for this version) |





<h2 id="2019-09-05-2">2. Context-Aware Monolingual Repair for Neural Machine Translation</h2> 
Title: [Context-Aware Monolingual Repair for Neural Machine Translation](https://arxiv.org/abs/1909.01383)

Authors: [Elena Voita](https://arxiv.org/search/cs?searchtype=author&query=Voita%2C+E), [Rico Sennrich](https://arxiv.org/search/cs?searchtype=author&query=Sennrich%2C+R), [Ivan Titov](https://arxiv.org/search/cs?searchtype=author&query=Titov%2C+I)

*(Submitted on 3 Sep 2019)*

> Modern sentence-level NMT systems often produce plausible translations of isolated sentences. However, when put in context, these translations may end up being inconsistent with each other. We propose a monolingual DocRepair model to correct inconsistencies between sentence-level translations. DocRepair performs automatic post-editing on a sequence of sentence-level translations, refining translations of sentences in context of each other. For training, the DocRepair model requires only monolingual document-level data in the target language. It is trained as a monolingual sequence-to-sequence model that maps inconsistent groups of sentences into consistent ones. The consistent groups come from the original training data; the inconsistent groups are obtained by sampling round-trip translations for each isolated sentence. We show that this approach successfully imitates inconsistencies we aim to fix: using contrastive evaluation, we show large improvements in the translation of several contextual phenomena in an English-Russian translation task, as well as improvements in the BLEU score. We also conduct a human evaluation and show a strong preference of the annotators to corrected translations over the baseline ones. Moreover, we analyze which discourse phenomena are hard to capture using monolingual data only.

| Comments: | EMNLP 2019 (camera-ready)                            |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.01383 [cs.CL]**                         |
|           | (or **arXiv:1909.01383v1 [cs.CL]** for this version) |





<h2 id="2019-09-05-3">3. Simpler and Faster Learning of Adaptive Policies for Simultaneous Translation</h2> 
Title: [Simpler and Faster Learning of Adaptive Policies for Simultaneous Translation](https://arxiv.org/abs/1909.01559)

Authors: [Baigong Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+B), [Renjie Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+R), [Mingbo Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+M), [Liang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+L)

*(Submitted on 4 Sep 2019)*

> Simultaneous translation is widely useful but remains challenging. Previous work falls into two main categories: (a) fixed-latency policies such as Ma et al. (2019) and (b) adaptive policies such as Gu et al. (2017). The former are simple and effective, but have to aggressively predict future content due to diverging source-target word order; the latter do not anticipate, but suffer from unstable and inefficient training. To combine the merits of both approaches, we propose a simple supervised-learning framework to learn an adaptive policy from oracle READ/WRITE sequences generated from parallel text. At each step, such an oracle sequence chooses to WRITE the next target word if the available source sentence context provides enough information to do so, otherwise READ the next source word. Experiments on German<->English show that our method, without retraining the underlying NMT model, can learn flexible policies with better BLEU scores and similar latencies compared to previous work.

| Comments: | EMNLP 2019                                           |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.01559 [cs.CL]**                         |
|           | (or **arXiv:1909.01559v1 [cs.CL]** for this version) |





<h2 id="2019-09-05-4">4. Do We Really Need Fully Unsupervised Cross-Lingual Embeddings?</h2> 
Title: [Do We Really Need Fully Unsupervised Cross-Lingual Embeddings?](https://arxiv.org/abs/1909.01638)

Authors: [Ivan Vulić](https://arxiv.org/search/cs?searchtype=author&query=Vulić%2C+I), [Goran Glavaš](https://arxiv.org/search/cs?searchtype=author&query=Glavaš%2C+G), [Roi Reichart](https://arxiv.org/search/cs?searchtype=author&query=Reichart%2C+R), [Anna Korhonen](https://arxiv.org/search/cs?searchtype=author&query=Korhonen%2C+A)

*(Submitted on 4 Sep 2019)*

> Recent efforts in cross-lingual word embedding (CLWE) learning have predominantly focused on fully unsupervised approaches that project monolingual embeddings into a shared cross-lingual space without any cross-lingual signal. The lack of any supervision makes such approaches conceptually attractive. Yet, their only core difference from (weakly) supervised projection-based CLWE methods is in the way they obtain a seed dictionary used to initialize an iterative self-learning procedure. The fully unsupervised methods have arguably become more robust, and their primary use case is CLWE induction for pairs of resource-poor and distant languages. In this paper, we question the ability of even the most robust unsupervised CLWE approaches to induce meaningful CLWEs in these more challenging settings. A series of bilingual lexicon induction (BLI) experiments with 15 diverse languages (210 language pairs) show that fully unsupervised CLWE methods still fail for a large number of language pairs (e.g., they yield zero BLI performance for 87/210 pairs). Even when they succeed, they never surpass the performance of weakly supervised methods (seeded with 500-1,000 translation pairs) using the same self-learning procedure in any BLI setup, and the gaps are often substantial. These findings call for revisiting the main motivations behind fully unsupervised CLWE methods.

| Comments: | EMNLP 2019 (Long paper)                              |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.01638 [cs.CL]**                         |
|           | (or **arXiv:1909.01638v1 [cs.CL]** for this version) |





<h2 id="2019-09-05-5">5. SAO WMT19 Test Suite: Machine Translation of Audit Reports</h2> 
Title: [SAO WMT19 Test Suite: Machine Translation of Audit Reports](https://arxiv.org/abs/1909.01701)

Authors: [Tereza Vojtěchová](https://arxiv.org/search/cs?searchtype=author&query=Vojtěchová%2C+T), [Michal Novák](https://arxiv.org/search/cs?searchtype=author&query=Novák%2C+M), [Miloš Klouček](https://arxiv.org/search/cs?searchtype=author&query=Klouček%2C+M), [Ondřej Bojar](https://arxiv.org/search/cs?searchtype=author&query=Bojar%2C+O)

*(Submitted on 4 Sep 2019)*

> This paper describes a machine translation test set of documents from the auditing domain and its use as one of the "test suites" in the WMT19 News Translation Task for translation directions involving Czech, English and German. 
> Our evaluation suggests that current MT systems optimized for the general news domain can perform quite well even in the particular domain of audit reports. The detailed manual evaluation however indicates that deep factual knowledge of the domain is necessary. For the naked eye of a non-expert, translations by many systems seem almost perfect and automatic MT evaluation with one reference is practically useless for considering these details. 
> Furthermore, we show on a sample document from the domain of agreements that even the best systems completely fail in preserving the semantics of the agreement, namely the identity of the parties.

| Comments:          | WMT19 ([this http URL](http://www.statmt.org/wmt19/))        |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**                         |
| Journal reference: | Vojt\v{e}chov\'a et al. (2019): SAO WMT19 Test Suite: Machine Translation of Audit Reports. In: Fourth Conference on Machine Translation - Proceedings of the Conference, pp. 680-692, ACL, ISBN 978-1-950737-27-7 |
| Cite as:           | **arXiv:1909.01701 [cs.CL]**                                 |
|                    | (or **arXiv:1909.01701v1 [cs.CL]** for this version)         |





# 2019-09-04

[Return to Index](#Index)



<h2 id="2019-09-04-1">1. Hybrid Data-Model Parallel Training for Sequence-to-Sequence Recurrent Neural Network Machine Translation</h2> 
Title: [Hybrid Data-Model Parallel Training for Sequence-to-Sequence Recurrent Neural Network Machine Translation](https://arxiv.org/abs/1909.00562)

Authors: [Junya Ono](https://arxiv.org/search/cs?searchtype=author&query=Ono%2C+J), [Masao Utiyama](https://arxiv.org/search/cs?searchtype=author&query=Utiyama%2C+M), [Eiichiro Sumita](https://arxiv.org/search/cs?searchtype=author&query=Sumita%2C+E)

*(Submitted on 2 Sep 2019)*

> Reduction of training time is an important issue in many tasks like patent translation involving neural networks. Data parallelism and model parallelism are two common approaches for reducing training time using multiple graphics processing units (GPUs) on one machine. In this paper, we propose a hybrid data-model parallel approach for sequence-to-sequence (Seq2Seq) recurrent neural network (RNN) machine translation. We apply a model parallel approach to the RNN encoder-decoder part of the Seq2Seq model and a data parallel approach to the attention-softmax part of the model. We achieved a speed-up of 4.13 to 4.20 times when using 4 GPUs compared with the training speed when using 1 GPU without affecting machine translation accuracy as measured in terms of BLEU scores.

| Comments: | 9 pages, 4 figures, 5 tables                                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Distributed, Parallel, and Cluster Computing (cs.DC)**; Computation and Language (cs.CL); Machine Learning (cs.LG); Neural and Evolutionary Computing (cs.NE) |
| Cite as:  | **arXiv:1909.00562 [cs.DC]**                                 |
|           | (or **arXiv:1909.00562v1 [cs.DC]** for this version)         |





<h2 id="2019-09-04-2">2. Handling Syntactic Divergence in Low-resource Machine Translation</h2> 
Title: [Handling Syntactic Divergence in Low-resource Machine Translation](https://arxiv.org/abs/1909.00040)

Authors: [Chunting Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+C), [Xuezhe Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+X), [Junjie Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+J), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G)

*(Submitted on 30 Aug 2019)*

> Despite impressive empirical successes of neural machine translation (NMT) on standard benchmarks, limited parallel data impedes the application of NMT models to many language pairs. Data augmentation methods such as back-translation make it possible to use monolingual data to help alleviate these issues, but back-translation itself fails in extreme low-resource scenarios, especially for syntactically divergent languages. In this paper, we propose a simple yet effective solution, whereby target-language sentences are re-ordered to match the order of the source and used as an additional source of training-time supervision. Experiments with simulated low-resource Japanese-to-English, and real low-resource Uyghur-to-English scenarios find significant improvements over other semi-supervised alternatives.

| Comments: | Accepted by EMNLP 2019 (short paper)                 |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.00040 [cs.CL]**                         |
|           | (or **arXiv:1909.00040v1 [cs.CL]** for this version) |







<h2 id="2019-09-04-3">3. Evaluating Pronominal Anaphora in Machine Translation: An Evaluation Measure and a Test Suite</h2> 
Title: [Evaluating Pronominal Anaphora in Machine Translation: An Evaluation Measure and a Test Suite](https://arxiv.org/abs/1909.00131)

Authors: [Prathyusha Jwalapuram](https://arxiv.org/search/cs?searchtype=author&query=Jwalapuram%2C+P), [Shafiq Joty](https://arxiv.org/search/cs?searchtype=author&query=Joty%2C+S), [Irina Temnikova](https://arxiv.org/search/cs?searchtype=author&query=Temnikova%2C+I), [Preslav Nakov](https://arxiv.org/search/cs?searchtype=author&query=Nakov%2C+P)

*(Submitted on 31 Aug 2019)*

> The ongoing neural revolution in machine translation has made it easier to model larger contexts beyond the sentence-level, which can potentially help resolve some discourse-level ambiguities such as pronominal anaphora, thus enabling better translations. Unfortunately, even when the resulting improvements are seen as substantial by humans, they remain virtually unnoticed by traditional automatic evaluation measures like BLEU, as only a few words end up being affected. Thus, specialized evaluation measures are needed. With this aim in mind, we contribute an extensive, targeted dataset that can be used as a test suite for pronoun translation, covering multiple source languages and different pronoun errors drawn from real system translations, for English. We further propose an evaluation measure to differentiate good and bad pronoun translations. We also conduct a user study to report correlations with human judgments.

| Comments: | Accepted at EMNLP 2019                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1909.00131 [cs.CL]**                                 |
|           | (or **arXiv:1909.00131v1 [cs.CL]** for this version)         |







<h2 id="2019-09-04-4">4. Improving Back-Translation with Uncertainty-based Confidence Estimation</h2> 
Title: [Improving Back-Translation with Uncertainty-based Confidence Estimation](https://arxiv.org/abs/1909.00157)

Authors: [Shuo Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Yang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Chao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Huanbo Luan](https://arxiv.org/search/cs?searchtype=author&query=Luan%2C+H), [Maosong Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+M)

*(Submitted on 31 Aug 2019)*

> While back-translation is simple and effective in exploiting abundant monolingual corpora to improve low-resource neural machine translation (NMT), the synthetic bilingual corpora generated by NMT models trained on limited authentic bilingual data are inevitably noisy. In this work, we propose to quantify the confidence of NMT model predictions based on model uncertainty. With word- and sentence-level confidence measures based on uncertainty, it is possible for back-translation to better cope with noise in synthetic bilingual corpora. Experiments on Chinese-English and English-German translation tasks show that uncertainty-based confidence estimation significantly improves the performance of back-translation.

| Comments: | EMNLP 2019                                           |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.00157 [cs.CL]**                         |
|           | (or **arXiv:1909.00157v1 [cs.CL]** for this version) |







<h2 id="2019-09-04-5">5. Explicit Cross-lingual Pre-training for Unsupervised Machine Translation</h2> 
Title: [Explicit Cross-lingual Pre-training for Unsupervised Machine Translation](https://arxiv.org/abs/1909.00180)

Authors: [Shuo Ren](https://arxiv.org/search/cs?searchtype=author&query=Ren%2C+S), [Yu Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+Y), [Shujie Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+S), [Ming Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+M), [Shuai Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+S)

*(Submitted on 31 Aug 2019)*

> Pre-training has proven to be effective in unsupervised machine translation due to its ability to model deep context information in cross-lingual scenarios. However, the cross-lingual information obtained from shared BPE spaces is inexplicit and limited. In this paper, we propose a novel cross-lingual pre-training method for unsupervised machine translation by incorporating explicit cross-lingual training signals. Specifically, we first calculate cross-lingual n-gram embeddings and infer an n-gram translation table from them. With those n-gram translation pairs, we propose a new pre-training model called Cross-lingual Masked Language Model (CMLM), which randomly chooses source n-grams in the input text stream and predicts their translation candidates at each time step. Experiments show that our method can incorporate beneficial cross-lingual information into pre-trained models. Taking pre-trained CMLM models as the encoder and decoder, we significantly improve the performance of unsupervised machine translation.

| Comments: | Accepted to EMNLP2019; 10 pages, 2 figures           |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.00180 [cs.CL]**                         |
|           | (or **arXiv:1909.00180v1 [cs.CL]** for this version) |







<h2 id="2019-09-04-6">6. Towards Understanding Neural Machine Translation with Word Importance</h2> 
Title: [Towards Understanding Neural Machine Translation with Word Importance](https://arxiv.org/abs/1909.00326)

Authors: [Shilin He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+S), [Zhaopeng Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+Z), [Xing Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Longyue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Michael R. Lyu](https://arxiv.org/search/cs?searchtype=author&query=Lyu%2C+M+R), [Shuming Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+S)

*(Submitted on 1 Sep 2019)*

> Although neural machine translation (NMT) has advanced the state-of-the-art on various language pairs, the interpretability of NMT remains unsatisfactory. In this work, we propose to address this gap by focusing on understanding the input-output behavior of NMT models. Specifically, we measure the word importance by attributing the NMT output to every input word through a gradient-based method. We validate the approach on a couple of perturbation operations, language pairs, and model architectures, demonstrating its superiority on identifying input words with higher influence on translation performance. Encouragingly, the calculated importance can serve as indicators of input words that are under-translated by NMT models. Furthermore, our analysis reveals that words of certain syntactic categories have higher importance while the categories vary across language pairs, which can inspire better design principles of NMT architectures for multi-lingual translation.

| Comments: | EMNLP 2019                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1909.00326 [cs.CL]**                                 |
|           | (or **arXiv:1909.00326v1 [cs.CL]** for this version)         |







<h2 id="2019-09-04-7">7. One Model to Learn Both: Zero Pronoun Prediction and Translation</h2> 
Title: [One Model to Learn Both: Zero Pronoun Prediction and Translation](https://arxiv.org/abs/1909.00369)

Authors: [Longyue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Zhaopeng Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+Z), [Xing Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Shuming Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+S)

*(Submitted on 1 Sep 2019)*

> Zero pronouns (ZPs) are frequently omitted in pro-drop languages, but should be recalled in non-pro-drop languages. This discourse phenomenon poses a significant challenge for machine translation (MT) when translating texts from pro-drop to non-pro-drop languages. In this paper, we propose a unified and discourse-aware ZP translation approach for neural MT models. Specifically, we jointly learn to predict and translate ZPs in an end-to-end manner, allowing both components to interact with each other. In addition, we employ hierarchical neural networks to exploit discourse-level context, which is beneficial for ZP prediction and thus translation. Experimental results on both Chinese-English and Japanese-English data show that our approach significantly and accumulatively improves both translation performance and ZP prediction accuracy over not only baseline but also previous works using external ZP prediction models. Extensive analyses confirm that the performance improvement comes from the alleviation of different kinds of errors especially caused by subjective ZPs.

| Comments: | EMNLP 2019                                           |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.00369 [cs.CL]**                         |
|           | (or **arXiv:1909.00369v1 [cs.CL]** for this version) |







<h2 id="2019-09-04-8">8. Evaluating the Cross-Lingual Effectiveness of Massively Multilingual Neural Machine Translation</h2> 
Title: [Evaluating the Cross-Lingual Effectiveness of Massively Multilingual Neural Machine Translation](https://arxiv.org/abs/1909.00437)

Authors: [Aditya Siddhant](https://arxiv.org/search/cs?searchtype=author&query=Siddhant%2C+A), [Melvin Johnson](https://arxiv.org/search/cs?searchtype=author&query=Johnson%2C+M), [Henry Tsai](https://arxiv.org/search/cs?searchtype=author&query=Tsai%2C+H), [Naveen Arivazhagan](https://arxiv.org/search/cs?searchtype=author&query=Arivazhagan%2C+N), [Jason Riesa](https://arxiv.org/search/cs?searchtype=author&query=Riesa%2C+J), [Ankur Bapna](https://arxiv.org/search/cs?searchtype=author&query=Bapna%2C+A), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O), [Karthik Raman](https://arxiv.org/search/cs?searchtype=author&query=Raman%2C+K)

*(Submitted on 1 Sep 2019)*

> The recently proposed massively multilingual neural machine translation (NMT) system has been shown to be capable of translating over 100 languages to and from English within a single model. Its improved translation performance on low resource languages hints at potential cross-lingual transfer capability for downstream tasks. In this paper, we evaluate the cross-lingual effectiveness of representations from the encoder of a massively multilingual NMT model on 5 downstream classification and sequence labeling tasks covering a diverse set of over 50 languages. We compare against a strong baseline, multilingual BERT (mBERT), in different cross-lingual transfer learning scenarios and show gains in zero-shot transfer in 4 out of these 5 tasks.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1909.00437 [cs.CL]**                         |
|           | (or **arXiv:1909.00437v1 [cs.CL]** for this version) |







<h2 id="2019-09-04-9">9. Improving Context-aware Neural Machine Translation with Target-side Context</h2> 
Title: [Improving Context-aware Neural Machine Translation with Target-side Context](https://arxiv.org/abs/1909.00531)

Authors: [Hayahide Yamagishi](https://arxiv.org/search/cs?searchtype=author&query=Yamagishi%2C+H), [Mamoru Komachi](https://arxiv.org/search/cs?searchtype=author&query=Komachi%2C+M)

*(Submitted on 2 Sep 2019)*

> In recent years, several studies on neural machine translation (NMT) have attempted to use document-level context by using a multi-encoder and two attention mechanisms to read the current and previous sentences to incorporate the context of the previous sentences. These studies concluded that the target-side context is less useful than the source-side context. However, we considered that the reason why the target-side context is less useful lies in the architecture used to model these contexts. 
> Therefore, in this study, we investigate how the target-side context can improve context-aware neural machine translation. We propose a weight sharing method wherein NMT saves decoder states and calculates an attention vector using the saved states when translating a current sentence. Our experiments show that the target-side context is also useful if we plug it into NMT as the decoder state when translating a previous sentence.

| Comments: | 12 pages; PACLING 2019                               |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.00531 [cs.CL]**                         |
|           | (or **arXiv:1909.00531v1 [cs.CL]** for this version) |







<h2 id="2019-09-04-10">10. Enhancing Context Modeling with a Query-Guided Capsule Network for Document-level Translation</h2> 
Title: [Enhancing Context Modeling with a Query-Guided Capsule Network for Document-level Translation](https://arxiv.org/abs/1909.00564)

Authors: [Zhengxin Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Jinchao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+J), [Fandong Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+F), [Shuhao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+S), [Yang Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+Y), [Jie Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J)

*(Submitted on 2 Sep 2019)*

> Context modeling is essential to generate coherent and consistent translation for Document-level Neural Machine Translations. The widely used method for document-level translation usually compresses the context information into a representation via hierarchical attention networks. However, this method neither considers the relationship between context words nor distinguishes the roles of context words. To address this problem, we propose a query-guided capsule networks to cluster context information into different perspectives from which the target translation may concern. Experiment results show that our method can significantly outperform strong baselines on multiple data sets of different domains.

| Comments: | 11 pages, 7 figures, 2019 Conference on Empirical Methods in Natural Language Processing |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1909.00564 [cs.CL]**                                 |
|           | (or **arXiv:1909.00564v1 [cs.CL]** for this version)         |







<h2 id="2019-09-04-11">11. Unicoder: A Universal Language Encoder by Pre-training with Multiple Cross-lingual Tasks</h2> 
Title: [Unicoder: A Universal Language Encoder by Pre-training with Multiple Cross-lingual Tasks](https://arxiv.org/abs/1909.00964)

Authors: [Haoyang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+H), [Yaobo Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+Y), [Nan Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan%2C+N), [Ming Gong](https://arxiv.org/search/cs?searchtype=author&query=Gong%2C+M), [Linjun Shou](https://arxiv.org/search/cs?searchtype=author&query=Shou%2C+L), [Daxin Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+D), [Ming Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+M)

*(Submitted on 3 Sep 2019 ([v1](https://arxiv.org/abs/1909.00964v1)), last revised 4 Sep 2019 (this version, v2))*

> We present Unicoder, a universal language encoder that is insensitive to different languages. Given an arbitrary NLP task, a model can be trained with Unicoder using training data in one language and directly applied to inputs of the same task in other languages. Comparing to similar efforts such as Multilingual BERT and XLM, three new cross-lingual pre-training tasks are proposed, including cross-lingual word recovery, cross-lingual paraphrase classification and cross-lingual masked language model. These tasks help Unicoder learn the mappings among different languages from more perspectives. We also find that doing fine-tuning on multiple languages together can bring further improvement. Experiments are performed on two tasks: cross-lingual natural language inference (XNLI) and cross-lingual question answering (XQA), where XLM is our baseline. On XNLI, 1.8% averaged accuracy improvement (on 15 languages) is obtained. On XQA, which is a new cross-lingual dataset built by us, 5.5% averaged accuracy improvement (on French and German) is obtained.

| Comments: | Accepted to EMNLP2019; 10 pages, 2 figures           |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.00964 [cs.CL]**                         |
|           | (or **arXiv:1909.00964v2 [cs.CL]** for this version) |







<h2 id="2019-09-04-12">12. Multi-agent Learning for Neural Machine Translation</h2> 
Title: [Multi-agent Learning for Neural Machine Translation](https://arxiv.org/abs/1909.01101)

Authors: [Tianchi Bi](https://arxiv.org/search/cs?searchtype=author&query=Bi%2C+T), [Hao Xiong](https://arxiv.org/search/cs?searchtype=author&query=Xiong%2C+H), [Zhongjun He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+Z), [Hua Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+H), [Haifeng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+H)

*(Submitted on 3 Sep 2019)*

> Conventional Neural Machine Translation (NMT) models benefit from the training with an additional agent, e.g., dual learning, and bidirectional decoding with one agent decoding from left to right and the other decoding in the opposite direction. In this paper, we extend the training framework to the multi-agent scenario by introducing diverse agents in an interactive updating process. At training time, each agent learns advanced knowledge from others, and they work together to improve translation quality. Experimental results on NIST Chinese-English, IWSLT 2014 German-English, WMT 2014 English-German and large-scale Chinese-English translation tasks indicate that our approach achieves absolute improvements over the strong baseline systems and shows competitive performance on all tasks.

| Comments: | Accepted by EMNLP2019                                |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.01101 [cs.CL]**                         |
|           | (or **arXiv:1909.01101v1 [cs.CL]** for this version) |







<h2 id="2019-09-04-13">13. Bilingual is At Least Monolingual (BALM): A Novel Translation Algorithm that Encodes Monolingual Priors</h2> 
Title: [Bilingual is At Least Monolingual (BALM): A Novel Translation Algorithm that Encodes Monolingual Priors](https://arxiv.org/abs/1909.01146)

Authors: [Jeffrey Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng%2C+J), [Chris Callison-Burch](https://arxiv.org/search/cs?searchtype=author&query=Callison-Burch%2C+C)

*(Submitted on 30 Aug 2019)*

> State-of-the-art machine translation (MT) models do not use knowledge of any single language's structure; this is the equivalent of asking someone to translate from English to German while knowing neither language. BALM is a framework incorporates monolingual priors into an MT pipeline; by casting input and output languages into embedded space using BERT, we can solve machine translation with much simpler models. We find that English-to-German translation on the Multi30k dataset can be solved with a simple feedforward network under the BALM framework with near-SOTA BLEU scores.

| Comments: | 15 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1909.01146 [cs.CL]**                                 |
|           | (or **arXiv:1909.01146v1 [cs.CL]** for this version)         |







# 2019-09-02

[Return to Index](#Index)



<h2 id="2019-09-02-1">1. Latent Part-of-Speech Sequences for Neural Machine Translation</h2> 
Title: [Latent Part-of-Speech Sequences for Neural Machine Translation](https://arxiv.org/abs/1908.11782)

Authors: [Xuewen Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+X), [Yingru Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Dongliang Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+D), [Xin Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Niranjan Balasubramanian](https://arxiv.org/search/cs?searchtype=author&query=Balasubramanian%2C+N)

*(Submitted on 30 Aug 2019)*

> Learning target side syntactic structure has been shown to improve Neural Machine Translation (NMT). However, incorporating syntax through latent variables introduces additional complexity in inference, as the models need to marginalize over the latent syntactic structures. To avoid this, models often resort to greedy search which only allows them to explore a limited portion of the latent space. In this work, we introduce a new latent variable model, LaSyn, that captures the co-dependence between syntax and semantics, while allowing for effective and efficient inference over the latent space. LaSyn decouples direct dependence between successive latent variables, which allows its decoder to exhaustively search through the latent syntactic choices, while keeping decoding speed proportional to the size of the latent variable vocabulary. We implement LaSyn by modifying a transformer-based NMT system and design a neural expectation maximization algorithm that we regularize with part-of-speech information as the latent sequences. Evaluations on four different MT tasks show that incorporating target side syntax with LaSyn improves both translation quality, and also provides an opportunity to improve diversity.

| Comments: | In proceedings of EMNLP 2019                                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Artificial Intelligence (cs.AI)**; Computation and Language (cs.CL) |
| Cite as:  | **arXiv:1908.11782 [cs.AI]**                                 |
|           | (or **arXiv:1908.11782v1 [cs.AI]** for this version)         |





<h2 id="2019-09-02-2">2. Encoders Help You Disambiguate Word Senses in Neural Machine Translation</h2> 
Title: [Encoders Help You Disambiguate Word Senses in Neural Machine Translation](https://arxiv.org/abs/1908.11771)

Authors: [Gongbo Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+G), [Rico Sennrich](https://arxiv.org/search/cs?searchtype=author&query=Sennrich%2C+R), [Joakim Nivre](https://arxiv.org/search/cs?searchtype=author&query=Nivre%2C+J)

*(Submitted on 30 Aug 2019)*

> Neural machine translation (NMT) has achieved new state-of-the-art performance in translating ambiguous words. However, it is still unclear which component dominates the process of disambiguation. In this paper, we explore the ability of NMT encoders and decoders to disambiguate word senses by evaluating hidden states and investigating the distributions of self-attention. We train a classifier to predict whether a translation is correct given the representation of an ambiguous noun. We find that encoder hidden states outperform word embeddings significantly which indicates that encoders adequately encode relevant information for disambiguation into hidden states. In contrast to encoders, the effect of decoder is different in models with different architectures. Moreover, the attention weights and attention entropy show that self-attention can detect ambiguous nouns and distribute more attention to the context.

| Comments: | Accepted by EMNLP 2019, camera-ready version         |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1908.11771 [cs.CL]**                         |
|           | (or **arXiv:1908.11771v1 [cs.CL]** for this version) |
