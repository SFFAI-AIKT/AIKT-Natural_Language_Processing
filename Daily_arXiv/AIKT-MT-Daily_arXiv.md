# Daily arXiv: Machine Translation - Apr., 2019

### Index

- [2019-04-16](#2019-04-16)
  - [1. Attention-Passing Models for Robust and Data-Efficient End-to-End Speech Translation](#2019-04-16-1)
  - [2. Distributed representation of multi-sense words: A loss-driven approach](#2019-04-16-2)
  - [3. End-to-end Text-to-speech for Low-resource Languages by Cross-Lingual Transfer Learning](#2019-04-16-3)
  - [4. An Empirical Investigation of Global and Local Normalization for Recurrent Neural Sequence Models Using a Continuous Relaxation to Beam Search](#2019-04-16-4)
- [2019-04-15](#2019-04-15)
  - [1. Direct speech-to-speech translation with a sequence-to-sequence model](#2019-04-15-1)
- [2019-04-12](#2019-04-12)
  - [1. Membership Inference Attacks on Sequence-to-Sequence Models](#2019-04-12-1)
  - [2. Scalable Cross-Lingual Transfer of Neural Sentence Embeddings](#2019-04-12-2)
  - [3. Gating Mechanisms for Combining Character and Word-level Word Representations: An Empirical Study](#2019-04-12-3)
- [2019-04-11](#2019-04-11)
  - [1. Cross-lingual Visual Verb Sense Disambiguation](#2019-04-11-1)
- [2019-04-10](#2019-04-10)
  - [1. Text Generation with Exemplar-based Adaptive Decoding](#2019-04-10-1)
  - [2. Bilingual-GAN: A Step Towards Parallel Text Generation](#2019-04-10-2)
  - [3.Text Repair Model for Neural Machine Translation ](#2019-04-10-3)
- [2019-04-09](#2019-04-09)
  - [1. Differentiable Sampling with Flexible Reference Word Order for Neural Machine Translation](#2019-04-09-1)
  - [2. Revisiting Adversarial Autoencoder for Unsupervised Word Translation with Cycle Consistency and Improved Training](#2019-04-09-2)
- [2019-04-08](#2019-04-08)
  - [1. Modeling Recurrence for Transformer](#2019-04-08-1)
  - [2. Convolutional Self-Attention Networks](#2019-04-08-2)
- [2019-04-05](#2019-04-05)
  - [1. Consistency by Agreement in Zero-shot Neural Machine Translation](#2019-04-05-1)
  - [2. The Effect of Downstream Classification Tasks for Evaluating Sentence Embeddings](#2019-04-05-2)
  - [3. Density Matching for Bilingual Word Embedding](#2019-04-05-3)
  - [4. ReWE: Regressing Word Embeddings for Regularization of Neural Machine Translation Systems](#2019-04-05-4)
  - [5. An Attentive Survey of Attention Models](#2019-04-05-5)
- [2019-04-04](#2019-04-04)
  - [1. Attentive Mimicking: Better Word Embeddings by Attending to Informative Contexts](#2019-04-04-1)
  - [2. Identification, Interpretability, and Bayesian Word Embeddings](#2019-04-04-2)
  - [3. Cross-lingual transfer learning for spoken language understanding](#2019-04-04-3)
  - [4. Modeling Vocabulary for Big Code Machine Learning](#2019-04-04-4)
- [2019-04-03](#2019-04-03)
  - [1. Learning to Stop in Structured Prediction for Neural Machine Translation](#2019-04-03-1)
  - [2. A Multi-Task Approach for Disentangling Syntax and Semantics in Sentence Representations](#2019-04-03-2)
  - [3. Pragmatically Informative Text Generation](#2019-04-03-3)
  - [4. Using Multi-Sense Vector Embeddings for Reverse Dictionaries](#2019-04-03-4)
  - [5. Neural Vector Conceptualization for Word Vector Space Interpretation](#2019-04-03-5)
- [2019-04-02](#2019-04-02)
  - [1. Machine translation considering context information using Encoder-Decoder model](#2019-04-01-1)
  - [2. Multimodal Machine Translation with Embedding Prediction](#2019-04-02-2)
  - [3. Lost in Interpretation: Predicting Untranslated Terminology in Simultaneous Interpretation](#2019-04-02-3)

* [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)
* [2019-02](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-02.md)



# 2019-04-16

[Return to Index](#Index)

<h2 id="2019-04-16-1">1. Attention-Passing Models for Robust and Data-Efficient End-to-End Speech Translation</h2>

Title: [Attention-Passing Models for Robust and Data-Efficient End-to-End Speech Translation](<https://arxiv.org/abs/1904.07209>)

Authors: [Matthias Sperber](https://arxiv.org/search/cs?searchtype=author&query=Sperber%2C+M), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G), [Jan Niehues](https://arxiv.org/search/cs?searchtype=author&query=Niehues%2C+J), [Alex Waibel](https://arxiv.org/search/cs?searchtype=author&query=Waibel%2C+A)

*(Submitted on 15 Apr 2019)*

> Speech translation has traditionally been approached through cascaded models consisting of a speech recognizer trained on a corpus of transcribed speech, and a machine translation system trained on parallel texts. Several recent works have shown the feasibility of collapsing the cascade into a single, direct model that can be trained in an end-to-end fashion on a corpus of translated speech. However, experiments are inconclusive on whether the cascade or the direct model is stronger, and have only been conducted under the unrealistic assumption that both are trained on equal amounts of data, ignoring other available speech recognition and machine translation corpora. 
> In this paper, we demonstrate that direct speech translation models require more data to perform well than cascaded models, and while they allow including auxiliary data through multi-task training, they are poor at exploiting such data, putting them at a severe disadvantage. As a remedy, we propose the use of end-to-end trainable models with two attention mechanisms, the first establishing source speech to source text alignments, the second modeling source to target text alignment. We show that such models naturally decompose into multi-task-trainable recognition and translation tasks and propose an attention-passing technique that alleviates error propagation issues in a previous formulation of a model with two attention stages. Our proposed model outperforms all examined baselines and is able to exploit auxiliary training data much more effectively than direct attentional models.

| Comments: | Authors' final version, accepted at TACL 2019                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.07209](https://arxiv.org/abs/1904.07209) [cs.CL] |
|           | (or **arXiv:1904.07209v1 [cs.CL]** for this version)         |



<h2 id="2019-04-16-2">2. Distributed representation of multi-sense words: A loss-driven approach</h2>

Title: [Distributed representation of multi-sense words: A loss-driven approach](<https://arxiv.org/abs/1904.06725>)

Authors: [Saurav Manchanda](https://arxiv.org/search/cs?searchtype=author&query=Manchanda%2C+S), [George Karypis](https://arxiv.org/search/cs?searchtype=author&query=Karypis%2C+G)

*(Submitted on 14 Apr 2019)*

> Word2Vec's Skip Gram model is the current state-of-the-art approach for estimating the distributed representation of words. However, it assumes a single vector per word, which is not well-suited for representing words that have multiple senses. This work presents LDMI, a new model for estimating distributional representations of words. LDMI relies on the idea that, if a word carries multiple senses, then having a different representation for each of its senses should lead to a lower loss associated with predicting its co-occurring words, as opposed to the case when a single vector representation is used for all the senses. After identifying the multi-sense words, LDMI clusters the occurrences of these words to assign a sense to each occurrence. Experiments on the contextual word similarity task show that LDMI leads to better performance than competing approaches.

| Comments:          | PAKDD 2018 Best paper award runner-up                        |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Journal reference: | Advances in Knowledge Discovery and Data Mining. PAKDD 2018. Lecture Notes in Computer Science, vol 10938. Springer, Cham |
| DOI:               | [10.1007/978-3-319-93037-4_27](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1007%252F978-3-319-93037-4_27&v=e65a05b1) |
| Cite as:           | [arXiv:1904.06725](https://arxiv.org/abs/1904.06725) [cs.CL] |
|                    | (or **arXiv:1904.06725v1 [cs.CL]** for this version)         |



<h2 id="2019-04-16-3">3. End-to-end Text-to-speech for Low-resource Languages by Cross-Lingual Transfer Learning</h2>

Title: [End-to-end Text-to-speech for Low-resource Languages by Cross-Lingual Transfer Learning](<https://arxiv.org/abs/1904.06508>)

Authors: [Tao Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+T), [Yuan-Jui Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Cheng-chieh Yeh](https://arxiv.org/search/cs?searchtype=author&query=Yeh%2C+C), [Hung-yi Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+H)

*(Submitted on 13 Apr 2019)*

> End-to-end text-to-speech (TTS) has shown great success on large quantities of paired text plus speech data. However, laborious data collection remains difficult for at least 95% of the languages over the world, which hinders the development of TTS in different languages. In this paper, we aim to build TTS systems for such low-resource (target) languages where only very limited paired data are available. We show such TTS can be effectively constructed by transferring knowledge from a high-resource (source) language. Since the model trained on source language cannot be directly applied to target language due to input space mismatch, we propose a method to learn a mapping between source and target linguistic symbols. Benefiting from this learned mapping, pronunciation information can be preserved throughout the transferring procedure. Preliminary experiments show that we only need around 15 minutes of paired data to obtain a relatively good TTS system. Furthermore, analytic studies demonstrated that the automatically discovered mapping correlate well with the phonetic expertise.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1904.06508](https://arxiv.org/abs/1904.06508) [cs.CL] |
|           | (or **arXiv:1904.06508v1 [cs.CL]** for this version)         |



<h2 id="2019-04-16-4">4. An Empirical Investigation of Global and Local Normalization for Recurrent Neural Sequence Models Using a Continuous Relaxation to Beam Search</h2>

Title: [An Empirical Investigation of Global and Local Normalization for Recurrent Neural Sequence Models Using a Continuous Relaxation to Beam Search](<https://arxiv.org/abs/1904.06834>)

Authors: [Kartik Goyal](https://arxiv.org/search/cs?searchtype=author&query=Goyal%2C+K), [Chris Dyer](https://arxiv.org/search/cs?searchtype=author&query=Dyer%2C+C), [Taylor Berg-Kirkpatrick](https://arxiv.org/search/cs?searchtype=author&query=Berg-Kirkpatrick%2C+T)

*(Submitted on 15 Apr 2019)*

> Globally normalized neural sequence models are considered superior to their locally normalized equivalents because they may ameliorate the effects of label bias. However, when considering high-capacity neural parametrizations that condition on the whole input sequence, both model classes are theoretically equivalent in terms of the distributions they are capable of representing. Thus, the practical advantage of global normalization in the context of modern neural methods remains unclear. In this paper, we attempt to shed light on this problem through an empirical study. We extend an approach for search-aware training via a continuous relaxation of beam search (Goyal et al., 2017b) in order to enable training of globally normalized recurrent sequence models through simple backpropagation. We then use this technique to conduct an empirical study of the interaction between global normalization, high-capacity encoders, and search-aware optimization. We observe that in the context of inexact search, globally normalized neural models are still more effective than their locally normalized counterparts. Further, since our training approach is sensitive to warm-starting with pre-trained models, we also propose a novel initialization strategy based on self-normalization for pre-training globally normalized models. We perform analysis of our approach on two tasks: CCG supertagging and Machine Translation, and demonstrate the importance of global normalization under different conditions while using search-aware training.

| Comments: | Long paper at NAACL 2019                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Machine Learning (stat.ML) |
| Cite as:  | [arXiv:1904.06834](https://arxiv.org/abs/1904.06834) [cs.LG] |
|           | (or **arXiv:1904.06834v1 [cs.LG]** for this version)         |



# 2019-04-15

[Return to Index](#Index)

<h2 id="2019-04-15-1">1. Direct speech-to-speech translation with a sequence-to-sequence model</h2>

Title: [Direct speech-to-speech translation with a sequence-to-sequence model](<https://arxiv.org/abs/1904.06037>)

Authors: [Ye Jia](https://arxiv.org/search/cs?searchtype=author&query=Jia%2C+Y), [Ron J. Weiss](https://arxiv.org/search/cs?searchtype=author&query=Weiss%2C+R+J), [Fadi Biadsy](https://arxiv.org/search/cs?searchtype=author&query=Biadsy%2C+F), [Wolfgang Macherey](https://arxiv.org/search/cs?searchtype=author&query=Macherey%2C+W), [Melvin Johnson](https://arxiv.org/search/cs?searchtype=author&query=Johnson%2C+M), [Zhifeng Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Z), [Yonghui Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+Y)

*(Submitted on 12 Apr 2019)*

> We present an attention-based sequence-to-sequence neural network which can directly translate speech from one language into speech in another language, without relying on an intermediate text representation. The network is trained end-to-end, learning to map speech spectrograms into target spectrograms in another language, corresponding to the translated content (in a different canonical voice). We further demonstrate the ability to synthesize translated speech using the voice of the source speaker. We conduct experiments on two Spanish-to-English speech translation datasets, and find that the proposed model slightly underperforms a baseline cascade of a direct speech-to-text translation model and a text-to-speech synthesis model, demonstrating the feasibility of the approach on this very challenging task.

| Comments: | Submitted to Interspeech 2019                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | [arXiv:1904.06037](https://arxiv.org/abs/1904.06037) [cs.CL] |
|           | (or **arXiv:1904.06037v1 [cs.CL]** for this version)         |



# 2019-04-12

[Return to Index](#Index)

<h2 id="2019-04-12-1">1. Membership Inference Attacks on Sequence-to-Sequence Models</h2> 

Title: [Membership Inference Attacks on Sequence-to-Sequence Models](<https://arxiv.org/abs/1904.05506>)

Authors: [Sorami Hisamoto](https://arxiv.org/search/cs?searchtype=author&query=Hisamoto%2C+S), [Matt Post](https://arxiv.org/search/cs?searchtype=author&query=Post%2C+M), [Kevin Duh](https://arxiv.org/search/cs?searchtype=author&query=Duh%2C+K)

*(Submitted on 11 Apr 2019)*

> Data privacy is an important issue for "machine learning as a service" providers. We focus on the problem of membership inference attacks: given a data sample and black-box access to a model's API, determine whether the sample existed in the model's training data. Our contribution is an investigation of this problem in the context of sequence-to-sequence models, which are important in applications such as machine translation and video captioning. We define the membership inference problem for sequence generation, provide an open dataset based on state-of-the-art machine translation models, and report initial results on whether these models leak private information against several kinds of membership inference attacks.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1904.05506](https://arxiv.org/abs/1904.05506) [cs.LG] |
|           | (or **arXiv:1904.05506v1 [cs.LG]** for this version)         |



<h2 id="2019-04-12-2">2. Scalable Cross-Lingual Transfer of Neural Sentence Embeddings</h2> 

Title: [Scalable Cross-Lingual Transfer of Neural Sentence Embeddings](<https://arxiv.org/abs/1904.05542>)

Authors: [Hanan Aldarmaki](https://arxiv.org/search/cs?searchtype=author&query=Aldarmaki%2C+H), [Mona Diab](https://arxiv.org/search/cs?searchtype=author&query=Diab%2C+M)

*(Submitted on 11 Apr 2019)*

> We develop and investigate several cross-lingual alignment approaches for neural sentence embedding models, such as the supervised inference classifier, InferSent, and sequential encoder-decoder models. We evaluate three alignment frameworks applied to these models: joint modeling, representation transfer learning, and sentence mapping, using parallel text to guide the alignment. Our results support representation transfer as a scalable approach for modular cross-lingual alignment of neural sentence embeddings, where we observe better performance compared to joint models in intrinsic and extrinsic evaluations, particularly with smaller sets of parallel data.

| Comments: | accepted in *SEM 2019                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.05542](https://arxiv.org/abs/1904.05542) [cs.CL] |
|           | (or **arXiv:1904.05542v1 [cs.CL]** for this version)         |



<h2 id="2019-04-12-3">3. Gating Mechanisms for Combining Character and Word-level Word Representations: An Empirical Study
</h2> 

Title: [Gating Mechanisms for Combining Character and Word-level Word Representations: An Empirical Study](<https://arxiv.org/abs/1904.05584>)

Authors: [Jorge A. Balazs](https://arxiv.org/search/cs?searchtype=author&query=Balazs%2C+J+A), [Yutaka Matsuo](https://arxiv.org/search/cs?searchtype=author&query=Matsuo%2C+Y)

*(Submitted on 11 Apr 2019)*

> In this paper we study how different ways of combining character and word-level representations affect the quality of both final word and sentence representations. We provide strong empirical evidence that modeling characters improves the learned representations at the word and sentence levels, and that doing so is particularly useful when representing less frequent words. We further show that a feature-wise sigmoid gating mechanism is a robust method for creating representations that encode semantic similarity, as it performed reasonably well in several word similarity datasets. Finally, our findings suggest that properly capturing semantic similarity at the word level does not consistently yield improved performance in downstream sentence-level tasks. Our code is available at [this https URL](https://github.com/jabalazs/gating)

| Comments: | Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Student Research Workshop |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (stat.ML) |
| Cite as:  | [arXiv:1904.05584](https://arxiv.org/abs/1904.05584) [cs.CL] |
|           | (or **arXiv:1904.05584v1 [cs.CL]** for this version)         |



# 2019-04-11

[Return to Index](#Index)

<h2 id="2019-04-11-1">1. Cross-lingual Visual Verb Sense Disambiguation</h2> 

Title: [Cross-lingual Visual Verb Sense Disambiguation](<https://arxiv.org/abs/1904.05092>)

Authors: [Spandana Gella](https://arxiv.org/search/cs?searchtype=author&query=Gella%2C+S), [Desmond Elliott](https://arxiv.org/search/cs?searchtype=author&query=Elliott%2C+D), [Frank Keller](https://arxiv.org/search/cs?searchtype=author&query=Keller%2C+F)

*(Submitted on 10 Apr 2019)*

> Recent work has shown that visual context improves cross-lingual sense disambiguation for nouns. We extend this line of work to the more challenging task of cross-lingual verb sense disambiguation, introducing the MultiSense dataset of 9,504 images annotated with English, German, and Spanish verbs. Each image in MultiSense is annotated with an English verb and its translation in German or Spanish. We show that cross-lingual verb sense disambiguation models benefit from visual context, compared to unimodal baselines. We also show that the verb sense predicted by our best disambiguation model can improve the results of a text-only machine translation system when used for a multimodal translation task.

| Comments: | NAACL 2019                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | [arXiv:1904.05092](https://arxiv.org/abs/1904.05092) [cs.CL] |
|           | (or **arXiv:1904.05092v1 [cs.CL]** for this version)         |



# 2019-04-10

[Return to Index](#Index)

<h2 id="2019-04-10-1">1. Text Generation with Exemplar-based Adaptive Decoding</h2> 

Title: [Text Generation with Exemplar-based Adaptive Decoding](<https://arxiv.org/abs/1904.04428>)

Authors: [Hao Peng](https://arxiv.org/search/cs?searchtype=author&query=Peng%2C+H), [Ankur P. Parikh](https://arxiv.org/search/cs?searchtype=author&query=Parikh%2C+A+P), [Manaal Faruqui](https://arxiv.org/search/cs?searchtype=author&query=Faruqui%2C+M), [Bhuwan Dhingra](https://arxiv.org/search/cs?searchtype=author&query=Dhingra%2C+B), [Dipanjan Das](https://arxiv.org/search/cs?searchtype=author&query=Das%2C+D)

*(Submitted on 9 Apr 2019)*

> We propose a novel conditioned text generation model. It draws inspiration from traditional template-based text generation techniques, where the source provides the content (i.e., what to say), and the template influences how to say it. Building on the successful encoder-decoder paradigm, it first encodes the content representation from the given input text; to produce the output, it retrieves exemplar text from the training data as "soft templates," which are then used to construct an exemplar-specific decoder. We evaluate the proposed model on abstractive text summarization and data-to-text generation. Empirical results show that this model achieves strong performance and outperforms comparable baselines.

| Comments: | NAACL 2019                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.04428](https://arxiv.org/abs/1904.04428) [cs.CL] |
|           | (or **arXiv:1904.04428v1 [cs.CL]** for this version)         |



<h2 id="2019-04-10-2">2. Bilingual-GAN: A Step Towards Parallel Text Generation</h2> 

Title: [Bilingual-GAN: A Step Towards Parallel Text Generation](<https://arxiv.org/abs/1904.04742>)

Authors: [Ahmad Rashid](https://arxiv.org/search/cs?searchtype=author&query=Rashid%2C+A), [Alan Do-Omri](https://arxiv.org/search/cs?searchtype=author&query=Do-Omri%2C+A), [Md. Akmal Haidar](https://arxiv.org/search/cs?searchtype=author&query=Haidar%2C+M+A), [Qun Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Q), [Mehdi Rezagholizadeh](https://arxiv.org/search/cs?searchtype=author&query=Rezagholizadeh%2C+M)

*(Submitted on 9 Apr 2019)*

> Latent space based GAN methods and attention based sequence to sequence models have achieved impressive results in text generation and unsupervised machine translation respectively. Leveraging the two domains, we propose an adversarial latent space based model capable of generating parallel sentences in two languages concurrently and translating bidirectionally. The bilingual generation goal is achieved by sampling from the latent space that is shared between both languages. First two denoising autoencoders are trained, with shared encoders and back-translation to enforce a shared latent state between the two languages. The decoder is shared for the two translation directions. Next, a GAN is trained to generate synthetic "code" mimicking the languages' shared latent space. This code is then fed into the decoder to generate text in either language. We perform our experiments on Europarl and Multi30k datasets, on the English-French language pair, and document our performance using both supervised and unsupervised machine translation.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1904.04742](https://arxiv.org/abs/1904.04742) [cs.CL] |
|           | (or **arXiv:1904.04742v1 [cs.CL]** for this version)         |



<h2 id="2019-04-10-3">3. Text Repair Model for Neural Machine Translation</h2> 

Title: [Text Repair Model for Neural Machine Translation](<https://arxiv.org/abs/1904.04790>)

Authors: [Markus Freitag](https://arxiv.org/search/cs?searchtype=author&query=Freitag%2C+M), [Isaac Caswell](https://arxiv.org/search/cs?searchtype=author&query=Caswell%2C+I), [Scott Roy](https://arxiv.org/search/cs?searchtype=author&query=Roy%2C+S)

*(Submitted on 9 Apr 2019)*

> In this work, we train a text repair model as a post-processor for Neural Machine Translation (NMT). The goal of the repair model is to correct typical errors introduced by the translation process, and convert the "translationese" output into natural text. The repair model is trained on monolingual data that has been round-trip translated through English, to mimic errors that are similar to the ones introduced by NMT. Having a trained repair model, we apply it to the output of existing NMT systems. We run experiments for both the WMT18 English to German and the WMT16 English to Romanian task. Furthermore, we apply the repair model on the output of the top submissions of the most recent WMT evaluation campaigns. We see quality improvements on all tasks of up to 2.5 BLEU points.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1904.04790](https://arxiv.org/abs/1904.04790) [cs.CL] |
|           | (or **arXiv:1904.04790v1 [cs.CL]** for this version)         |





# 2019-04-09

[Return to Index](#Index)

<h2 id="2019-04-09-1">1. Differentiable Sampling with Flexible Reference Word Order for Neural Machine Translation</h2> 

Title: [Differentiable Sampling with Flexible Reference Word Order for Neural Machine Translation](<https://arxiv.org/abs/1904.04079>)

Authors: [Weijia Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+W), [Xing Niu](https://arxiv.org/search/cs?searchtype=author&query=Niu%2C+X), [Marine Carpuat](https://arxiv.org/search/cs?searchtype=author&query=Carpuat%2C+M)

*(Submitted on 4 Apr 2019)*

> Despite some empirical success at correcting exposure bias in machine translation, scheduled sampling algorithms suffer from a major drawback: they incorrectly assume that words in the reference translations and in sampled sequences are aligned at each time step. Our new differentiable sampling algorithm addresses this issue by optimizing the probability that the reference can be aligned with the sampled output, based on a soft alignment predicted by the model itself. As a result, the output distribution at each time step is evaluated with respect to the whole predicted sequence. Experiments on IWSLT translation tasks show that our approach improves BLEU compared to maximum likelihood and scheduled sampling baselines. In addition, our approach is simpler to train with no need for sampling schedule and yields models that achieve larger improvements with smaller beam sizes.

| Comments: | Accepted at NAACL 2019                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (stat.ML) |
| Cite as:  | [arXiv:1904.04079](https://arxiv.org/abs/1904.04079) [cs.CL] |
|           | (or **arXiv:1904.04079v1 [cs.CL]** for this version)         |



<h2 id="2019-04-09-2">2. Revisiting Adversarial Autoencoder for Unsupervised Word Translation with Cycle Consistency and Improved Training</h2> 

Title: [Revisiting Adversarial Autoencoder for Unsupervised Word Translation with Cycle Consistency and Improved Training](<https://arxiv.org/abs/1904.04116>)

Authors: [Tasnim Mohiuddin](https://arxiv.org/search/cs?searchtype=author&query=Mohiuddin%2C+T), [Shafiq Joty](https://arxiv.org/search/cs?searchtype=author&query=Joty%2C+S)

*(Submitted on 4 Apr 2019)*

> Adversarial training has shown impressive success in learning bilingual dictionary without any parallel data by mapping monolingual embeddings to a shared space. However, recent work has shown superior performance for non-adversarial methods in more challenging language pairs. In this work, we revisit adversarial autoencoder for unsupervised word translation and propose two novel extensions to it that yield more stable training and improved results. Our method includes regularization terms to enforce cycle consistency and input reconstruction, and puts the target encoders as an adversary against the corresponding discriminator. Extensive experimentations with European, non-European and low-resource languages show that our method is more robust and achieves better performance than recently proposed adversarial and non-adversarial approaches.

| Comments: | Published in NAACL-HLT 2019                                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Machine Learning (stat.ML) |
| Cite as:  | [arXiv:1904.04116](https://arxiv.org/abs/1904.04116) [cs.CL] |
|           | (or **arXiv:1904.04116v1 [cs.CL]** for this version)         |





# 2019-04-08

[Return to Index](#Index)

<h2 id="2019-04-08-1">1. Modeling Recurrence for Transformer</h2> 

Title： [Modeling Recurrence for Transformer](<https://arxiv.org/abs/1904.03092>)

Authors: [Jie Hao](https://arxiv.org/search/cs?searchtype=author&query=Hao%2C+J), [Xing Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Baosong Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+B), [Longyue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Jinfeng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+J), [Zhaopeng Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+Z)

*(Submitted on 5 Apr 2019)*

> Recently, the Transformer model that is based solely on attention mechanisms, has advanced the state-of-the-art on various machine translation tasks. However, recent studies reveal that the lack of recurrence hinders its further improvement of translation capacity. In response to this problem, we propose to directly model recurrence for Transformer with an additional recurrence encoder. In addition to the standard recurrent neural network, we introduce a novel attentive recurrent network to leverage the strengths of both attention and recurrent networks. Experimental results on the widely-used WMT14 English-German and WMT17 Chinese-English translation tasks demonstrate the effectiveness of the proposed approach. Our studies also reveal that the proposed model benefits from a short-cut that bridges the source and target sequences with a single recurrent layer, which outperforms its deep counterpart.

| Comments: | NAACL 2019                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | [arXiv:1904.03092](https://arxiv.org/abs/1904.03092) [cs.CL] |
|           | (or **arXiv:1904.03092v1 [cs.CL]** for this version)         |



<h2 id="2019-04-08-2">2. Convolutional Self-Attention Networks</h2> 

Title: [Convolutional Self-Attention Networks](<https://arxiv.org/abs/1904.03107>)

Authors: [Baosong Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+B), [Longyue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Derek Wong](https://arxiv.org/search/cs?searchtype=author&query=Wong%2C+D), [Lidia S. Chao](https://arxiv.org/search/cs?searchtype=author&query=Chao%2C+L+S), [Zhaopeng Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+Z)

*(Submitted on 5 Apr 2019)*

> Self-attention networks (SANs) have drawn increasing interest due to their high parallelization in computation and flexibility in modeling dependencies. SANs can be further enhanced with multi-head attention by allowing the model to attend to information from different representation subspaces. In this work, we propose novel convolutional self-attention networks, which offer SANs the abilities to 1) strengthen dependencies among neighboring elements, and 2) model the interaction between features extracted by multiple attention heads. Experimental results of machine translation on different language pairs and model settings show that our approach outperforms both the strong Transformer baseline and other existing models on enhancing the locality of SANs. Comparing with prior studies, the proposed model is parameter free in terms of introducing no more parameters.

| Comments: | NAACL 2019                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | [arXiv:1904.03107](https://arxiv.org/abs/1904.03107) [cs.CL] |
|           | (or **arXiv:1904.03107v1 [cs.CL]** for this version)         |



# 2019-04-05 

[Return to Index](#Index)

<h2 id="2019-04-05-1">1. Consistency by Agreement in Zero-shot Neural Machine Translation</h2> 

Title: [Consistency by Agreement in Zero-shot Neural Machine Translation](<https://arxiv.org/abs/1904.02338>)

Authors: [Maruan Al-Shedivat](https://arxiv.org/search/cs?searchtype=author&query=Al-Shedivat%2C+M), [Ankur P. Parikh](https://arxiv.org/search/cs?searchtype=author&query=Parikh%2C+A+P)

*(Submitted on 4 Apr 2019)*

> Generalization and reliability of multilingual translation often highly depend on the amount of available parallel data for each language pair of interest. In this paper, we focus on zero-shot generalization---a challenging setup that tests models on translation directions they have not been optimized for at training time. To solve the problem, we (i) reformulate multilingual translation as probabilistic inference, (ii) define the notion of zero-shot consistency and show why standard training often results in models unsuitable for zero-shot tasks, and (iii) introduce a consistent agreement-based training method that encourages the model to produce equivalent translations of parallel sentences in auxiliary languages. We test our multilingual NMT models on multiple public zero-shot translation benchmarks (IWSLT17, UN corpus, Europarl) and show that agreement-based learning often results in 2-3 BLEU zero-shot improvement over strong baselines without any loss in performance on supervised translation directions.

| Comments: | NAACL 2019                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Neural and Evolutionary Computing (cs.NE); Machine Learning (stat.ML) |
| Cite as:  | [arXiv:1904.02338](https://arxiv.org/abs/1904.02338) [cs.LG] |
|           | (or **arXiv:1904.02338v1 [cs.LG]** for this version)         |



<h2 id="2019-04-05-2">2. The Effect of Downstream Classification Tasks for Evaluating Sentence Embeddings</h2> 

Title: [The Effect of Downstream Classification Tasks for Evaluating Sentence Embeddings](<https://arxiv.org/abs/1904.02228>)

Authors: [Peter Potash](https://arxiv.org/search/cs?searchtype=author&query=Potash%2C+P)

*(Submitted on 3 Apr 2019)*

> One popular method for quantitatively evaluating the performance of sentence embeddings involves their usage on downstream language processing tasks that require sentence representations as input. One simple such task is classification, where the sentence representations are used to train and test models on several classification datasets. We argue that by evaluating sentence representations in such a manner, the goal of the representations becomes learning a low-dimensional factorization of a sentence-task label matrix. We show how characteristics of this matrix can affect the ability for a low-dimensional factorization to perform as sentence representations in a suite of classification tasks. Primarily, sentences that have more labels across all possible classification tasks have a higher reconstruction loss, though this effect can be drastically negated if the amount of such sentences is small.

| Comments: | 5 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.02228](https://arxiv.org/abs/1904.02228) [cs.CL] |
|           | (or **arXiv:1904.02228v1 [cs.CL]** for this version)         |



<h2 id="2019-04-05-3">3. Density Matching for Bilingual Word Embedding</h2> 

Title： [Density Matching for Bilingual Word Embedding](<https://arxiv.org/abs/1904.02343>)

Authors: [Chunting Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+C), [Xuezhe Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+X), [Di Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+D), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G)

*(Submitted on 4 Apr 2019)*

> Recent approaches to cross-lingual word embedding have generally been based on linear transformations between the sets of embedding vectors in the two languages. In this paper, we propose an approach that instead expresses the two monolingual embedding spaces as probability densities defined by a Gaussian mixture model, and matches the two densities using a method called normalizing flow. The method requires no explicit supervision, and can be learned with only a seed dictionary of words that have identical strings. We argue that this formulation has several intuitively attractive properties, particularly with the respect to improving robustness and generalization to mappings between difficult language pairs or word pairs. On a benchmark data set of bilingual lexicon induction and cross-lingual word similarity, our approach can achieve competitive or superior performance compared to state-of-the-art published results, with particularly strong results being found on etymologically distant and/or morphologically rich languages.

| Comments: | Accepted by NAACL-HLT 2019                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1904.02343](https://arxiv.org/abs/1904.02343) [cs.CL] |
|           | (or **arXiv:1904.02343v1 [cs.CL]** for this version)         |



<h2 id="2019-04-05-4">4. ReWE: Regressing Word Embeddings for Regularization of Neural Machine Translation Systems</h2> 

Title: [ReWE: Regressing Word Embeddings for Regularization of Neural Machine Translation Systems](<https://arxiv.org/abs/1904.02461>)

Authors: [Inigo Jauregi Unanue](https://arxiv.org/search/cs?searchtype=author&query=Unanue%2C+I+J), [Ehsan Zare Borzeshi](https://arxiv.org/search/cs?searchtype=author&query=Borzeshi%2C+E+Z), [Nazanin Esmaili](https://arxiv.org/search/cs?searchtype=author&query=Esmaili%2C+N), [Massimo Piccardi](https://arxiv.org/search/cs?searchtype=author&query=Piccardi%2C+M)

*(Submitted on 4 Apr 2019)*

> Regularization of neural machine translation is still a significant problem, especially in low-resource settings. To mollify this problem, we propose regressing word embeddings (ReWE) as a new regularization technique in a system that is jointly trained to predict the next word in the translation (categorical value) and its word embedding (continuous value). Such a joint training allows the proposed system to learn the distributional properties represented by the word embeddings, empirically improving the generalization to unseen sentences. Experiments over three translation datasets have showed a consistent improvement over a strong baseline, ranging between 0.91 and 2.54 BLEU points, and also a marked improvement over a state-of-the-art system.

| Comments: | Accepted at NAACL-HLT 2019                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.02461](https://arxiv.org/abs/1904.02461) [cs.CL] |
|           | (or **arXiv:1904.02461v1 [cs.CL]** for this version)         |



<h2 id="2019-04-05-5">5. An Attentive Survey of Attention Models</h2> 

Title: [An Attentive Survey of Attention Models](<https://arxiv.org/abs/1904.02874>)

Authors: [Sneha Chaudhari](https://arxiv.org/search/cs?searchtype=author&query=Chaudhari%2C+S), [Gungor Polatkan](https://arxiv.org/search/cs?searchtype=author&query=Polatkan%2C+G), [Rohan Ramanath](https://arxiv.org/search/cs?searchtype=author&query=Ramanath%2C+R), [Varun Mithal](https://arxiv.org/search/cs?searchtype=author&query=Mithal%2C+V)

*(Submitted on 5 Apr 2019)*

> Attention Model has now become an important concept in neural networks that has been researched within diverse application domains. This survey provides a structured and comprehensive overview of the developments in modeling attention. In particular, we propose a taxonomy which groups existing techniques into coherent categories. We review the different neural architectures in which attention has been incorporated, and also show how attention improves interpretability of neural models. Finally, we discuss some applications in which modeling attention has a significant impact. We hope this survey will provide a succinct introduction to attention models and guide practitioners while developing approaches for their applications.

| Comments: | submitted to IJCAI 2019 Survey Track; 6 pages, 4 figures, 2 tables |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Machine Learning (stat.ML)     |
| Cite as:  | [arXiv:1904.02874](https://arxiv.org/abs/1904.02874) [cs.LG] |
|           | (or **arXiv:1904.02874v1 [cs.LG]** for this version)         |



# 2019-04-04

[Return to Index](#Index)

<h2 id="2019-04-04-1">1. Attentive Mimicking: Better Word Embeddings by Attending to Informative Contexts</h2> 

Title: [Attentive Mimicking: Better Word Embeddings by Attending to Informative Contexts](<https://arxiv.org/abs/1904.01617>)

Authors: [Timo Schick](https://arxiv.org/search/cs?searchtype=author&query=Schick%2C+T), [Hinrich Schütze](https://arxiv.org/search/cs?searchtype=author&query=Sch%C3%BCtze%2C+H)

*(Submitted on 2 Apr 2019)*

> Learning high-quality embeddings for rare words is a hard problem because of sparse context information. Mimicking (Pinter et al., 2017) has been proposed as a solution: given embeddings learned by a standard algorithm, a model is first trained to reproduce embeddings of frequent words from their surface form and then used to compute embeddings for rare words. In this paper, we introduce attentive mimicking: the mimicking model is given access not only to a word's surface form, but also to all available contexts and learns to attend to the most informative and reliable contexts for computing an embedding. In an evaluation on four tasks, we show that attentive mimicking outperforms previous work for both rare and medium-frequency words. Thus, compared to previous work, attentive mimicking improves embeddings for a much larger part of the vocabulary, including the medium-frequency range.

| Comments: | Accepted at NAACL2019                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.01617](https://arxiv.org/abs/1904.01617) [cs.CL] |
|           | (or **arXiv:1904.01617v1 [cs.CL]** for this version)         |



<h2 id="2019-04-04-2">2. Identification, Interpretability, and Bayesian Word Embeddings</h2> 

Title: [Identification, Interpretability, and Bayesian Word Embeddings](<https://arxiv.org/abs/1904.01628>)

Authors: [Adam M. Lauretig](https://arxiv.org/search/cs?searchtype=author&query=Lauretig%2C+A+M)

*(Submitted on 2 Apr 2019)*

> Social scientists have recently turned to analyzing text using tools from natural language processing like word embeddings to measure concepts like ideology, bias, and affinity. However, word embeddings are difficult to use in the regression framework familiar to social scientists: embeddings are are neither identified, nor directly interpretable. I offer two advances on standard embedding models to remedy these problems. First, I develop Bayesian Word Embeddings with Automatic Relevance Determination priors, relaxing the assumption that all embedding dimensions have equal weight. Second, I apply work identifying latent variable models to anchor the dimensions of the resulting embeddings, identifying them, and making them interpretable and usable in a regression. I then apply this model and anchoring approach to two cases, the shift in internationalist rhetoric in the American presidents' inaugural addresses, and the relationship between bellicosity in American foreign policy decision-makers' deliberations. I find that inaugural addresses became less internationalist after 1945, which goes against the conventional wisdom, and that an increase in bellicosity is associated with an increase in hostile actions by the United States, showing that elite deliberations are not cheap talk, and helping confirm the validity of the model.

| Comments: | Accepted to the Third Workshop on Natural Language Processing and Computational Social Science at NAACL-HLT 2019 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Applications (stat.AP) |
| Cite as:  | [arXiv:1904.01628](https://arxiv.org/abs/1904.01628) [cs.CL] |
|           | (or **arXiv:1904.01628v1 [cs.CL]** for this version)         |

<h2 id="2019-04-04-3">3. Cross-lingual transfer learning for spoken language understanding</h2> 

Title: [Cross-lingual transfer learning for spoken language understanding](<https://arxiv.org/abs/1904.01825>)

Authors: [Quynh Ngoc Thi Do](https://arxiv.org/search/cs?searchtype=author&query=Do%2C+Q+N+T), [Judith Gaspers](https://arxiv.org/search/cs?searchtype=author&query=Gaspers%2C+J)

*(Submitted on 3 Apr 2019)*

> Typically, spoken language understanding (SLU) models are trained on annotated data which are costly to gather. Aiming to reduce data needs for bootstrapping a SLU system for a new language, we present a simple but effective weight transfer approach using data from another language. The approach is evaluated with our promising multi-task SLU framework developed towards different languages. We evaluate our approach on the ATIS and a real-world SLU dataset, showing that i) our monolingual models outperform the state-of-the-art, ii) we can reduce data amounts needed for bootstrapping a SLU system for a new language greatly, and iii) while multitask training improves over separate training, different weight transfer settings may work best for different SLU modules.

| Comments: | accepted at ICASSP, 2019                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.01825](https://arxiv.org/abs/1904.01825) [cs.CL] |
|           | (or **arXiv:1904.01825v1 [cs.CL]** for this version)         |



<h2 id="2019-04-04-4">4. Modeling Vocabulary for Big Code Machine Learning</h2> 

Title: [Modeling Vocabulary for Big Code Machine Learning](<https://arxiv.org/abs/1904.01873>)

Authors: [Hlib Babii](https://arxiv.org/search/cs?searchtype=author&query=Babii%2C+H), [Andrea Janes](https://arxiv.org/search/cs?searchtype=author&query=Janes%2C+A), [Romain Robbes](https://arxiv.org/search/cs?searchtype=author&query=Robbes%2C+R)

*(Submitted on 3 Apr 2019)*

> When building machine learning models that operate on source code, several decisions have to be made to model source-code vocabulary. These decisions can have a large impact: some can lead to not being able to train models at all, others significantly affect performance, particularly for Neural Language Models. Yet, these decisions are not often fully described. This paper lists important modeling choices for source code vocabulary, and explores their impact on the resulting vocabulary on a large-scale corpus of 14,436 projects. We show that a subset of decisions have decisive characteristics, allowing to train accurate Neural Language Models quickly on a large corpus of 10,106 projects.

| Comments: | 12 pages, 1 figure                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Software Engineering (cs.SE) |
| Cite as:  | [arXiv:1904.01873](https://arxiv.org/abs/1904.01873) [cs.CL] |
|           | (or **arXiv:1904.01873v1 [cs.CL]** for this version)         |



# 2019-04-03

[Return to Index](#Index)

<h2 id="2019-04-03-1">1. Learning to Stop in Structured Prediction for Neural Machine Translation</h2> 

Title: [Learning to Stop in Structured Prediction for Neural Machine Translation](<https://arxiv.org/abs/1904.01032>)

Authors: [Mingbo Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+M), [Renjie Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+R), [Liang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+L)

*(Submitted on 1 Apr 2019)*

> Beam search optimization resolves many issues in neural machine translation. However, this method lacks principled stopping criteria and does not learn how to stop during training, and the model naturally prefers the longer hypotheses during the testing time in practice since they use the raw score instead of the probability-based score. We propose a novel ranking method which enables an optimal beam search stopping criteria. We further introduce a structured prediction loss function which penalizes suboptimal finished candidates produced by beam search during training. Experiments of neural machine translation on both synthetic data and real languages (German-to-English and Chinese-to-English) demonstrate our proposed methods lead to better length and BLEU score.

| Comments:          | 5 pages                                                      |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**                         |
| Journal reference: | NAACL 2019                                                   |
| Cite as:           | [arXiv:1904.01032](https://arxiv.org/abs/1904.01032) [cs.CL] |
|                    | (or **arXiv:1904.01032v1 [cs.CL]** for this version)         |



<h2 id="2019-04-03-2">2. A Multi-Task Approach for Disentangling Syntax and Semantics in Sentence Representations</h2> 

Title: [A Multi-Task Approach for Disentangling Syntax and Semantics in Sentence Representations](<https://arxiv.org/abs/1904.01173>)

Authors: [Mingda Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+M), [Qingming Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+Q), [Sam Wiseman](https://arxiv.org/search/cs?searchtype=author&query=Wiseman%2C+S), [Kevin Gimpel](https://arxiv.org/search/cs?searchtype=author&query=Gimpel%2C+K)

*(Submitted on 2 Apr 2019)*

> We propose a generative model for a sentence that uses two latent variables, with one intended to represent the syntax of the sentence and the other to represent its semantics. We show we can achieve better disentanglement between semantic and syntactic representations by training with multiple losses, including losses that exploit aligned paraphrastic sentences and word-order information. We also investigate the effect of moving from bag-of-words to recurrent neural network modules. We evaluate our models as well as several popular pretrained embeddings on standard semantic similarity tasks and novel syntactic similarity tasks. Empirically, we find that the model with the best performing syntactic and semantic representations also gives rise to the most disentangled representations.

| Comments:          | NAACL 2019 Long paper                                        |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**                         |
| Journal reference: | NAACL 2019                                                   |
| Cite as:           | [arXiv:1904.01173](https://arxiv.org/abs/1904.01173) [cs.CL] |
|                    | (or **arXiv:1904.01173v1 [cs.CL]** for this version)         |



<h2 id="2019-04-03-3">3. Pragmatically Informative Text Generation</h2> 

Title: [Pragmatically Informative Text Generation](<https://arxiv.org/abs/1904.01301>)

Authors: [Sheng Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+S), [Daniel Fried](https://arxiv.org/search/cs?searchtype=author&query=Fried%2C+D), [Jacob Andreas](https://arxiv.org/search/cs?searchtype=author&query=Andreas%2C+J), [Dan Klein](https://arxiv.org/search/cs?searchtype=author&query=Klein%2C+D)

*(Submitted on 2 Apr 2019)*

> We improve the informativeness of models for conditional text generation using techniques from computational pragmatics. These techniques formulate language production as a game between speakers and listeners, in which a speaker should generate output text that a listener can use to correctly identify the original input that the text describes. While such approaches are widely used in cognitive science and grounded language learning, they have received less attention for more standard language generation tasks. We consider two pragmatic modeling methods for text generation: one where pragmatics is imposed by information preservation, and another where pragmatics is imposed by explicit modeling of distractors. We find that these methods improve the performance of strong existing systems for abstractive summarization and generation from structured meaning representations.

| Comments: | 8 pages. accepted as a conference paper at NAACL2019 (short paper) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.01301](https://arxiv.org/abs/1904.01301) [cs.CL] |
|           | (or **arXiv:1904.01301v1 [cs.CL]** for this version)         |



<h2 id="2019-04-03-4">4. Using Multi-Sense Vector Embeddings for Reverse Dictionaries</h2> 

Title: [Using Multi-Sense Vector Embeddings for Reverse Dictionaries](<https://arxiv.org/abs/1904.01451>)

Authors: [Michael A. Hedderich](https://arxiv.org/search/cs?searchtype=author&query=Hedderich%2C+M+A), [Andrew Yates](https://arxiv.org/search/cs?searchtype=author&query=Yates%2C+A), [Dietrich Klakow](https://arxiv.org/search/cs?searchtype=author&query=Klakow%2C+D), [Gerard de Melo](https://arxiv.org/search/cs?searchtype=author&query=de+Melo%2C+G)

*(Submitted on 2 Apr 2019)*

> Popular word embedding methods such as word2vec and GloVe assign a single vector representation to each word, even if a word has multiple distinct meanings. Multi-sense embeddings instead provide different vectors for each sense of a word. However, they typically cannot serve as a drop-in replacement for conventional single-sense embeddings, because the correct sense vector needs to be selected for each word. In this work, we study the effect of multi-sense embeddings on the task of reverse dictionaries. We propose a technique to easily integrate them into an existing neural network architecture using an attention mechanism. Our experiments demonstrate that large improvements can be obtained when employing multi-sense embeddings both in the input sequence as well as for the target representation. An analysis of the sense distributions and of the learned attention is provided as well.

| Comments: | Accepted as long paper at the 13th International Conference on Computational Semantics (IWCS 2019) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1904.01451](https://arxiv.org/abs/1904.01451) [cs.CL] |
|           | (or **arXiv:1904.01451v1 [cs.CL]** for this version)         |



<h2 id="2019-04-03-5">5. Neural Vector Conceptualization for Word Vector Space Interpretation</h2> 

Title: [Neural Vector Conceptualization for Word Vector Space Interpretation](<https://arxiv.org/abs/1904.01500>)

Authors: [Robert Schwarzenberg](https://arxiv.org/search/cs?searchtype=author&query=Schwarzenberg%2C+R), [Lisa Raithel](https://arxiv.org/search/cs?searchtype=author&query=Raithel%2C+L), [David Harbecke](https://arxiv.org/search/cs?searchtype=author&query=Harbecke%2C+D)

*(Submitted on 2 Apr 2019)*

> Distributed word vector spaces are considered hard to interpret which hinders the understanding of natural language processing (NLP) models. In this work, we introduce a new method to interpret arbitrary samples from a word vector space. To this end, we train a neural model to conceptualize word vectors, which means that it activates higher order concepts it recognizes in a given vector. Contrary to prior approaches, our model operates in the original vector space and is capable of learning non-linear relations between word vectors and concepts. Furthermore, we show that it produces considerably less entropic concept activation profiles than the popular cosine similarity.

| Comments: | NAACL-HLT 2019 Workshop on Evaluating Vector Space Representations for NLP (RepEval) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1904.01500](https://arxiv.org/abs/1904.01500) [cs.CL] |
|           | (or **arXiv:1904.01500v1 [cs.CL]** for this version)         |



# 2019-04-02

[Return to Index](#Index)

<h2 id="2019-04-02-1">1. Machine translation considering context information using Encoder-Decoder model</h2> 

Title: [Machine translation considering context information using Encoder-Decoder model](<https://arxiv.org/abs/1904.00160>)

Authors:[Tetsuto Takano](https://arxiv.org/search/cs?searchtype=author&query=Takano%2C+T), [Satoshi Yamane](https://arxiv.org/search/cs?searchtype=author&query=Yamane%2C+S)

*(Submitted on 30 Mar 2019)*

> In the task of machine translation, context information is one of the important factor. But considering the context information model dose not proposed. The paper propose a new model which can integrate context information and make translation. In this paper, we create a new model based Encoder Decoder model. When translating current sentence, the model integrates output from preceding encoder with current encoder. The model can consider context information and the result score is higher than existing model.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1904.00160](https://arxiv.org/abs/1904.00160) [cs.CL] |
|           | (or **arXiv:1904.00160v1 [cs.CL]** for this version)         |



<h2 id="2019-04-02-2">2. Multimodal Machine Translation with Embedding Prediction</h2> 

Title: [Multimodal Machine Translation with Embedding Prediction](<https://arxiv.org/abs/1904.00639>)

Authors: [Tosho Hirasawa](https://arxiv.org/search/cs?searchtype=author&query=Hirasawa%2C+T), [Hayahide Yamagishi](https://arxiv.org/search/cs?searchtype=author&query=Yamagishi%2C+H), [Yukio Matsumura](https://arxiv.org/search/cs?searchtype=author&query=Matsumura%2C+Y), [Mamoru Komachi](https://arxiv.org/search/cs?searchtype=author&query=Komachi%2C+M)

*(Submitted on 1 Apr 2019)*

> Multimodal machine translation is an attractive application of neural machine translation (NMT). It helps computers to deeply understand visual objects and their relations with natural languages. However, multimodal NMT systems suffer from a shortage of available training data, resulting in poor performance for translating rare words. In NMT, pretrained word embeddings have been shown to improve NMT of low-resource domains, and a search-based approach is proposed to address the rare word problem. In this study, we effectively combine these two approaches in the context of multimodal NMT and explore how we can take full advantage of pretrained word embeddings to better translate rare words. We report overall performance improvements of 1.24 METEOR and 2.49 BLEU and achieve an improvement of 7.67 F-score for rare word translation.

| Comments: | 6 pages; NAACL 2019 Student Research Workshop                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.00639](https://arxiv.org/abs/1904.00639) [cs.CL] |
|           | (or **arXiv:1904.00639v1 [cs.CL]** for this version)         |



<h2 id="2019-04-02-3">3. Lost in Interpretation: Predicting Untranslated Terminology in Simultaneous Interpretation</h2> 

Title: [Lost in Interpretation: Predicting Untranslated Terminology in Simultaneous Interpretation](<https://arxiv.org/abs/1904.00930>)

Authors: [Nikolai Vogler](https://arxiv.org/search/cs?searchtype=author&query=Vogler%2C+N), [Craig Stewart](https://arxiv.org/search/cs?searchtype=author&query=Stewart%2C+C), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G)

*(Submitted on 1 Apr 2019)*

> Simultaneous interpretation, the translation of speech from one language to another in real-time, is an inherently difficult and strenuous task. One of the greatest challenges faced by interpreters is the accurate translation of difficult terminology like proper names, numbers, or other entities. Intelligent computer-assisted interpreting (CAI) tools that could analyze the spoken word and detect terms likely to be untranslated by an interpreter could reduce translation error and improve interpreter performance. In this paper, we propose a task of predicting which terminology simultaneous interpreters will leave untranslated, and examine methods that perform this task using supervised sequence taggers. We describe a number of task-specific features explicitly designed to indicate when an interpreter may struggle with translating a word. Experimental results on a newly-annotated version of the NAIST Simultaneous Translation Corpus (Shimizu et al., 2014) indicate the promise of our proposed method.

| Comments: | NAACL 2019                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1904.00930](https://arxiv.org/abs/1904.00930) [cs.CL] |
|           | (or **arXiv:1904.00930v1 [cs.CL]** for this version)         |