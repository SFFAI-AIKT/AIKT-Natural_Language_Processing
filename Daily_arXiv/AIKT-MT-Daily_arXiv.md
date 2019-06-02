# Daily arXiv: Machine Translation - May, 2019

### Index

- [2019-05-31](#2019-05-31)
  - [1. Latent Space Secrets of Denoising Text-Autoencoders](#2019-05-31-1)
  - [2. Choosing Transfer Languages for Cross-Lingual Learning](#2019-05-31-2)
  - [3. Unbabel's Submission to the WMT2019 APE Shared Task: BERT-based Encoder-Decoder for Automatic Post-Editing](#2019-05-31-3)
- [2019-05-29](#2019-05-29)
  - [1. On Measuring Gender Bias in Translation of Gender-neutral Pronouns](#2019-05-29-1)
  - [2. Revisiting Low-Resource Neural Machine Translation: A Case Study](#2019-05-29-2)
- [2019-05-28](#2019-05-28)
  - [1. DebiasingWord Embeddings Improves Multimodal Machine Translation](#2019-05-28-1)
  - [2. Soft Contextual Data Augmentation for Neural Machine Translation](#2019-05-28-2)
  - [3. Levenshtein Transformer](#2019-05-28-3)
- [2019-05-27](#2019-05-27)
  - [1. Training language GANs from Scratch](#2019-05-27-1)
  - [2. An Analysis of Source-Side Grammatical Errors in NMT](#2019-05-27-2)
  - [3. mu-Forcing: Training Variational Recurrent Autoencoders for Text Generation](#2019-05-27-3)
- [2019-05-24](#2019-05-24)
  - [1. Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](#2019-05-24-1)
  - [2. Misspelling Oblivious Word Embeddings](#2019-05-24-2)
- [2019-05-23](#2019-05-23)
  - [1. Corpus Augmentation by Sentence Segmentation for Low-Resource Neural Machine Translation](#2019-05-23-1)
- [2019-05-22](#2019-05-22)
  - [1. CNNs found to jump around more skillfully than RNNs: Compositional generalization in seq2seq convolutional networks](#2019-05-22-1)
- [2019-05-21](#2019-05-21)
  - [1. Target Conditioned Sampling: Optimizing Data Selection for Multilingual Neural Machine Translation](#2019-05-21-1)
- [2019-05-20](#2019-05-20)
  - [1. Adaptation of Deep Bidirectional Multilingual Transformers for Russian Language](#2019-05-20-1)
  - [2. Learning Cross-lingual Embeddings from Twitter via Distant Supervision](#2019-05-20-2)
- [2019-05-17](#2019-05-17)
  - [1. Joint Source-Target Self Attention with Locality Constraints](#2019-05-17-1)
  - [2. Towards Interlingua Neural Machine Translation](#2019-05-17-2)
- [2019-05-16](#2019-05-16)
  - [1. Curriculum Learning for Domain Adaptation in Neural Machine Translation](#2019-05-16-1)
  - [2. When a Good Translation is Wrong in Context: Context-Aware Machine Translation Improves on Deixis, Ellipsis, and Lexical Cohesion](#2019-05-16-2)
  - [3. What do you learn from context? Probing for sentence structure in contextualized word representations](#2019-05-16-3)
- [2019-05-15](#2019-05-15)
  - [1. Effective Cross-lingual Transfer of Neural Machine Translation Models without Shared Vocabularies](#2019-05-15-1)
  - [2. Deep Residual Output Layers for Neural Language Generation](#2019-05-15-2)
  - [3. Is Word Segmentation Necessary for Deep Learning of Chinese Representations?](#2019-05-15-3)
  - [4. Sparse Sequence-to-Sequence Models](#2019-05-15-4)
- [2019-05-14](#2019-05-14)
  - [1. Synchronous Bidirectional Neural Machine Translation](#2019-05-14-1)
- [2019-05-13](#2019-05-13)
  - [1. Language Modeling with Deep Transformers](#2019-05-13-1)
  - [2. Densifying Assumed-sparse Tensors: Improving Memory Efficiency and MPI Collective Performance during Tensor Accumulation for Parallelized Training of Neural Machine Translation Models](#2019-05-13-2)
- [2019-05-09](#2019-05-09)
  - [1. Syntax-Enhanced Neural Machine Translation with Syntax-Aware Word Representations](#2019-05-09-1)
  - [2. Unified Language Model Pre-training for Natural Language Understanding and Generation](#2019-05-09-2)
- [2019-05-08](#2019-05-08-1)
  - [1. English-Bhojpuri SMT System: Insights from the Karaka Model](#2019-05-08-1)
  - [2. MASS: Masked Sequence to Sequence Pre-training for Language Generation](#2019-05-08-2)
- [2019-05-07](#2019-05-07)
  - [1. BVS Corpus: A Multilingual Parallel Corpus of Biomedical Scientific Texts](#2019-05-07-1)
  - [2. A Parallel Corpus of Theses and Dissertations Abstracts](#2019-05-07-2)
  - [3. A Large Parallel Corpus of Full-Text Scientific Articles](#2019-05-07-3)
  - [4. UFRGS Participation on the WMT Biomedical Translation Shared Task](#2019-05-07-4)
  - [5. TextKD-GAN: Text Generation using KnowledgeDistillation and Generative Adversarial Networks](#2019-05-07-5)

* [2019-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-04.md)
* [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)
* [2019-02](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-02.md)



# 2019-05-31

[Return to Index](#Index)
<h2 id="2019-05-31-1">1. Latent Space Secrets of Denoising Text-Autoencoders</h2>

Title: [Latent Space Secrets of Denoising Text-Autoencoders](https://arxiv.org/abs/1905.12777)

Authors: [Tianxiao Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+T), [Jonas Mueller](https://arxiv.org/search/cs?searchtype=author&query=Mueller%2C+J), [Regina Barzilay](https://arxiv.org/search/cs?searchtype=author&query=Barzilay%2C+R), [Tommi Jaakkola](https://arxiv.org/search/cs?searchtype=author&query=Jaakkola%2C+T)

*(Submitted on 29 May 2019)*

> While neural language models have recently demonstrated impressive performance in unconditional text generation, controllable generation and manipulation of text remain challenging. Latent variable generative models provide a natural approach for control, but their application to text has proven more difficult than to images. Models such as variational autoencoders may suffer from posterior collapse or learning an irregular latent geometry. We propose to instead employ adversarial autoencoders (AAEs) and add local perturbations by randomly replacing/removing words from input sentences during training. Within the prior enforced by the adversary, structured perturbations in the data space begin to carve and organize the latent space. Theoretically, we prove that perturbations encourage similar sentences to map to similar latent representations. Experimentally, we investigate the trade-off between text-generation and autoencoder-reconstruction capabilities. Our straightforward approach significantly improves over regular AAEs as well as other autoencoders, and enables altering the tense/sentiment of sentences through simple addition of a fixed vector offset to their latent representation.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1905.12777 [cs.LG]**                                 |
|           | (or **arXiv:1905.12777v1 [cs.LG]** for this version)         |

<h2 id="2019-05-31-2">2. Choosing Transfer Languages for Cross-Lingual Learning</h2>

Title: [Choosing Transfer Languages for Cross-Lingual Learning](https://arxiv.org/abs/1905.12688)

Authors: [Yu-Hsiang Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+Y), [Chian-Yu Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+C), [Jean Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+J), [Zirui Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Yuyan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Mengzhou Xia](https://arxiv.org/search/cs?searchtype=author&query=Xia%2C+M), [Shruti Rijhwani](https://arxiv.org/search/cs?searchtype=author&query=Rijhwani%2C+S), [Junxian He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+J), [Zhisong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Xuezhe Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+X), [Antonios Anastasopoulos](https://arxiv.org/search/cs?searchtype=author&query=Anastasopoulos%2C+A), [Patrick Littell](https://arxiv.org/search/cs?searchtype=author&query=Littell%2C+P), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G)

*(Submitted on 29 May 2019)*

> Cross-lingual transfer, where a high-resource transfer language is used to improve the accuracy of a low-resource task language, is now an invaluable tool for improving performance of natural language processing (NLP) on low-resource languages. However, given a particular task language, it is not clear which language to transfer from, and the standard strategy is to select languages based on ad hoc criteria, usually the intuition of the experimenter. Since a large number of features contribute to the success of cross-lingual transfer (including phylogenetic similarity, typological properties, lexical overlap, or size of available data), even the most enlightened experimenter rarely considers all these factors for the particular task at hand. In this paper, we consider this task of automatically selecting optimal transfer languages as a ranking problem, and build models that consider the aforementioned features to perform this prediction. In experiments on representative NLP tasks, we demonstrate that our model predicts good transfer languages much better than ad hoc baselines considering single features in isolation, and glean insights on what features are most informative for each different NLP tasks, which may inform future ad hoc selection even without use of our method. Code, data, and pre-trained models are available at [this https URL](https://github.com/neulab/langrank)

| Comments: | Proceedings of ACL 2019                              |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1905.12688 [cs.CL]**                         |
|           | (or **arXiv:1905.12688v1 [cs.CL]** for this version) |

<h2 id="2019-05-31-3">3. Unbabel's Submission to the WMT2019 APE Shared Task: BERT-based Encoder-Decoder for Automatic Post-Editing</h2>

Title: [Unbabel's Submission to the WMT2019 APE Shared Task: BERT-based Encoder-Decoder for Automatic Post-Editing](https://arxiv.org/abs/1905.13068)

Authors: [António V. Lopes](https://arxiv.org/search/cs?searchtype=author&query=Lopes%2C+A+V), [M. Amin Farajian](https://arxiv.org/search/cs?searchtype=author&query=Farajian%2C+M+A), [Gonçalo M. Correia](https://arxiv.org/search/cs?searchtype=author&query=Correia%2C+G+M), [Jonay Trenous](https://arxiv.org/search/cs?searchtype=author&query=Trenous%2C+J), [André F. T. Martins](https://arxiv.org/search/cs?searchtype=author&query=Martins%2C+A+F+T)

*(Submitted on 30 May 2019)*

> This paper describes Unbabel's submission to the WMT2019 APE Shared Task for the English-German language pair. Following the recent rise of large, powerful, pre-trained models, we adapt the BERT pretrained model to perform Automatic Post-Editing in an encoder-decoder framework. Analogously to dual-encoder architectures we develop a BERT-based encoder-decoder (BED) model in which a single pretrained BERT encoder receives both the source src and machine translation tgt strings. Furthermore, we explore a conservativeness factor to constrain the APE system to perform fewer edits. As the official results show, when trained on a weighted combination of in-domain and artificial training data, our BED system with the conservativeness penalty improves significantly the translations of a strong Neural Machine Translation system by −0.78 and +1.23 in terms of TER and BLEU, respectively.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1905.13068 [cs.CL]**                         |
|           | (or **arXiv:1905.13068v1 [cs.CL]** for this version) |


# 2019-05-29

[Return to Index](#Index)

<h2 id="2019-05-29-1">1. 
On Measuring Gender Bias in Translation of Gender-neutral Pronouns</h2>
Title: [On Measuring Gender Bias in Translation of Gender-neutral Pronouns](https://arxiv.org/abs/1905.11684)

Authors: [Won Ik Cho](https://arxiv.org/search/cs?searchtype=author&query=Cho%2C+W+I), [Ji Won Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+J+W), [Seok Min Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+S+M), [Nam Soo Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+N+S)

*(Submitted on 28 May 2019)*

> Ethics regarding social bias has recently thrown striking issues in natural language processing. Especially for gender-related topics, the need for a system that reduces the model bias has grown in areas such as image captioning, content recommendation, and automated employment. However, detection and evaluation of gender bias in the machine translation systems are not yet thoroughly investigated, for the task being cross-lingual and challenging to define. In this paper, we propose a scheme for making up a test set that evaluates the gender bias in a machine translation system, with Korean, a language with gender-neutral pronouns. Three word/phrase sets are primarily constructed, each incorporating positive/negative expressions or occupations; all the terms are gender-independent or at least not biased to one side severely. Then, additional sentence lists are constructed concerning formality of the pronouns and politeness of the sentences. With the generated sentence set of size 4,236 in total, we evaluate gender bias in conventional machine translation systems utilizing the proposed measure, which is termed here as translation gender bias index (TGBI). The corpus and the code for evaluation is available on-line.

| Comments: | Accepted to 1st ACL Workshop on Gender Bias for Natural Language Processing (GeBNLP 2019) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1905.11684 [cs.CL]**                                 |
|           | (or **arXiv:1905.11684v1 [cs.CL]** for this version)         |



<h2 id="2019-05-29-2">2. Revisiting Low-Resource Neural Machine Translation: A Case Study</h2>

Title: [Revisiting Low-Resource Neural Machine Translation: A Case Study](https://arxiv.org/abs/1905.11901)

Authors: [Rico Sennrich](https://arxiv.org/search/cs?searchtype=author&query=Sennrich%2C+R), [Biao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+B)

*(Submitted on 28 May 2019)*

> It has been shown that the performance of neural machine translation (NMT) drops starkly in low-resource conditions, underperforming phrase-based statistical machine translation (PBSMT) and requiring large amounts of auxiliary data to achieve competitive results. In this paper, we re-assess the validity of these results, arguing that they are the result of lack of system adaptation to low-resource settings. We discuss some pitfalls to be aware of when training low-resource NMT systems, and recent techniques that have shown to be especially helpful in low-resource settings, resulting in a set of best practices for low-resource NMT. In our experiments on German--English with different amounts of IWSLT14 training data, we show that, without the use of any auxiliary monolingual or multilingual data, an optimized NMT system can outperform PBSMT with far less data than previously claimed. We also apply these techniques to a low-resource Korean-English dataset, surpassing previously reported results by 4 BLEU.

| Comments: | to appear at ACL 2019                                |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1905.11901 [cs.CL]**                         |
|           | (or **arXiv:1905.11901v1 [cs.CL]** for this version) |



# 2019-05-28
[Return to Index](#Index)

<h2 id="2019-05-28-1">1. DebiasingWord Embeddings Improves Multimodal Machine Translation</h2>

Title: [DebiasingWord Embeddings Improves Multimodal Machine Translation](https://arxiv.org/abs/1905.10464)

Authors: [Tosho Hirasawa](https://arxiv.org/search/cs?searchtype=author&query=Hirasawa%2C+T), [Mamoru Komachi](https://arxiv.org/search/cs?searchtype=author&query=Komachi%2C+M)

*(Submitted on 24 May 2019)*

> In recent years, pretrained word embeddings have proved useful for multimodal neural machine translation (NMT) models to address the shortage of available datasets. However, the integration of pretrained word embeddings has not yet been explored extensively. Further, pretrained word embeddings in high dimensional spaces have been reported to suffer from the hubness problem. Although some debiasing techniques have been proposed to address this problem for other natural language processing tasks, they have seldom been studied for multimodal NMT models. In this study, we examine various kinds of word embeddings and introduce two debiasing techniques for three multimodal NMT models and two language pairs -- English-German translation and English-French translation. With our optimal settings, the overall performance of multimodal models was improved by up to +1.93 BLEU and +2.02 METEOR for English-German translation and +1.73 BLEU and +0.95 METEOR for English-French translation.

| Comments: | 10 pages; MT Summit 2019                             |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1905.10464 [cs.CL]**                         |
|           | (or **arXiv:1905.10464v1 [cs.CL]** for this version) |

<h2 id="2019-05-28-2">2. 
Soft Contextual Data Augmentation for Neural Machine Translation</h2>

Title: [Soft Contextual Data Augmentation for Neural Machine Translation](https://arxiv.org/abs/1905.10523)

Authors: [Jinhua Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+J), [Fei Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+F), [Lijun Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+L), [Yingce Xia](https://arxiv.org/search/cs?searchtype=author&query=Xia%2C+Y), [Tao Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+T), [Wengang Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+W), [Xueqi Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng%2C+X), [Tie-Yan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+T)

*(Submitted on 25 May 2019)*

> While data augmentation is an important trick to boost the accuracy of deep learning methods in computer vision tasks, its study in natural language tasks is still very limited. In this paper, we present a novel data augmentation method for neural machine translation. Different from previous augmentation methods that randomly drop, swap or replace words with other words in a sentence, we softly augment a randomly chosen word in a sentence by its contextual mixture of multiple related words. More accurately, we replace the one-hot representation of a word by a distribution (provided by a language model) over the vocabulary, i.e., replacing the embedding of this word by a weighted combination of multiple semantically similar words. Since the weights of those words depend on the contextual information of the word to be replaced, the newly generated sentences capture much richer information than previous augmentation methods. Experimental results on both small scale and large scale machine translation datasets demonstrate the superiority of our method over strong baselines.

| Comments: | Accepted by ACL 2019 as short paper                  |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1905.10523 [cs.CL]**                         |
|           | (or **arXiv:1905.10523v1 [cs.CL]** for this version) |

<h2 id="2019-05-28-3">3. Levenshtein Transformer</h2>

Title: [Levenshtein Transformer](https://arxiv.org/abs/1905.11006)

Authors: [Jiatao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+J), [Changhan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Jake Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+J)

*(Submitted on 27 May 2019)*

> Modern neural sequence generation models are built to either generate tokens step-by-step from scratch or (iteratively) modify a sequence of tokens bounded by a fixed length. In this work, we develop Levenshtein Transformer, a new partially autoregressive model devised for more flexible and amenable sequence generation. Unlike previous approaches, the atomic operations of our model are insertion and deletion. The combination of them facilitates not only generation but also sequence refinement allowing dynamic length changes. We also propose a set of new training techniques dedicated at them, effectively exploiting one as the other's learning signal thanks to their complementary nature. Experiments applying the proposed model achieve comparable performance but much-improved efficiency on both generation (e.g. machine translation, text summarization) and refinement tasks (e.g. automatic post-editing). We further confirm the flexibility of our model by showing a Levenshtein Transformer trained by machine translation can straightforwardly be used for automatic post-editing.

| Comments: | 16 pages (6 pages appendix). Work in progress                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1905.11006 [cs.CL]**                                 |
|           | (or **arXiv:1905.11006v1 [cs.CL]** for this version)         |




# 2019-05-27
[Return to Index](#Index)

<h2 id="2019-05-27-1">1. Training language GANs from Scratch</h2>

Title: [Training language GANs from Scratch](https://arxiv.org/abs/1905.09922)

Authors: [Cyprien de Masson d'Autume](https://arxiv.org/search/cs?searchtype=author&query=de+Masson+d'Autume%2C+C), [Mihaela Rosca](https://arxiv.org/search/cs?searchtype=author&query=Rosca%2C+M), [Jack Rae](https://arxiv.org/search/cs?searchtype=author&query=Rae%2C+J), [Shakir Mohamed](https://arxiv.org/search/cs?searchtype=author&query=Mohamed%2C+S)

*(Submitted on 23 May 2019)*

> Generative Adversarial Networks (GANs) enjoy great success at image generation, but have proven difficult to train in the domain of natural language. Challenges with gradient estimation, optimization instability, and mode collapse have lead practitioners to resort to maximum likelihood pre-training, followed by small amounts of adversarial fine-tuning. The benefits of GAN fine-tuning for language generation are unclear, as the resulting models produce comparable or worse samples than traditional language models. We show it is in fact possible to train a language GAN from scratch -- without maximum likelihood pre-training. We combine existing techniques such as large batch sizes, dense rewards and discriminator regularization to stabilize and improve language GANs. The resulting model, ScratchGAN, performs comparably to maximum likelihood training on EMNLP2017 News and WikiText-103 corpora according to quality and diversity metrics.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1905.09922 [cs.CL]**                                 |
|           | (or **arXiv:1905.09922v1 [cs.CL]** for this version)         |

<h2 id="2019-05-27-2">2. An Analysis of Source-Side Grammatical Errors in NMT</h2>

Title: [An Analysis of Source-Side Grammatical Errors in NMT](https://arxiv.org/abs/1905.10024)

Authors: [Antonios Anastasopoulos](https://arxiv.org/search/cs?searchtype=author&query=Anastasopoulos%2C+A)

*(Submitted on 24 May 2019)*

> The quality of Neural Machine Translation (NMT) has been shown to significantly degrade when confronted with source-side noise. We present the first large-scale study of state-of-the-art English-to-German NMT on real grammatical noise, by evaluating on several Grammar Correction corpora. We present methods for evaluating NMT robustness without true references, and we use them for extensive analysis of the effects that different grammatical errors have on the NMT output. We also introduce a technique for visualizing the divergence distribution caused by a source-side error, which allows for additional insights.

| Comments: | Accepted and to be presented at BlackboxNLP 2019     |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1905.10024 [cs.CL]**                         |
|           | (or **arXiv:1905.10024v1 [cs.CL]** for this version) |

<h2 id="2019-05-27-3">3. 
mu-Forcing: Training Variational Recurrent Autoencoders for Text Generation</h2>

Title: [mu-Forcing: Training Variational Recurrent Autoencoders for Text Generation](https://arxiv.org/abs/1905.10072)

Authors: [Dayiheng Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+D), [Xu Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+X), [Feng He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+F), [Yuanyuan Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Jiancheng Lv](https://arxiv.org/search/cs?searchtype=author&query=Lv%2C+J)

*(Submitted on 24 May 2019)*

> It has been previously observed that training Variational Recurrent Autoencoders (VRAE) for text generation suffers from serious uninformative latent variables problem. The model would collapse into a plain language model that totally ignore the latent variables and can only generate repeating and dull samples. In this paper, we explore the reason behind this issue and propose an effective regularizer based approach to address it. The proposed method directly injects extra constraints on the posteriors of latent variables into the learning process of VRAE, which can flexibly and stably control the trade-off between the KL term and the reconstruction term, making the model learn dense and meaningful latent representations. The experimental results show that the proposed method outperforms several strong baselines and can make the model learn interpretable latent variables and generate diverse meaningful sentences. Furthermore, the proposed method can perform well without using other strategies, such as KL annealing.

| Comments: | To appear in the ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1905.10072 [cs.CL]**                                 |
|           | (or **arXiv:1905.10072v1 [cs.CL]** for this version)         |


# 2019-05-24
[Return to Index](#Index)

<h2 id="2019-05-24-1">1. Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned</h2>
Title: [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://arxiv.org/abs/1905.09418)
Authors: [Elena Voita](https://arxiv.org/search/cs?searchtype=author&query=Voita%2C+E), [David Talbot](https://arxiv.org/search/cs?searchtype=author&query=Talbot%2C+D), [Fedor Moiseev](https://arxiv.org/search/cs?searchtype=author&query=Moiseev%2C+F), [Rico Sennrich](https://arxiv.org/search/cs?searchtype=author&query=Sennrich%2C+R), [Ivan Titov](https://arxiv.org/search/cs?searchtype=author&query=Titov%2C+I)

*(Submitted on 23 May 2019)*

> Multi-head self-attention is a key component of the Transformer, a state-of-the-art architecture for neural machine translation. In this work we evaluate the contribution made by individual attention heads in the encoder to the overall performance of the model and analyze the roles played by them. We find that the most important and confident heads play consistent and often linguistically-interpretable roles. When pruning heads using a method based on stochastic gates and a differentiable relaxation of the L0 penalty, we observe that specialized heads are last to be pruned. Our novel pruning method removes the vast majority of heads without seriously affecting performance. For example, on the English-Russian WMT dataset, pruning 38 out of 48 encoder heads results in a drop of only 0.15 BLEU.

| Comments: | To appear at ACL 2019                                |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1905.09418 [cs.CL]**                         |
|           | (or **arXiv:1905.09418v1 [cs.CL]** for this version) |


<h2 id="2019-05-24-2">2. Misspelling Oblivious Word Embeddings</h2>
Title: [Misspelling Oblivious Word Embeddings](https://arxiv.org/abs/1905.09755)

Authors: [Bora Edizel](https://arxiv.org/search/cs?searchtype=author&query=Edizel%2C+B), [Aleksandra Piktus](https://arxiv.org/search/cs?searchtype=author&query=Piktus%2C+A), [Piotr Bojanowski](https://arxiv.org/search/cs?searchtype=author&query=Bojanowski%2C+P), [Rui Ferreira](https://arxiv.org/search/cs?searchtype=author&query=Ferreira%2C+R), [Edouard Grave](https://arxiv.org/search/cs?searchtype=author&query=Grave%2C+E), [Fabrizio Silvestri](https://arxiv.org/search/cs?searchtype=author&query=Silvestri%2C+F)

*(Submitted on 23 May 2019)*

> In this paper we present a method to learn word embeddings that are resilient to misspellings. Existing word embeddings have limited applicability to malformed texts, which contain a non-negligible amount of out-of-vocabulary words. We propose a method combining FastText with subwords and a supervised task of learning misspelling patterns. In our method, misspellings of each word are embedded close to their correct variants. We train these embeddings on a new dataset we are releasing publicly. Finally, we experimentally show the advantages of this approach on both intrinsic and extrinsic NLP tasks using public test sets.

| Comments: | 9 Pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1905.09755 [cs.CL]**                                 |
|           | (or **arXiv:1905.09755v1 [cs.CL]** for this version)         |



# 2019-05-23
[Return to Index](#Index)

<h2 id="2019-05-23-1">1. Corpus Augmentation by Sentence Segmentation for Low-Resource Neural Machine Translation</h2>

Title: [Corpus Augmentation by Sentence Segmentation for Low-Resource Neural Machine Translation](https://arxiv.org/abs/1905.08945)

Authors: [Jinyi Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+J), [Tadahiro Matsumoto](https://arxiv.org/search/cs?searchtype=author&query=Matsumoto%2C+T)

*(Submitted on 22 May 2019)*

> Neural Machine Translation (NMT) has been proven to achieve impressive results. The NMT system translation results depend strongly on the size and quality of parallel corpora. Nevertheless, for many language pairs, no rich-resource parallel corpora exist. As described in this paper, we propose a corpus augmentation method by segmenting long sentences in a corpus using back-translation and generating pseudo-parallel sentence pairs. The experiment results of the Japanese-Chinese and Chinese-Japanese translation with Japanese-Chinese scientific paper excerpt corpus (ASPEC-JC) show that the method improves translation performance.

| Comments: | 4 pages. The version before Applied. Sciences                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| DOI:      | [10.3390/app9102036](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.3390%2Fapp9102036&v=8deb72f8) |
| Cite as:  | **arXiv:1905.08945 [cs.CL]**                                 |
|           | (or **arXiv:1905.08945v1 [cs.CL]** for this version)         |




# 2019-05-22

[Return to Index](#Index)

<h2 id="2019-05-22-1">1. CNNs found to jump around more skillfully than RNNs: Compositional generalization in seq2seq convolutional networks</h2>

Title: [CNNs found to jump around more skillfully than RNNs: Compositional generalization in seq2seq convolutional networks](https://arxiv.org/abs/1905.08527)
Authors: [Roberto Dessì](https://arxiv.org/search/cs?searchtype=author&query=Dessì%2C+R), [Marco Baroni](https://arxiv.org/search/cs?searchtype=author&query=Baroni%2C+M)

*(Submitted on 21 May 2019)*

> Lake and Baroni (2018) introduced the SCAN dataset probing the ability of seq2seq models to capture compositional generalizations, such as inferring the meaning of "jump around" 0-shot from the component words. Recurrent networks (RNNs) were found to completely fail the most challenging generalization cases. We test here a convolutional network (CNN) on these tasks, reporting hugely improved performance with respect to RNNs. Despite the big improvement, the CNN has however not induced systematic rules, suggesting that the difference between compositional and non-compositional behaviour is not clear-cut.

| Comments: | accepted as a short paper at ACL 2019                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1905.08527 [cs.CL]**                                 |
|           | (or **arXiv:1905.08527v1 [cs.CL]** for this version)         |





# 2019-05-21

[Return to Index](#Index)

<h2 id="2019-05-21-1">1. Target Conditioned Sampling: Optimizing Data Selection for Multilingual Neural Machine Translation</h2>

Title: [Target Conditioned Sampling: Optimizing Data Selection for Multilingual Neural Machine Translation](https://arxiv.org/abs/1905.08212)

Authors: [Xinyi Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G)

*(Submitted on 20 May 2019)*

> To improve low-resource Neural Machine Translation (NMT) with multilingual corpora, training on the most related high-resource language only is often more effective than using all data available (Neubig and Hu, 2018). However, it is possible that an intelligent data selection strategy can further improve low-resource NMT with data from other auxiliary languages. In this paper, we seek to construct a sampling distribution over all multilingual data, so that it minimizes the training loss of the low-resource language. Based on this formulation, we propose an efficient algorithm, Target Conditioned Sampling (TCS), which first samples a target sentence, and then conditionally samples its source sentence. Experiments show that TCS brings significant gains of up to 2 BLEU on three of four languages we test, with minimal training overhead.

| Comments: | Accepted at ACL 2019                                 |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1905.08212 [cs.CL]**                         |
|           | (or **arXiv:1905.08212v1 [cs.CL]** for this version) |



# 2019-05-20

[Return to Index](#Index)

<h2 id="2019-05-20-1">1. Adaptation of Deep Bidirectional Multilingual Transformers for Russian Language</h2>

Title: [Adaptation of Deep Bidirectional Multilingual Transformers for Russian Language](https://arxiv.org/abs/1905.07213)

Authors: [Yuri Kuratov](https://arxiv.org/search/cs?searchtype=author&query=Kuratov%2C+Y), [Mikhail Arkhipov](https://arxiv.org/search/cs?searchtype=author&query=Arkhipov%2C+M)

*(Submitted on 17 May 2019)*

> The paper introduces methods of adaptation of multilingual masked language models for a specific language. Pre-trained bidirectional language models show state-of-the-art performance on a wide range of tasks including reading comprehension, natural language inference, and sentiment analysis. At the moment there are two alternative approaches to train such models: monolingual and multilingual. While language specific models show superior performance, multilingual models allow to perform a transfer from one language to another and solve tasks for different languages simultaneously. This work shows that transfer learning from a multilingual model to monolingual model results in significant growth of performance on such tasks as reading comprehension, paraphrase detection, and sentiment analysis. Furthermore, multilingual initialization of monolingual model substantially reduces training time. Pre-trained models for the Russian language are open sourced.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1905.07213 [cs.CL]**                         |
|           | (or **arXiv:1905.07213v1 [cs.CL]** for this version) |

<h2 id="2019-05-20-2">2. Learning Cross-lingual Embeddings from Twitter via Distant Supervision</h2>

Title: [Learning Cross-lingual Embeddings from Twitter via Distant Supervision](https://arxiv.org/abs/1905.07358)

Authors: [Jose Camacho-Collados](https://arxiv.org/search/cs?searchtype=author&query=Camacho-Collados%2C+J), [Yerai Doval](https://arxiv.org/search/cs?searchtype=author&query=Doval%2C+Y), [Eugenio Martínez-Cámara](https://arxiv.org/search/cs?searchtype=author&query=Martínez-Cámara%2C+E), [Luis Espinosa-Anke](https://arxiv.org/search/cs?searchtype=author&query=Espinosa-Anke%2C+L), [Francesco Barbieri](https://arxiv.org/search/cs?searchtype=author&query=Barbieri%2C+F), [Steven Schockaert](https://arxiv.org/search/cs?searchtype=author&query=Schockaert%2C+S)

*(Submitted on 17 May 2019)*

> Cross-lingual embeddings represent the meaning of words from different languages in the same vector space. Recent work has shown that it is possible to construct such representations by aligning independently learned monolingual embedding spaces, and that accurate alignments can be obtained even without external bilingual data. In this paper we explore a research direction which has been surprisingly neglected in the literature: leveraging noisy user-generated text to learn cross-lingual embeddings particularly tailored towards social media applications. While the noisiness and informal nature of the social media genre poses additional challenges to cross-lingual embedding methods, we find that it also provides key opportunities due to the abundance of code-switching and the existence of a shared vocabulary of emoji and named entities. Our contribution consists in a very simple post-processing step that exploits these phenomena to significantly improve the performance of state-of-the-art alignment methods.

| Comments: | 11 pages, 5 tables, 1 figure, 1 appendix                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Social and Information Networks (cs.SI) |
| Cite as:  | **arXiv:1905.07358 [cs.CL]**                                 |
|           | (or **arXiv:1905.07358v1 [cs.CL]** for this version)         |



# 2019-05-17

[Return to Index](#Index)

<h2 id="2019-05-17-1">1. Joint Source-Target Self Attention with Locality Constraints</h2>

Title: [Joint Source-Target Self Attention with Locality Constraints](https://arxiv.org/abs/1905.06596)

Authors: [José A. R. Fonollosa](https://arxiv.org/search/cs?searchtype=author&query=Fonollosa%2C+J+A+R), [Noe Casas](https://arxiv.org/search/cs?searchtype=author&query=Casas%2C+N), [Marta R. Costa-jussà](https://arxiv.org/search/cs?searchtype=author&query=Costa-jussà%2C+M+R)

*(Submitted on 16 May 2019)*

> The dominant neural machine translation models are based on the encoder-decoder structure, and many of them rely on an unconstrained receptive field over source and target sequences. In this paper we study a new architecture that breaks with both conventions. Our simplified architecture consists in the decoder part of a transformer model, based on self-attention, but with locality constraints applied on the attention receptive field. As input for training, both source and target sentences are fed to the network, which is trained as a language model. At inference time, the target tokens are predicted autoregressively starting with the source sequence as previous tokens. The proposed model achieves a new state of the art of 35.7 BLEU on IWSLT'14 German-English and matches the best reported results in the literature on the WMT'14 English-German and WMT'14 English-French translation benchmarks.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **arXiv:1905.06596 [cs.CL]**                                 |
|           | (or **arXiv:1905.06596v1 [cs.CL]** for this version)         |



<h2 id="2019-05-17-2">2. Towards Interlingua Neural Machine Translation</h2>

Title: [Towards Interlingua Neural Machine Translation](https://arxiv.org/abs/1905.06831)

Authors: [Carlos Escolano](https://arxiv.org/search/cs?searchtype=author&query=Escolano%2C+C), [Marta R. Costa-jussà](https://arxiv.org/search/cs?searchtype=author&query=Costa-jussà%2C+M+R), [José A. R. Fonollosa](https://arxiv.org/search/cs?searchtype=author&query=Fonollosa%2C+J+A+R)

*(Submitted on 15 May 2019)*

> A common intermediate language representation or an interlingua is the holy grail in machine translation. Thanks to the new neural machine translation approach, it seems that there are good perspectives towards this goal. In this paper, we propose a new architecture based on introducing an interlingua loss as an additional training objective. By adding and forcing this interlingua loss, we are able to train multiple encoders and decoders for each language, sharing a common intermediate representation. 
> Preliminary translation results on the WMT Turkish/English and WMT 2019 Kazakh/English tasks show improvements over the baseline system. Additionally, since the final objective of our architecture is having compatible encoder/decoders based on a common representation, we visualize and evaluate the learned intermediate representations. What is most relevant from our study is that our architecture shows the benefits of the dreamed interlingua since it is capable of: (1) reducing the number of production systems, with respect to the number of languages, from quadratic to linear (2) incrementally adding a new language in the system without retraining languages previously there and (3) allowing for translations from the new language to all the others present in the system

| Comments: | arXiv admin note: substantial text overlap with [arXiv:1810.06351](https://arxiv.org/abs/1810.06351) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1905.06831 [cs.CL]**                                 |
|           | (or **arXiv:1905.06831v1 [cs.CL]** for this version)         |



# 2019-05-16

[Return to Index](#Index)

<h2 id="2019-05-16-1">1. Curriculum Learning for Domain Adaptation in Neural Machine Translation</h2>

Title: [Curriculum Learning for Domain Adaptation in Neural Machine Translation](https://arxiv.org/abs/1905.05816)

Authors: [Xuan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+X), [Pamela Shapiro](https://arxiv.org/search/cs?searchtype=author&query=Shapiro%2C+P), [Gaurav Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+G), [Paul McNamee](https://arxiv.org/search/cs?searchtype=author&query=McNamee%2C+P), [Marine Carpuat](https://arxiv.org/search/cs?searchtype=author&query=Carpuat%2C+M), [Kevin Duh](https://arxiv.org/search/cs?searchtype=author&query=Duh%2C+K)

*(Submitted on 14 May 2019)*

> We introduce a curriculum learning approach to adapt generic neural machine translation models to a specific domain. Samples are grouped by their similarities to the domain of interest and each group is fed to the training algorithm with a particular schedule. This approach is simple to implement on top of any neural framework or architecture, and consistently outperforms both unadapted and adapted baselines in experiments with two distinct domains and two language pairs.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1905.05816 [cs.CL]**                         |
|           | (or **arXiv:1905.05816v1 [cs.CL]** for this version) |



<h2 id="2019-05-16-2">2. When a Good Translation is Wrong in Context: Context-Aware Machine Translation Improves on Deixis, Ellipsis, and Lexical Cohesion</h2>

Title: [When a Good Translation is Wrong in Context: Context-Aware Machine Translation Improves on Deixis, Ellipsis, and Lexical Cohesion](https://arxiv.org/abs/1905.05979)

Authors: [Elena Voita](https://arxiv.org/search/cs?searchtype=author&query=Voita%2C+E), [Rico Sennrich](https://arxiv.org/search/cs?searchtype=author&query=Sennrich%2C+R), [Ivan Titov](https://arxiv.org/search/cs?searchtype=author&query=Titov%2C+I)

*(Submitted on 15 May 2019)*

> Though machine translation errors caused by the lack of context beyond one sentence have long been acknowledged, the development of context-aware NMT systems is hampered by several problems. Firstly, standard metrics are not sensitive to improvements in consistency in document-level translations. Secondly, previous work on context-aware NMT assumed that the sentence-aligned parallel data consisted of complete documents while in most practical scenarios such document-level data constitutes only a fraction of the available parallel data. To address the first issue, we perform a human study on an English-Russian subtitles dataset and identify deixis, ellipsis and lexical cohesion as three main sources of inconsistency. We then create test sets targeting these phenomena. To address the second shortcoming, we consider a set-up in which a much larger amount of sentence-level data is available compared to that aligned at the document level. We introduce a model that is suitable for this scenario and demonstrate major gains over a context-agnostic baseline on our new benchmarks without sacrificing performance as measured with BLEU.

| Comments: | To appear at ACL 2019                                |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1905.05979 [cs.CL]**                         |
|           | (or **arXiv:1905.05979v1 [cs.CL]** for this version) |



<h2 id="2019-05-16-3">3. What do you learn from context? Probing for sentence structure in contextualized word representations</h2>

Title: [What do you learn from context? Probing for sentence structure in contextualized word representations](https://arxiv.org/abs/1905.06316)

Authors: [Ian Tenney](https://arxiv.org/search/cs?searchtype=author&query=Tenney%2C+I), [Patrick Xia](https://arxiv.org/search/cs?searchtype=author&query=Xia%2C+P), [Berlin Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+B), [Alex Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+A), [Adam Poliak](https://arxiv.org/search/cs?searchtype=author&query=Poliak%2C+A), [R Thomas McCoy](https://arxiv.org/search/cs?searchtype=author&query=McCoy%2C+R+T), [Najoung Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+N), [Benjamin Van Durme](https://arxiv.org/search/cs?searchtype=author&query=Van+Durme%2C+B), [Samuel R. Bowman](https://arxiv.org/search/cs?searchtype=author&query=Bowman%2C+S+R), [Dipanjan Das](https://arxiv.org/search/cs?searchtype=author&query=Das%2C+D), [Ellie Pavlick](https://arxiv.org/search/cs?searchtype=author&query=Pavlick%2C+E)

*(Submitted on 15 May 2019)*

> Contextualized representation models such as ELMo (Peters et al., 2018a) and BERT (Devlin et al., 2018) have recently achieved state-of-the-art results on a diverse array of downstream NLP tasks. Building on recent token-level probing work, we introduce a novel edge probing task design and construct a broad suite of sub-sentence tasks derived from the traditional structured NLP pipeline. We probe word-level contextual representations from four recent models and investigate how they encode sentence structure across a range of syntactic, semantic, local, and long-range phenomena. We find that existing models trained on language modeling and translation produce strong representations for syntactic phenomena, but only offer comparably small improvements on semantic tasks over a non-contextual baseline.

| Comments: | ICLR 2019 camera-ready version, 17 pages including appendices |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1905.06316 [cs.CL]**                                 |
|           | (or **arXiv:1905.06316v1 [cs.CL]** for this version)         |



# 2019-05-15

[Return to Index](#Index)

<h2 id="2019-05-15-1">1. Effective Cross-lingual Transfer of Neural Machine Translation Models without Shared Vocabularies</h2>

Title: [Effective Cross-lingual Transfer of Neural Machine Translation Models without Shared Vocabularies](https://arxiv.org/abs/1905.05475)

Authors: [Yunsu Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+Y), [Yingbo Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+Y), [Hermann Ney](https://arxiv.org/search/cs?searchtype=author&query=Ney%2C+H)

*(Submitted on 14 May 2019)*

> Transfer learning or multilingual model is essential for low-resource neural machine translation (NMT), but the applicability is limited to cognate languages by sharing their vocabularies. This paper shows effective techniques to transfer a pre-trained NMT model to a new, unrelated language without shared vocabularies. We relieve the vocabulary mismatch by using cross-lingual word embedding, train a more language-agnostic encoder by injecting artificial noises, and generate synthetic data easily from the pre-training data without back-translation. Our methods do not require restructuring the vocabulary or retraining the model. We improve plain NMT transfer by up to +5.1% BLEU in five low-resource translation tasks, outperforming multilingual joint training by a large margin. We also provide extensive ablation studies on pre-trained embedding, synthetic data, vocabulary size, and parameter freezing for a better understanding of NMT transfer.

| Comments: | Will appear in ACL 2019                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1905.05475 [cs.CL]**                                 |
|           | (or **arXiv:1905.05475v1 [cs.CL]** for this version)         |



<h2 id="2019-05-15-2">2. Deep Residual Output Layers for Neural Language Generation</h2>

Title: [Deep Residual Output Layers for Neural Language Generation](https://arxiv.org/abs/1905.05513)

Authors: [Nikolaos Pappas](https://arxiv.org/search/cs?searchtype=author&query=Pappas%2C+N), [James Henderson](https://arxiv.org/search/cs?searchtype=author&query=Henderson%2C+J)

*(Submitted on 14 May 2019)*

> Many tasks, including language generation, benefit from learning the structure of the output space, particularly when the space of output labels is large and the data is sparse. State-of-the-art neural language models indirectly capture the output space structure in their classifier weights since they lack parameter sharing across output labels. Learning shared output label mappings helps, but existing methods have limited expressivity and are prone to overfitting. In this paper, we investigate the usefulness of more powerful shared mappings for output labels, and propose a deep residual output mapping with dropout between layers to better capture the structure of the output space and avoid overfitting. Evaluations on three language generation tasks show that our output label mapping can match or improve state-of-the-art recurrent and self-attention architectures, and suggest that the classifier does not necessarily need to be high-rank to better model natural language if it is better at capturing the structure of the output space.

| Comments: | To appear in ICML 2019                               |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1905.05513 [cs.CL]**                         |
|           | (or **arXiv:1905.05513v1 [cs.CL]** for this version) |



<h2 id="2019-05-15-3">3. 
Is Word Segmentation Necessary for Deep Learning of Chinese Representations
?</h2>

Title: [Is Word Segmentation Necessary for Deep Learning of Chinese Representations?](https://arxiv.org/abs/1905.05526)

Authors: [Yuxian Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+Y), [Xiaoya Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+X), [Xiaofei Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+X), [Qinghong Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+Q), [Arianna Yuan](https://arxiv.org/search/cs?searchtype=author&query=Yuan%2C+A), [Jiwei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J)

*(Submitted on 14 May 2019)*

> Segmenting a chunk of text into words is usually the first step of processing Chinese text, but its necessity has rarely been explored. In this paper, we ask the fundamental question of whether Chinese word segmentation (CWS) is necessary for deep learning-based Chinese Natural Language Processing. We benchmark neural word-based models which rely on word segmentation against neural char-based models which do not involve word segmentation in four end-to-end NLP benchmark tasks: language modeling, machine translation, sentence matching/paraphrase and text classification. Through direct comparisons between these two types of models, we find that char-based models consistently outperform word-based models. Based on these observations, we conduct comprehensive experiments to study why word-based models underperform char-based models in these deep learning-based NLP tasks. We show that it is because word-based models are more vulnerable to data sparsity and the presence of out-of-vocabulary (OOV) words, and thus more prone to overfitting. We hope this paper could encourage researchers in the community to rethink the necessity of word segmentation in deep learning-based Chinese Natural Language Processing. \footnote{Yuxian Meng and Xiaoya Li contributed equally to this paper.}

| Comments: | to appear at ACL2019                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **arXiv:1905.05526 [cs.CL]**                                 |
|           | (or **arXiv:1905.05526v1 [cs.CL]** for this version)         |



<h2 id="2019-05-15-4">4. Sparse Sequence-to-Sequence Models</h2>

Title: [Sparse Sequence-to-Sequence Models](https://arxiv.org/abs/1905.05702)

Authors: [Ben Peters](https://arxiv.org/search/cs?searchtype=author&query=Peters%2C+B), [Vlad Niculae](https://arxiv.org/search/cs?searchtype=author&query=Niculae%2C+V), [André F.T. Martins](https://arxiv.org/search/cs?searchtype=author&query=Martins%2C+A+F)

*(Submitted on 14 May 2019)*

> Sequence-to-sequence models are a powerful workhorse of NLP. Most variants employ a softmax transformation in both their attention mechanism and output layer, leading to dense alignments and strictly positive output probabilities. This density is wasteful, making models less interpretable and assigning probability mass to many implausible outputs. In this paper, we propose sparse sequence-to-sequence models, rooted in a new family of α-entmax transformations, which includes softmax and sparsemax as particular cases, and is sparse for any α>1. We provide fast algorithms to evaluate these transformations and their gradients, which scale well for large vocabulary sizes. Our models are able to produce sparse alignments and to assign nonzero probability to a short list of plausible outputs, sometimes rendering beam search exact. Experiments on morphological inflection and machine translation reveal consistent gains over dense models.

| Comments: | Accepted to ACL 2019                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1905.05702 [cs.CL]**                                 |
|           | (or **arXiv:1905.05702v1 [cs.CL]** for this version)         |



# 2019-05-14

[Return to Index](#Index)

<h2 id="2019-05-14-1">1. Synchronous Bidirectional Neural Machine Translation</h2>

Title: [Synchronous Bidirectional Neural Machine Translation](https://arxiv.org/abs/1905.04847)

Authors: [Long Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+L), [Jiajun Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+J), [Chengqing Zong](https://arxiv.org/search/cs?searchtype=author&query=Zong%2C+C)

*(Submitted on 13 May 2019)*

> Existing approaches to neural machine translation (NMT) generate the target language sequence token by token from left to right. However, this kind of unidirectional decoding framework cannot make full use of the target-side future contexts which can be produced in a right-to-left decoding direction, and thus suffers from the issue of unbalanced outputs. In this paper, we introduce a synchronous bidirectional neural machine translation (SB-NMT) that predicts its outputs using left-to-right and right-to-left decoding simultaneously and interactively, in order to leverage both of the history and future information at the same time. Specifically, we first propose a new algorithm that enables synchronous bidirectional decoding in a single model. Then, we present an interactive decoding model in which left-to-right (right-to-left) generation does not only depend on its previously generated outputs, but also relies on future contexts predicted by right-to-left (left-to-right) decoding. We extensively evaluate the proposed SB-NMT model on large-scale NIST Chinese-English, WMT14 English-German, and WMT18 Russian-English translation tasks. Experimental results demonstrate that our model achieves significant improvements over the strong Transformer model by 3.92, 1.49 and 1.04 BLEU points respectively, and obtains the state-of-the-art performance on Chinese-English and English-German translation tasks.

| Comments: | Published by TACL 2019, 15 pages, 9 figures, 9 tabels        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1905.04847 [cs.CL]**                                 |
|           | (or **arXiv:1905.04847v1 [cs.CL]** for this version)         |



# 2019-05-13

[Return to Index](#Index)

<h2 id="2019-05-13-1">1. Language Modeling with Deep Transformers</h2>

Title: [Language Modeling with Deep Transformers](https://arxiv.org/abs/1905.04226)

Authors: Language Modeling with Deep Transformers

[Kazuki Irie](https://arxiv.org/search/cs?searchtype=author&query=Irie%2C+K), [Albert Zeyer](https://arxiv.org/search/cs?searchtype=author&query=Zeyer%2C+A), [Ralf Schlüter](https://arxiv.org/search/cs?searchtype=author&query=Schlüter%2C+R), [Hermann Ney](https://arxiv.org/search/cs?searchtype=author&query=Ney%2C+H)

*(Submitted on 10 May 2019)*

> We explore multi-layer autoregressive Transformer models in language modeling for speech recognition. We focus on two aspects. First, we revisit Transformer model configurations specifically for language modeling. We show that well configured Transformer models outperform our baseline models based on the shallow stack of LSTM recurrent neural network layers. We carry out experiments on the open-source LibriSpeech 960hr task, for both 200K vocabulary word-level and 10K byte-pair encoding subword-level language modeling. We apply our word-level models to conventional hybrid speech recognition by lattice rescoring, and the subword-level models to attention based encoder-decoder models by shallow fusion. Second, we show that deep Transformer language models do not require positional encoding. The positional encoding is an essential augmentation for the self-attention mechanism which is invariant to sequence ordering. However, in autoregressive setup, as is the case for language modeling, the amount of information increases along the position dimension, which is a positional signal by its own. The analysis of attention weights shows that deep autoregressive self-attention models can automatically make use of such positional information. We find that removing the positional encoding even slightly improves the performance of these models.

| Comments: | Submitted to INTERSPEECH 2019                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1905.04226 [cs.CL]**                                 |
|           | (or **arXiv:1905.04226v1 [cs.CL]** for this version)         |



<h2 id="2019-05-13-2">2. Densifying Assumed-sparse Tensors: Improving Memory Efficiency and MPI Collective Performance during Tensor Accumulation for Parallelized Training of Neural Machine Translation Models</h2>

Title: [Densifying Assumed-sparse Tensors: Improving Memory Efficiency and MPI Collective Performance during Tensor Accumulation for Parallelized Training of Neural Machine Translation Models](https://arxiv.org/abs/1905.04035)

Authors: [Derya Cavdar](https://arxiv.org/search/cs?searchtype=author&query=Cavdar%2C+D), [Valeriu Codreanu](https://arxiv.org/search/cs?searchtype=author&query=Codreanu%2C+V), [Can Karakus](https://arxiv.org/search/cs?searchtype=author&query=Karakus%2C+C), [John A. Lockman III](https://arxiv.org/search/cs?searchtype=author&query=III%2C+J+A+L), [Damian Podareanu](https://arxiv.org/search/cs?searchtype=author&query=Podareanu%2C+D), [Vikram Saletore](https://arxiv.org/search/cs?searchtype=author&query=Saletore%2C+V), [Alexander Sergeev](https://arxiv.org/search/cs?searchtype=author&query=Sergeev%2C+A), [Don D. Smith II](https://arxiv.org/search/cs?searchtype=author&query=II%2C+D+D+S), [Victor Suthichai](https://arxiv.org/search/cs?searchtype=author&query=Suthichai%2C+V), [Quy Ta](https://arxiv.org/search/cs?searchtype=author&query=Ta%2C+Q), [Srinivas Varadharajan](https://arxiv.org/search/cs?searchtype=author&query=Varadharajan%2C+S), [Lucas A. Wilson](https://arxiv.org/search/cs?searchtype=author&query=Wilson%2C+L+A), [Rengan Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+R), [Pei Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+P)

*(Submitted on 10 May 2019)*

> Neural machine translation - using neural networks to translate human language - is an area of active research exploring new neuron types and network topologies with the goal of dramatically improving machine translation performance. Current state-of-the-art approaches, such as the multi-head attention-based transformer, require very large translation corpuses and many epochs to produce models of reasonable quality. Recent attempts to parallelize the official TensorFlow "Transformer" model across multiple nodes have hit roadblocks due to excessive memory use and resulting out of memory errors when performing MPI collectives. This paper describes modifications made to the Horovod MPI-based distributed training framework to reduce memory usage for transformer models by converting assumed-sparse tensors to dense tensors, and subsequently replacing sparse gradient gather with dense gradient reduction. The result is a dramatic increase in scale-out capability, with CPU-only scaling tests achieving 91% weak scaling efficiency up to 1200 MPI processes (300 nodes), and up to 65% strong scaling efficiency up to 400 MPI processes (200 nodes) using the Stampede2 supercomputer.

| Comments: | 18 pages, 10 figures, accepted at the 2019 International Supercomputing Conference |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Distributed, Parallel, and Cluster Computing (cs.DC) |
| Cite as:  | **arXiv:1905.04035 [cs.LG]**                                 |
|           | (or **arXiv:1905.04035v1 [cs.LG]** for this version)         |



# 2019-05-09

[Return to Index](#Index)

<h2 id="2019-05-09-1">1. Syntax-Enhanced Neural Machine Translation with Syntax-Aware Word Representations</h2>

Title: [Syntax-Enhanced Neural Machine Translation with Syntax-Aware Word Representations](https://arxiv.org/abs/1905.02878)

Authors: [Meishan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M), [Zhenghua Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Guohong Fu](https://arxiv.org/search/cs?searchtype=author&query=Fu%2C+G), [Min Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M)

*(Submitted on 8 May 2019)*

> Syntax has been demonstrated highly effective in neural machine translation (NMT). Previous NMT models integrate syntax by representing 1-best tree outputs from a well-trained parsing system, e.g., the representative Tree-RNN and Tree-Linearization methods, which may suffer from error propagation. In this work, we propose a novel method to integrate source-side syntax implicitly for NMT. The basic idea is to use the intermediate hidden representations of a well-trained end-to-end dependency parser, which are referred to as syntax-aware word representations (SAWRs). Then, we simply concatenate such SAWRs with ordinary word embeddings to enhance basic NMT models. The method can be straightforwardly integrated into the widely-used sequence-to-sequence (Seq2Seq) NMT models. We start with a representative RNN-based Seq2Seq baseline system, and test the effectiveness of our proposed method on two benchmark datasets of the Chinese-English and English-Vietnamese translation tasks, respectively. Experimental results show that the proposed approach is able to bring significant BLEU score improvements on the two datasets compared with the baseline, 1.74 points for Chinese-English translation and 0.80 point for English-Vietnamese translation, respectively. In addition, the approach also outperforms the explicit Tree-RNN and Tree-Linearization methods.

| Comments: | NAACL 2019                                           |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1905.02878 [cs.CL]**                         |
|           | (or **arXiv:1905.02878v1 [cs.CL]** for this version) |



<h2 id="2019-05-09-2">2. Unified Language Model Pre-training for Natural Language Understanding and Generation</h2>

Title: [Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)

Authors: [Li Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+L), [Nan Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+N), [Wenhui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+W), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F), [Xiaodong Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+X), [Yu Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Jianfeng Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+J), [Ming Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+M), [Hsiao-Wuen Hon](https://arxiv.org/search/cs?searchtype=author&query=Hon%2C+H)

*(Submitted on 8 May 2019)*

> This paper presents a new Unified pre-trained Language Model (UniLM) that can be fine-tuned for both natural language understanding and generation tasks. The model is pre-trained using three types of language modeling objectives: unidirectional (both left-to-right and right-to-left), bidirectional, and sequence-to-sequence prediction. The unified modeling is achieved by employing a shared Transformer network and utilizing specific self-attention masks to control what context the prediction conditions on. We can fine-tune UniLM as a unidirectional decoder, a bidirectional encoder, or a sequence-to-sequence model to support various downstream natural language understanding and generation tasks. 
> UniLM compares favorably with BERT on the GLUE benchmark, and the SQuAD 2.0 and CoQA question answering tasks. Moreover, our model achieves new state-of-the-art results on three natural language generation tasks, including improving the CNN/DailyMail abstractive summarization ROUGE-L to 40.63 (2.16 absolute improvement), pushing the CoQA generative question answering F1 score to 82.5 (37.1 absolute improvement), and the SQuAD question generation BLEU-4 to 22.88 (6.50 absolute improvement).

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1905.03197 [cs.CL]**                         |
|           | (or **arXiv:1905.03197v1 [cs.CL]** for this version) |





# 2019-05-08

[Return to Index](#Index)

<h2 id="2019-05-08-1">1. English-Bhojpuri SMT System: Insights from the Karaka Model</h2>

Title: [English-Bhojpuri SMT System: Insights from the Karaka Model](https://arxiv.org/abs/1905.02239)

Authors: [Atul Kr. Ojha](https://arxiv.org/search/cs?searchtype=author&query=Ojha%2C+A+K)

*(Submitted on 6 May 2019)*

> This thesis has been divided into six chapters namely: Introduction, Karaka Model and it impacts on Dependency Parsing, LT Resources for Bhojpuri, English-Bhojpuri SMT System: Experiment, Evaluation of EB-SMT System, and Conclusion. Chapter one introduces this PhD research by detailing the motivation of the study, the methodology used for the study and the literature review of the existing MT related work in Indian Languages. Chapter two talks of the theoretical background of Karaka and Karaka model. Along with this, it talks about previous related work. It also discusses the impacts of the Karaka model in NLP and dependency parsing. It compares Karaka dependency and Universal Dependency. It also presents a brief idea of the implementation of these models in the SMT system for English-Bhojpuri language pair.

| Comments: | 211 pages and Submitted at JNU New Delhi             |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1905.02239 [cs.CL]**                         |
|           | (or **arXiv:1905.02239v1 [cs.CL]** for this version) |



<h2 id="2019-05-08-2">2. MASS: Masked Sequence to Sequence Pre-training for Language Generation</h2>

Title: [MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/abs/1905.02450)

Authors: [Kaitao Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+K), [Xu Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+X), [Tao Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+T), [Jianfeng Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+J), [Tie-Yan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+T)

*(Submitted on 7 May 2019)*

> Pre-training and fine-tuning, e.g., BERT, have achieved great success in language understanding by transferring knowledge from rich-resource pre-training task to the low/zero-resource downstream tasks. Inspired by the success of BERT, we propose MAsked Sequence to Sequence pre-training (MASS) for the encoder-decoder based language generation tasks. MASS adopts the encoder-decoder framework to reconstruct a sentence fragment given the remaining part of the sentence: its encoder takes a sentence with randomly masked fragment (several consecutive tokens) as input, and its decoder tries to predict this masked fragment. In this way, MASS can jointly train the encoder and decoder to develop the capability of representation extraction and language modeling. By further fine-tuning on a variety of zero/low-resource language generation tasks, including neural machine translation, text summarization and conversational response generation (3 tasks and totally 8 datasets), MASS achieves significant improvements over the baselines without pre-training or with other pre-training methods. Specially, we achieve the state-of-the-art accuracy (37.5 in terms of BLEU score) on the unsupervised English-French translation, even beating the early attention-based supervised model.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1905.02450 [cs.CL]**                         |
|           | (or **arXiv:1905.02450v1 [cs.CL]** for this version) |



# 2019-05-07

[Return to Index](#Index)

<h2 id="2019-05-07-1">1. BVS Corpus: A Multilingual Parallel Corpus of Biomedical Scientific Texts</h2>

Title: [BVS Corpus: A Multilingual Parallel Corpus of Biomedical Scientific Texts](https://arxiv.org/abs/1905.01712)

Authors: [Felipe Soares](https://arxiv.org/search/cs?searchtype=author&query=Soares%2C+F), [Martin Krallinger](https://arxiv.org/search/cs?searchtype=author&query=Krallinger%2C+M)

*(Submitted on 5 May 2019)*

> The BVS database (Health Virtual Library) is a centralized source of biomedical information for Latin America and Carib, created in 1998 and coordinated by BIREME (Biblioteca Regional de Medicina) in agreement with the Pan American Health Organization (OPAS). Abstracts are available in English, Spanish, and Portuguese, with a subset in more than one language, thus being a possible source of parallel corpora. In this article, we present the development of parallel corpora from BVS in three languages: English, Portuguese, and Spanish. Sentences were automatically aligned using the Hunalign algorithm for EN/ES and EN/PT language pairs, and for a subset of trilingual articles also. We demonstrate the capabilities of our corpus by training a Neural Machine Translation (OpenNMT) system for each language pair, which outperformed related works on scientific biomedical articles. Sentence alignment was also manually evaluated, presenting an average 96% of correctly aligned sentences across all languages. Our parallel corpus is freely available, with complementary information regarding article metadata.

| Comments: | Accepted at the Copora conference. arXiv admin note: text overlap with [arXiv:1905.01715](https://arxiv.org/abs/1905.01715) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Information Retrieval (cs.IR) |
| Cite as:  | **arXiv:1905.01712 [cs.CL]**                                 |
|           | (or **arXiv:1905.01712v1 [cs.CL]** for this version)         |



<h2 id="2019-05-07-2">2. A Parallel Corpus of Theses and Dissertations Abstracts
</h2>

Title: [A Parallel Corpus of Theses and Dissertations Abstracts](https://arxiv.org/abs/1905.01715)

Authors: [Felipe Soares](https://arxiv.org/search/cs?searchtype=author&query=Soares%2C+F), [Gabrielli Harumi Yamashita](https://arxiv.org/search/cs?searchtype=author&query=Yamashita%2C+G+H), [Michel Jose Anzanello](https://arxiv.org/search/cs?searchtype=author&query=Anzanello%2C+M+J)

*(Submitted on 5 May 2019)*

> In Brazil, the governmental body responsible for overseeing and coordinating post-graduate programs, CAPES, keeps records of all theses and dissertations presented in the country. Information regarding such documents can be accessed online in the Theses and Dissertations Catalog (TDC), which contains abstracts in Portuguese and English, and additional metadata. Thus, this database can be a potential source of parallel corpora for the Portuguese and English languages. In this article, we present the development of a parallel corpus from TDC, which is made available by CAPES under the open data initiative. Approximately 240,000 documents were collected and aligned using the Hunalign tool. We demonstrate the capability of our developed corpus by training Statistical Machine Translation (SMT) and Neural Machine Translation (NMT) models for both language directions, followed by a comparison with Google Translate (GT). Both translation models presented better BLEU scores than GT, with NMT system being the most accurate one. Sentence alignment was also manually evaluated, presenting an average of 82.30% correctly aligned sentences. Our parallel corpus is freely available in TMX format, with complementary information regarding document metadata

| Comments:          | Published in the PROPOR Conference. arXiv admin note: text overlap with [arXiv:1905.01712](https://arxiv.org/abs/1905.01712) |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**                         |
| Journal reference: | Computational Processing of the Portuguese Language 2018     |
| DOI:               | [10.1007/978-3-319-99722-3_35](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1007%2F978-3-319-99722-3_35&v=71962c3d) |
| Cite as:           | **arXiv:1905.01715 [cs.CL]**                                 |
|                    | (or **arXiv:1905.01715v1 [cs.CL]** for this version)         |



<h2 id="2019-05-07-3">3. A Large Parallel Corpus of Full-Text Scientific Articles</h2>

Title: [A Large Parallel Corpus of Full-Text Scientific Articles](https://arxiv.org/abs/1905.01852)

Authors: [Felipe Soares](https://arxiv.org/search/cs?searchtype=author&query=Soares%2C+F), [Viviane Pereira Moreira](https://arxiv.org/search/cs?searchtype=author&query=Moreira%2C+V+P), [Karin Becker](https://arxiv.org/search/cs?searchtype=author&query=Becker%2C+K)

*(Submitted on 6 May 2019)*

> The Scielo database is an important source of scientific information in Latin America, containing articles from several research domains. A striking characteristic of Scielo is that many of its full-text contents are presented in more than one language, thus being a potential source of parallel corpora. In this article, we present the development of a parallel corpus from Scielo in three languages: English, Portuguese, and Spanish. Sentences were automatically aligned using the Hunalign algorithm for all language pairs, and for a subset of trilingual articles also. We demonstrate the capabilities of our corpus by training a Statistical Machine Translation system (Moses) for each language pair, which outperformed related works on scientific articles. Sentence alignment was also manually evaluated, presenting an average of 98.8% correctly aligned sentences across all languages. Our parallel corpus is freely available in the TMX format, with complementary information regarding article metadata.

| Comments: | Published in Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1905.01852 [cs.CL]**                                 |
|           | (or **arXiv:1905.01852v1 [cs.CL]** for this version)         |



<h2 id="2019-05-07-4">4. BVS Corpus: A Multilingual Parallel Corpus of Biomedical Scientific Texts</h2>

Title: [UFRGS Participation on the WMT Biomedical Translation Shared Task](https://arxiv.org/abs/1905.01855)

Authors: [Felipe Soares](https://arxiv.org/search/cs?searchtype=author&query=Soares%2C+F), [Karin Becker](https://arxiv.org/search/cs?searchtype=author&query=Becker%2C+K)

*(Submitted on 6 May 2019)*

> This paper describes the machine translation systems developed by the Universidade Federal do Rio Grande do Sul (UFRGS) team for the biomedical translation shared task. Our systems are based on statistical machine translation and neural machine translation, using the Moses and OpenNMT toolkits, respectively. We participated in four translation directions for the English/Spanish and English/Portuguese language pairs. To create our training data, we concatenated several parallel corpora, both from in-domain and out-of-domain sources, as well as terminological resources from UMLS. Our systems achieved the best BLEU scores according to the official shared task evaluation.

| Comments: | Published on the Third Conference on Machine Translation (WMT18) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1905.01855 [cs.CL]**                                 |
|           | (or **arXiv:1905.01855v1 [cs.CL]** for this version)         |



<h2 id="2019-05-07-5">5. TextKD-GAN: Text Generation using KnowledgeDistillation and Generative Adversarial Networks</h2>

Title: [TextKD-GAN: Text Generation using KnowledgeDistillation and Generative Adversarial Networks](https://arxiv.org/abs/1905.01976)

Authors: [Md. Akmal Haidar](https://arxiv.org/search/cs?searchtype=author&query=Haidar%2C+M+A), [Mehdi Rezagholizadeh](https://arxiv.org/search/cs?searchtype=author&query=Rezagholizadeh%2C+M)

*(Submitted on 23 Apr 2019)*

> Text generation is of particular interest in many NLP applications such as machine translation, language modeling, and text summarization. Generative adversarial networks (GANs) achieved a remarkable success in high quality image generation in computer vision,and recently, GANs have gained lots of interest from the NLP community as well. However, achieving similar success in NLP would be more challenging due to the discrete nature of text. In this work, we introduce a method using knowledge distillation to effectively exploit GAN setup for text generation. We demonstrate how autoencoders (AEs) can be used for providing a continuous representation of sentences, which is a smooth representation that assign non-zero probabilities to more than one word. We distill this representation to train the generator to synthesize similar smooth representations. We perform a number of experiments to validate our idea using different datasets and show that our proposed approach yields better performance in terms of the BLEU score and Jensen-Shannon distance (JSD) measure compared to traditional GAN-based text generation approaches without pre-training.

| Comments:          | arXiv admin note: text overlap with [arXiv:1904.07293](https://arxiv.org/abs/1904.07293) |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**                         |
| Journal reference: | 32nd Canadian Conference on Artificial Intelligence 2019     |
| Cite as:           | **arXiv:1905.01976 [cs.CL]**                                 |
|                    | (or **arXiv:1905.01976v1 [cs.CL]** for this version)         |