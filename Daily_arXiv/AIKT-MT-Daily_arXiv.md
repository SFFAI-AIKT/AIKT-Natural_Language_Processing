# Daily arXiv: Machine Translation - April, 2021

# Index

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






