# Daily arXiv: Machine Translation - September, 2020

# Index


- [2020-09-08](#2020-09-08)

  - [1. Measuring Massive Multitask Language Understanding](#2020-09-08-1)
  - [2. Recent Trends in the Use of Deep Learning Models for Grammar Error Handling](#2020-09-08-2)
  - [3. Bio-inspired Structure Identification in Language Embeddings](#2020-09-08-3)
  - [4. Why Not Simply Translate? A First Swedish Evaluation Benchmark for Semantic Similarity](#2020-09-08-4)
- [2020-09-07](#2020-09-07)
- [1. Data Readiness for Natural Language Processing](#2020-09-07-1)
  - [2. Dynamic Context-guided Capsule Network for Multimodal Machine Translation](#2020-09-07-2)
  - [3. AutoTrans: Automating Transformer Design via Reinforced Architecture Search](#2020-09-07-3)
  - [4. Going Beyond T-SNE: Exposing \texttt{whatlies} in Text Embeddings](#2020-09-07-4)
- [2020-09-01](#2020-09-01)

  - [1. Knowledge Efficient Deep Learning for Natural Language Processing](#2020-09-01-1)
- [2020-08](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-08.md)
- [2020-07](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-07.md)
- [2020-06](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-06.md)
- [2020-05](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-05.md)
- [2020-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-04.md)
- [2020-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-03.md)
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



# 2020-09-08

[Return to Index](#Index)



<h2 id="2020-09-08-1">1. Measuring Massive Multitask Language Understanding</h2>

Title: [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)

Authors: [Dan Hendrycks](https://arxiv.org/search/cs?searchtype=author&query=Hendrycks%2C+D), [Collin Burns](https://arxiv.org/search/cs?searchtype=author&query=Burns%2C+C), [Steven Basart](https://arxiv.org/search/cs?searchtype=author&query=Basart%2C+S), [Andy Zou](https://arxiv.org/search/cs?searchtype=author&query=Zou%2C+A), [Mantas Mazeika](https://arxiv.org/search/cs?searchtype=author&query=Mazeika%2C+M), [Dawn Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+D), [Jacob Steinhardt](https://arxiv.org/search/cs?searchtype=author&query=Steinhardt%2C+J)

> We propose a new test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability. We find that while most recent models have near random-chance accuracy, the very largest GPT-3 model improves over random chance by almost 20 percentage points on average. However, on every one of the 57 tasks, the best models still need substantial improvements before they can reach human-level accuracy. Models also have lopsided performance and frequently do not know when they are wrong. Worse, they still have near-random accuracy on some socially important subjects such as morality and law. By comprehensively evaluating the breadth and depth of a model's academic and professional understanding, our test can be used to analyze models across many tasks and to identify important shortcomings.

| Comments: | The test and code is available at [this https URL](https://github.com/hendrycks/test) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computers and Society (cs.CY)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2009.03300](https://arxiv.org/abs/2009.03300) [cs.CY]** |
|           | (or **[arXiv:2009.03300v1](https://arxiv.org/abs/2009.03300v1) [cs.CY]** for this version) |





<h2 id="2020-09-08-2">2. Recent Trends in the Use of Deep Learning Models for Grammar Error Handling</h2>

Title: [Recent Trends in the Use of Deep Learning Models for Grammar Error Handling](https://arxiv.org/abs/2009.02358)

Authors: [Mina Naghshnejad](https://arxiv.org/search/cs?searchtype=author&query=Naghshnejad%2C+M), [Tarun Joshi](https://arxiv.org/search/cs?searchtype=author&query=Joshi%2C+T), [Vijayan N. Nair](https://arxiv.org/search/cs?searchtype=author&query=Nair%2C+V+N)

> Grammar error handling (GEH) is an important topic in natural language processing (NLP). GEH includes both grammar error detection and grammar error correction. Recent advances in computation systems have promoted the use of deep learning (DL) models for NLP problems such as GEH. In this survey we focus on two main DL approaches for GEH: neural machine translation models and editor models. We describe the three main stages of the pipeline for these models: data preparation, training, and inference. Additionally, we discuss different techniques to improve the performance of these models at each stage of the pipeline. We compare the performance of different models and conclude with proposed future directions.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2009.02358](https://arxiv.org/abs/2009.02358) [cs.CL]** |
|           | (or **[arXiv:2009.02358v1](https://arxiv.org/abs/2009.02358v1) [cs.CL]** for this version) |





<h2 id="2020-09-08-3">3. Bio-inspired Structure Identification in Language Embeddings</h2>

Title: [Bio-inspired Structure Identification in Language Embeddings](https://arxiv.org/abs/2009.02459)

Authors: [Hongwei](https://arxiv.org/search/cs?searchtype=author&query=Hongwei) (Henry)[Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou), [Oskar Elek](https://arxiv.org/search/cs?searchtype=author&query=Elek%2C+O), [Pranav Anand](https://arxiv.org/search/cs?searchtype=author&query=Anand%2C+P), [Angus G. Forbes](https://arxiv.org/search/cs?searchtype=author&query=Forbes%2C+A+G)

> Word embeddings are a popular way to improve downstream per-formances in contemporary language modeling. However, the un-derlying geometric structure of the embedding space is not wellunderstood. We present a series of explorations using bio-inspiredmethodology to traverse and visualize word embeddings, demon-strating evidence of discernible structure. Moreover, our modelalso produces word similarity rankings that are plausible yet verydifferent from common similarity metrics, mainly cosine similarityand Euclidean distance. We show that our bio-inspired model canbe used to investigate how different word embedding techniquesresult in different semantic outputs, which can emphasize or obscureparticular interpretations in textual data.

| Comments: | 7 pages, 8 figures, 2 tables, Visualisation for the Digital Humanities 2020 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Human-Computer Interaction (cs.HC) |
| Cite as:  | **[arXiv:2009.02459](https://arxiv.org/abs/2009.02459) [cs.CL]** |
|           | (or **[arXiv:2009.02459v1](https://arxiv.org/abs/2009.02459v1) [cs.CL]** for this version) |





<h2 id="2020-09-08-4">4. Why Not Simply Translate? A First Swedish Evaluation Benchmark for Semantic Similarity</h2>

Title: [Why Not Simply Translate? A First Swedish Evaluation Benchmark for Semantic Similarity](https://arxiv.org/abs/2009.03116)

Authors: [Tim Isbister](https://arxiv.org/search/cs?searchtype=author&query=Isbister%2C+T), [Magnus Sahlgren](https://arxiv.org/search/cs?searchtype=author&query=Sahlgren%2C+M)

> This paper presents the first Swedish evaluation benchmark for textual semantic similarity. The benchmark is compiled by simply running the English STS-B dataset through the Google machine translation API. This paper discusses potential problems with using such a simple approach to compile a Swedish evaluation benchmark, including translation errors, vocabulary variation, and productive compounding. Despite some obvious problems with the resulting dataset, we use the benchmark to compare the majority of the currently existing Swedish text representations, demonstrating that native models outperform multilingual ones, and that simple bag of words performs remarkably well.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2009.03116](https://arxiv.org/abs/2009.03116) [cs.CL]** |
|           | (or **[arXiv:2009.03116v1](https://arxiv.org/abs/2009.03116v1) [cs.CL]** for this version) |





# 2020-09-07

[Return to Index](#Index)



<h2 id="2020-09-07-1">1. Data Readiness for Natural Language Processing</h2>

Title: [Data Readiness for Natural Language Processing](https://arxiv.org/abs/2009.02043)

Authors: [Fredrik Olsson](https://arxiv.org/search/cs?searchtype=author&query=Olsson%2C+F), [Magnus Sahlgren](https://arxiv.org/search/cs?searchtype=author&query=Sahlgren%2C+M)

> This document concerns data readiness in the context of machine learning and Natural Language Processing. It describes how an organization may proceed to identify, make available, validate, and prepare data to facilitate automated analysis methods. The contents of the document is based on the practical challenges and frequently asked questions we have encountered in our work as an applied research institute with helping organizations and companies, both in the public and private sectors, to use data in their business processes.

| Subjects: | **Computers and Society (cs.CY)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Databases (cs.DB); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2009.02043](https://arxiv.org/abs/2009.02043) [cs.CY]** |
|           | (or **[arXiv:2009.02043v1](https://arxiv.org/abs/2009.02043v1) [cs.CY]** for this version) |





<h2 id="2020-09-07-2">2. Dynamic Context-guided Capsule Network for Multimodal Machine Translation</h2>

Title: [Dynamic Context-guided Capsule Network for Multimodal Machine Translation](https://arxiv.org/abs/2009.02016)

Authors: [Huan Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+H), [Fandong Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+F), [Jinsong Su](https://arxiv.org/search/cs?searchtype=author&query=Su%2C+J), [Yongjing Yin](https://arxiv.org/search/cs?searchtype=author&query=Yin%2C+Y), [Zhengyuan Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Yubin Ge](https://arxiv.org/search/cs?searchtype=author&query=Ge%2C+Y), [Jie Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J), [Jiebo Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+J)

> Multimodal machine translation (MMT), which mainly focuses on enhancing text-only translation with visual features, has attracted considerable attention from both computer vision and natural language processing communities. Most current MMT models resort to attention mechanism, global context modeling or multimodal joint representation learning to utilize visual features. However, the attention mechanism lacks sufficient semantic interactions between modalities while the other two provide fixed visual context, which is unsuitable for modeling the observed variability when generating translation. To address the above issues, in this paper, we propose a novel Dynamic Context-guided Capsule Network (DCCN) for MMT. Specifically, at each timestep of decoding, we first employ the conventional source-target attention to produce a timestep-specific source-side context vector. Next, DCCN takes this vector as input and uses it to guide the iterative extraction of related visual features via a context-guided dynamic routing mechanism. Particularly, we represent the input image with global and regional visual features, we introduce two parallel DCCNs to model multimodal context vectors with visual features at different granularities. Finally, we obtain two multimodal context vectors, which are fused and incorporated into the decoder for the prediction of the target word. Experimental results on the Multi30K dataset of English-to-German and English-to-French translation demonstrate the superiority of DCCN. Our code is available on [this https URL](https://github.com/DeepLearnXMU/MM-DCCN).

| Subjects: | **Computation and Language (cs.CL)**; Multimedia (cs.MM)     |
| --------- | ------------------------------------------------------------ |
| DOI:      | [10.1145/3394171.3413715](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1145%2F3394171.3413715&v=711e661a) |
| Cite as:  | **[arXiv:2009.02016](https://arxiv.org/abs/2009.02016) [cs.CL]** |
|           | (or **[arXiv:2009.02016v1](https://arxiv.org/abs/2009.02016v1) [cs.CL]** for this version) |





<h2 id="2020-09-07-3">3. AutoTrans: Automating Transformer Design via Reinforced Architecture Search</h2>

Title: [AutoTrans: Automating Transformer Design via Reinforced Architecture Search](https://arxiv.org/abs/2009.02070)

Authors: [Wei Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+W), [Xiaoling Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Xipeng Qiu](https://arxiv.org/search/cs?searchtype=author&query=Qiu%2C+X), [Yuan Ni](https://arxiv.org/search/cs?searchtype=author&query=Ni%2C+Y), [Guotong Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+G)

> Though the transformer architectures have shown dominance in many natural language understanding tasks, there are still unsolved issues for the training of transformer models, especially the need for a principled way of warm-up which has shown importance for stable training of a transformer, as well as whether the task at hand prefer to scale the attention product or not. In this paper, we empirically explore automating the design choices in the transformer model, i.e., how to set layer-norm, whether to scale, number of layers, number of heads, activation function, etc, so that one can obtain a transformer architecture that better suits the tasks at hand. RL is employed to navigate along search space, and special parameter sharing strategies are designed to accelerate the search. It is shown that sampling a proportion of training data per epoch during search help to improve the search quality. Experiments on the CoNLL03, Multi-30k, IWSLT14 and WMT-14 shows that the searched transformer model can outperform the standard transformers. In particular, we show that our learned model can be trained more robustly with large learning rates without warm-up.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2009.02070](https://arxiv.org/abs/2009.02070) [cs.CL]** |
|           | (or **[arXiv:2009.02070v1](https://arxiv.org/abs/2009.02070v1) [cs.CL]** for this version) |





<h2 id="2020-09-07-4">4. Going Beyond T-SNE: Exposing \texttt{whatlies} in Text Embeddings</h2>

Title: [Going Beyond T-SNE: Exposing \texttt{whatlies} in Text Embeddings](https://arxiv.org/abs/2009.02113)

Authors: [Vincent D. Warmerdam](https://arxiv.org/search/cs?searchtype=author&query=Warmerdam%2C+V+D), [Thomas Kober](https://arxiv.org/search/cs?searchtype=author&query=Kober%2C+T), [Rachael Tatman](https://arxiv.org/search/cs?searchtype=author&query=Tatman%2C+R)

> We introduce whatlies, an open source toolkit for visually inspecting word and sentence embeddings. The project offers a unified and extensible API with current support for a range of popular embedding backends including spaCy, tfhub, huggingface transformers, gensim, fastText and BytePair embeddings. The package combines a domain specific language for vector arithmetic with visualisation tools that make exploring word embeddings more intuitive and concise. It offers support for many popular dimensionality reduction techniques as well as many interactive visualisations that can either be statically exported or shared via Jupyter notebooks. The project documentation is available from [this https URL](https://rasahq.github.io/whatlies/).

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2009.02113](https://arxiv.org/abs/2009.02113) [cs.CL]** |
|           | (or **[arXiv:2009.02113v1](https://arxiv.org/abs/2009.02113v1) [cs.CL]** for this version) |



# 2020-09-01

[Return to Index](#Index)



<h2 id="2020-09-01-1">1. Knowledge Efficient Deep Learning for Natural Language Processing</h2>

Title: [Knowledge Efficient Deep Learning for Natural Language Processing](https://arxiv.org/abs/2008.12878)

Authors: [Hai Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+H)

> Deep learning has become the workhorse for a wide range of natural language processing applications. But much of the success of deep learning relies on annotated examples. Annotation is time-consuming and expensive to produce at scale. Here we are interested in methods for reducing the required quantity of annotated data -- by making the learning methods more knowledge efficient so as to make them more applicable in low annotation (low resource) settings. There are various classical approaches to making the models more knowledge efficient such as multi-task learning, transfer learning, weakly supervised and unsupervised learning etc. This thesis focuses on adapting such classical methods to modern deep learning models and algorithms.
> This thesis describes four works aimed at making machine learning models more knowledge efficient. First, we propose a knowledge rich deep learning model (KRDL) as a unifying learning framework for incorporating prior knowledge into deep models. In particular, we apply KRDL built on Markov logic networks to denoise weak supervision. Second, we apply a KRDL model to assist the machine reading models to find the correct evidence sentences that can support their decision. Third, we investigate the knowledge transfer techniques in multilingual setting, where we proposed a method that can improve pre-trained multilingual BERT based on the bilingual dictionary. Fourth, we present an episodic memory network for language modelling, in which we encode the large external knowledge for the pre-trained GPT.

| Comments: | Ph.D thesis                                                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2008.12878](https://arxiv.org/abs/2008.12878) [cs.CL]** |
|           | (or **[arXiv:2008.12878v1](https://arxiv.org/abs/2008.12878v1) [cs.CL]** for this version) |

