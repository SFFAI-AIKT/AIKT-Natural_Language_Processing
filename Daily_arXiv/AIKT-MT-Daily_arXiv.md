# Daily arXiv: Machine Translation - Sep., 2019

# Index

- [2019-09-30](#2019-09-30)
  - [1. On the use of BERT for Neural Machine Translation](#2019-09-30-1)
  - [2. Improving Pre-Trained Multilingual Models with Vocabulary Expansion](#2019-09-30-2)
- [2019-09](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-09.md)
- [2019-08](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-08.md)
- [2019-07](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-07.md)
- [2019-06](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-06.md)
- [2019-05](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-05.md)
- [2019-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-04.md)
- [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)



# 2019-09-30

[Return to Index](#Index)



<h2 id="2019-09-30-1">1. On the use of BERT for Neural Machine Translation</h2> 
Title: [On the use of BERT for Neural Machine Translation](https://arxiv.org/abs/1909.12744)

Authors:[Stéphane Clinchant](https://arxiv.org/search/cs?searchtype=author&query=Clinchant%2C+S), [Kweon Woo Jung](https://arxiv.org/search/cs?searchtype=author&query=Jung%2C+K+W), [Vassilina Nikoulina](https://arxiv.org/search/cs?searchtype=author&query=Nikoulina%2C+V)

*(Submitted on 27 Sep 2019)*

> Exploiting large pretrained models for various NMT tasks have gained a lot of visibility recently. In this work we study how BERT pretrained models could be exploited for supervised Neural Machine Translation. We compare various ways to integrate pretrained BERT model with NMT model and study the impact of the monolingual data used for BERT training on the final translation quality. We use WMT-14 English-German, IWSLT15 English-German and IWSLT14 English-Russian datasets for these experiments. In addition to standard task test set evaluation, we perform evaluation on out-of-domain test sets and noise injected test sets, in order to assess how BERT pretrained representations affect model robustness.

| Comments: | 10 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1909.12744 [cs.CL]**                                 |
|           | (or **arXiv:1909.12744v1 [cs.CL]** for this version)         |





<h2 id="2019-09-30-2">2. Improving Pre-Trained Multilingual Models with Vocabulary Expansion</h2> 
Title: [Improving Pre-Trained Multilingual Models with Vocabulary Expansion](https://arxiv.org/abs/1909.12440)

Authors:[Hai Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+H), [Dian Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+D), [Kai Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+K), [Janshu Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+J), [Dong Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+D)

*(Submitted on 26 Sep 2019)*

> Recently, pre-trained language models have achieved remarkable success in a broad range of natural language processing tasks. However, in multilingual setting, it is extremely resource-consuming to pre-train a deep language model over large-scale corpora for each language. Instead of exhaustively pre-training monolingual language models independently, an alternative solution is to pre-train a powerful multilingual deep language model over large-scale corpora in hundreds of languages. However, the vocabulary size for each language in such a model is relatively small, especially for low-resource languages. This limitation inevitably hinders the performance of these multilingual models on tasks such as sequence labeling, wherein in-depth token-level or sentence-level understanding is essential.
> In this paper, inspired by previous methods designed for monolingual settings, we investigate two approaches (i.e., joint mapping and mixture mapping) based on a pre-trained multilingual model BERT for addressing the out-of-vocabulary (OOV) problem on a variety of tasks, including part-of-speech tagging, named entity recognition, machine translation quality estimation, and machine reading comprehension. Experimental results show that using mixture mapping is more promising. To the best of our knowledge, this is the first work that attempts to address and discuss the OOV issue in multilingual settings.

| Comments: | CONLL 2019 final version                             |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.12440 [cs.CL]**                         |
|           | (or **arXiv:1909.12440v1 [cs.CL]** for this version) |


