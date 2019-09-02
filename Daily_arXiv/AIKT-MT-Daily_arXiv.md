# Daily arXiv: Machine Translation - Sep., 2019

### Index

- [2019-09-02](#2019-09-02)
  - [1. Latent Part-of-Speech Sequences for Neural Machine Translation](#2019-09-02-1)
  - [2. Encoders Help You Disambiguate Word Senses in Neural Machine Translation](#2019-09-02-2)
* [2019-08](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-08.md)
* [2019-07](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-07.md)
* [2019-06](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-06.md)
* [2019-05](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-05.md)
* [2019-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-04.md)
* [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)



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