# Daily arXiv: Machine Translation - Jan., 2020

# Index


- [2020-02-03](#2020-02-03)

  - [1. Self-Adversarial Learning with Comparative Discrimination for Text Generation](#2020-02-03-1)
  - [2. Teaching Machines to Converse](#2020-02-03-2)
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



NULL



# 2020-02-03
[Return to Index](#Index)



<h2 id="2020-02-03-1">1. Self-Adversarial Learning with Comparative Discrimination for Text Generation</h2>

Title: [Self-Adversarial Learning with Comparative Discrimination for Text Generation](https://arxiv.org/abs/2001.11691)

Authors: [Wangchunshu Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+W), [Tao Ge](https://arxiv.org/search/cs?searchtype=author&query=Ge%2C+T), [Ke Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+K), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F), [Ming Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+M)

*(Submitted on 31 Jan 2020)*

> Conventional Generative Adversarial Networks (GANs) for text generation tend to have issues of reward sparsity and mode collapse that affect the quality and diversity of generated samples. To address the issues, we propose a novel self-adversarial learning (SAL) paradigm for improving GANs' performance in text generation. In contrast to standard GANs that use a binary classifier as its discriminator to predict whether a sample is real or generated, SAL employs a comparative discriminator which is a pairwise classifier for comparing the text quality between a pair of samples. During training, SAL rewards the generator when its currently generated sentence is found to be better than its previously generated samples. This self-improvement reward mechanism allows the model to receive credits more easily and avoid collapsing towards the limited number of real samples, which not only helps alleviate the reward sparsity issue but also reduces the risk of mode collapse. Experiments on text generation benchmark datasets show that our proposed approach substantially improves both the quality and the diversity, and yields more stable performance compared to the previous GANs for text generation.

| Comments: | to be published in ICLR 2020                                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | [arXiv:2001.11691](https://arxiv.org/abs/2001.11691) [cs.CL] |
|           | (or [arXiv:2001.11691v1](https://arxiv.org/abs/2001.11691v1) [cs.CL] for this version) |





<h2 id="2020-02-03-2">2. Teaching Machines to Converse</h2>

Title: [Teaching Machines to Converse](https://arxiv.org/abs/2001.11701)

Authors: [Jiwei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J)

*(Submitted on 31 Jan 2020)*

> The ability of a machine to communicate with humans has long been associated with the general success of AI. This dates back to Alan Turing's epoch-making work in the early 1950s, which proposes that a machine's intelligence can be tested by how well it, the machine, can fool a human into believing that the machine is a human through dialogue conversations. Many systems learn generation rules from a minimal set of authored rules or labels on top of hand-coded rules or templates, and thus are both expensive and difficult to extend to open-domain scenarios. Recently, the emergence of neural network models the potential to solve many of the problems in dialogue learning that earlier systems cannot tackle: the end-to-end neural frameworks offer the promise of scalability and language-independence, together with the ability to track the dialogue state and then mapping between states and dialogue actions in a way not possible with conventional systems. On the other hand, neural systems bring about new challenges: they tend to output dull and generic responses; they lack a consistent or a coherent persona; they are usually optimized through single-turn conversations and are incapable of handling the long-term success of a conversation; and they are not able to take the advantage of the interactions with humans. This dissertation attempts to tackle these challenges: Contributions are two-fold: (1) we address new challenges presented by neural network models in open-domain dialogue generation systems; (2) we develop interactive question-answering dialogue systems by (a) giving the agent the ability to ask questions and (b) training a conversation agent through interactions with humans in an online fashion, where a bot improves through communicating with humans and learning from the mistakes that it makes.

| Comments: | phd thesis                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:2001.11701](https://arxiv.org/abs/2001.11701) [cs.CL] |
|           | (or [arXiv:2001.11701v1](https://arxiv.org/abs/2001.11701v1) [cs.CL] for this version) |


