# Daily arXiv: Machine Translation - Jul., 2019

### Index

- [2019-07-01](#2019-07-01)
  - [1. Findings of the First Shared Task on Machine Translation Robustness](#2019-07-01-1)
  - [2. Lost in Translation: Loss and Decay of Linguistic Richness in Machine Translation](#2019-07-01-2)
  - [3. Widening the Representation Bottleneck in Neural Machine Translation with Lexical Shortcuts](#2019-07-01-3)

* [2019-06](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-06.md)
* [2019-05](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-05.md)
* [2019-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-04.md)
* [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)
* [2019-02](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-02.md)



# 2019-07-01

[Return to Index](#Index)

<h2 id="2019-07-01-1">1. Findings of the First Shared Task on Machine Translation Robustness</h2>
Title: [Findings of the First Shared Task on Machine Translation Robustness](https://arxiv.org/abs/1906.11943)

Authors: [Xian Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+X), [Paul Michel](https://arxiv.org/search/cs?searchtype=author&query=Michel%2C+P), [Antonios Anastasopoulos](https://arxiv.org/search/cs?searchtype=author&query=Anastasopoulos%2C+A), [Yonatan Belinkov](https://arxiv.org/search/cs?searchtype=author&query=Belinkov%2C+Y), [Nadir Durrani](https://arxiv.org/search/cs?searchtype=author&query=Durrani%2C+N), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O), [Philipp Koehn](https://arxiv.org/search/cs?searchtype=author&query=Koehn%2C+P), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G), [Juan Pino](https://arxiv.org/search/cs?searchtype=author&query=Pino%2C+J), [Hassan Sajjad](https://arxiv.org/search/cs?searchtype=author&query=Sajjad%2C+H)

*(Submitted on 27 Jun 2019)*

> We share the findings of the first shared task on improving robustness of Machine Translation (MT). The task provides a testbed representing challenges facing MT models deployed in the real world, and facilitates new approaches to improve models; robustness to noisy input and domain mismatch. We focus on two language pairs (English-French and English-Japanese), and the submitted systems are evaluated on a blind test set consisting of noisy comments on Reddit and professionally sourced translations. As a new task, we received 23 submissions by 11 participating teams from universities, companies, national labs, etc. All submitted systems achieved large improvements over baselines, with the best improvement having +22.33 BLEU. We evaluated submissions by both human judgment and automatic evaluation (BLEU), which shows high correlations (Pearson's r = 0.94 and 0.95). Furthermore, we conducted a qualitative analysis of the submitted systems using compare-mt, which revealed their salient differences in handling challenges in this task. Such analysis provides additional insights when there is occasional disagreement between human judgment and BLEU, e.g. systems better at producing colloquial expressions received higher score from human judgment.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1906.11943 [cs.CL]**                         |
|           | (or **arXiv:1906.11943v1 [cs.CL]** for this version) |

<h2 id="2019-07-01-2">2. Lost in Translation: Loss and Decay of Linguistic Richness in Machine Translation</h2>

Title: [Lost in Translation: Loss and Decay of Linguistic Richness in Machine Translation](https://arxiv.org/abs/1906.12068)

Authors: [Eva Vanmassenhove](https://arxiv.org/search/cs?searchtype=author&query=Vanmassenhove%2C+E), [Dimitar Shterionov](https://arxiv.org/search/cs?searchtype=author&query=Shterionov%2C+D), [Andy Way](https://arxiv.org/search/cs?searchtype=author&query=Way%2C+A)

*(Submitted on 28 Jun 2019)*

> This work presents an empirical approach to quantifying the loss of lexical richness in Machine Translation (MT) systems compared to Human Translation (HT). Our experiments show how current MT systems indeed fail to render the lexical diversity of human generated or translated text. The inability of MT systems to generate diverse outputs and its tendency to exacerbate already frequent patterns while ignoring less frequent ones, might be the underlying cause for, among others, the currently heavily debated issues related to gender biased output. Can we indeed, aside from biased data, talk about an algorithm that exacerbates seen biases?

| Comments: | Accepted for publication at the 17th Machine Translation Summit (MTSummit2019), Dublin, Ireland, August 2019 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1906.12068 [cs.CL]**                                 |
|           | (or **arXiv:1906.12068v1 [cs.CL]** for this version)         |

<h2 id="2019-07-01-3">3. Widening the Representation Bottleneck in Neural Machine Translation with Lexical Shortcuts</h2>

Title: [Widening the Representation Bottleneck in Neural Machine Translation with Lexical Shortcuts](https://arxiv.org/abs/1906.12284)

Authors: [Denis Emelin](https://arxiv.org/search/cs?searchtype=author&query=Emelin%2C+D), [Ivan Titov](https://arxiv.org/search/cs?searchtype=author&query=Titov%2C+I), [Rico Sennrich](https://arxiv.org/search/cs?searchtype=author&query=Sennrich%2C+R)

*(Submitted on 28 Jun 2019)*

> The transformer is a state-of-the-art neural translation model that uses attention to iteratively refine lexical representations with information drawn from the surrounding context. Lexical features are fed into the first layer and propagated through a deep network of hidden layers. We argue that the need to represent and propagate lexical features in each layer limits the model's capacity for learning and representing other information relevant to the task. To alleviate this bottleneck, we introduce gated shortcut connections between the embedding layer and each subsequent layer within the encoder and decoder. This enables the model to access relevant lexical content dynamically, without expending limited resources on storing it within intermediate states. We show that the proposed modification yields consistent improvements over a baseline transformer on standard WMT translation tasks in 5 translation directions (0.9 BLEU on average) and reduces the amount of lexical information passed along the hidden layers. We furthermore evaluate different ways to integrate lexical connections into the transformer architecture and present ablation experiments exploring the effect of proposed shortcuts on model behavior.

| Comments: | Accepted submission to WMT 2019                              |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1906.12284 [cs.CL]**                                 |
|           | (or **arXiv:1906.12284v1 [cs.CL]** for this version)         |