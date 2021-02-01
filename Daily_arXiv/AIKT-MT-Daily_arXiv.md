# Daily arXiv: Machine Translation - January, 2021

# Index


- [2021-02-01](#2021-02-01)

  - [1. Combining pre-trained language models and structured knowledge](#2021-02-01-1)
  - [2. Few-Shot Domain Adaptation for Grammatical Error Correction via Meta-Learning](#2021-02-01-2)
  - [3. Synthesizing Monolingual Data for Neural Machine Translation](#2021-02-01-3)
  - [4. Transition based Graph Decoder for Neural Machine Translation](#2021-02-01-4)
- [Other Columns](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-02-01

[Return to Index](#Index)



<h2 id="2021-02-01-1">1. Combining pre-trained language models and structured knowledge</h2>

Title: [Combining pre-trained language models and structured knowledge](https://arxiv.org/abs/2101.12294)

Authors: [Pedro Colon-Hernandez](https://arxiv.org/search/cs?searchtype=author&query=Colon-Hernandez%2C+P), [Catherine Havasi](https://arxiv.org/search/cs?searchtype=author&query=Havasi%2C+C), [Jason Alonso](https://arxiv.org/search/cs?searchtype=author&query=Alonso%2C+J), [Matthew Huggins](https://arxiv.org/search/cs?searchtype=author&query=Huggins%2C+M), [Cynthia Breazeal](https://arxiv.org/search/cs?searchtype=author&query=Breazeal%2C+C)

> In recent years, transformer-based language models have achieved state of the art performance in various NLP benchmarks. These models are able to extract mostly distributional information with some semantics from unstructured text, however it has proven challenging to integrate structured information, such as knowledge graphs into these models. We examine a variety of approaches to integrate structured knowledge into current language models and determine challenges, and possible opportunities to leverage both structured and unstructured information sources. From our survey, we find that there are still opportunities at exploiting adapter-based injections and that it may be possible to further combine various of the explored approaches into one system.

| Comments: | Initial Submission                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2101.12294](https://arxiv.org/abs/2101.12294) [cs.CL]** |
|           | (or **[arXiv:2101.12294v1](https://arxiv.org/abs/2101.12294v1) [cs.CL]** for this version) |





<h2 id="2021-02-01-2">2. Few-Shot Domain Adaptation for Grammatical Error Correction via Meta-Learning</h2>

Title: [Few-Shot Domain Adaptation for Grammatical Error Correction via Meta-Learning](https://arxiv.org/abs/2101.12409)

Authors: [Shengsheng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+S), [Yaping Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+Y), [Yun Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Liner Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+L), [Chencheng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Erhong Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+E)

> Most existing Grammatical Error Correction (GEC) methods based on sequence-to-sequence mainly focus on how to generate more pseudo data to obtain better performance. Few work addresses few-shot GEC domain adaptation. In this paper, we treat different GEC domains as different GEC tasks and propose to extend meta-learning to few-shot GEC domain adaptation without using any pseudo data. We exploit a set of data-rich source domains to learn the initialization of model parameters that facilitates fast adaptation on new resource-poor target domains. We adapt GEC model to the first language (L1) of the second language learner. To evaluate the proposed method, we use nine L1s as source domains and five L1s as target domains. Experiment results on the L1 GEC domain adaptation dataset demonstrate that the proposed approach outperforms the multi-task transfer learning baseline by 0.50 F0.5 score on average and enables us to effectively adapt to a new L1 domain with only 200 parallel sentences.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2101.12409](https://arxiv.org/abs/2101.12409) [cs.CL]** |
|           | (or **[arXiv:2101.12409v1](https://arxiv.org/abs/2101.12409v1) [cs.CL]** for this version) |





<h2 id="2021-02-01-3">3. Synthesizing Monolingual Data for Neural Machine Translation</h2>

Title: [Synthesizing Monolingual Data for Neural Machine Translation](https://arxiv.org/abs/2101.12462)

Authors: [Benjamin Marie](https://arxiv.org/search/cs?searchtype=author&query=Marie%2C+B), [Atsushi Fujita](https://arxiv.org/search/cs?searchtype=author&query=Fujita%2C+A)

> In neural machine translation (NMT), monolingual data in the target language are usually exploited through a method so-called "back-translation" to synthesize additional training parallel data. The synthetic data have been shown helpful to train better NMT, especially for low-resource language pairs and domains. Nonetheless, large monolingual data in the target domains or languages are not always available to generate large synthetic parallel data. In this work, we propose a new method to generate large synthetic parallel data leveraging very small monolingual data in a specific domain. We fine-tune a pre-trained GPT-2 model on such small in-domain monolingual data and use the resulting model to generate a large amount of synthetic in-domain monolingual data. Then, we perform back-translation, or forward translation, to generate synthetic in-domain parallel data. Our preliminary experiments on three language pairs and five domains show the effectiveness of our method to generate fully synthetic but useful in-domain parallel data for improving NMT in all configurations. We also show promising results in extreme adaptation for personalized NMT.

| Comments: | Preliminary work                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2101.12462](https://arxiv.org/abs/2101.12462) [cs.CL]** |
|           | (or **[arXiv:2101.12462v1](https://arxiv.org/abs/2101.12462v1) [cs.CL]** for this version) |





<h2 id="2021-02-01-4">4. Transition based Graph Decoder for Neural Machine Translation</h2>

Title: [Transition based Graph Decoder for Neural Machine Translation](https://arxiv.org/abs/2101.12640)

Authors: [Leshem Choshen](https://arxiv.org/search/cs?searchtype=author&query=Choshen%2C+L), [Omri Abend](https://arxiv.org/search/cs?searchtype=author&query=Abend%2C+O)

> While a number of works showed gains from incorporating source-side symbolic syntactic and semantic structure into neural machine translation (NMT), much fewer works addressed the decoding of such structure.
> We propose a general Transformer-based approach for tree and graph decoding based on generating a sequence of transitions, inspired by a similar approach that uses RNNs by Dyer (2016).
> Experiments with using the proposed decoder with Universal Dependencies syntax on English-German, German-English and English-Russian show improved performance over the standard Transformer decoder, as well as over ablated versions of the model.\tacltxt{\footnote{All code implementing the presented models will be released upon acceptance.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2101.12640](https://arxiv.org/abs/2101.12640) [cs.CL]** |
|           | (or **[arXiv:2101.12640v1](https://arxiv.org/abs/2101.12640v1) [cs.CL]** for this version) |