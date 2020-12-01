# Daily arXiv: Machine Translation - December, 2020

# Index


- [2020-12-01](#2020-12-01)
  - [1. EdgeBERT: Optimizing On-Chip Inference for Multi-Task NLP](#2020-12-01-1)
  - [2. Understanding How BERT Learns to Identify Edits](#2020-12-01-2)
  - [3. Using Multiple Subwords to Improve English-Esperanto Automated Literary Translation Quality](#2020-12-01-3)
  - [4. Intrinsic Knowledge Evaluation on Chinese Language Models](#2020-12-01-4)
  - [5. Dynamic Curriculum Learning for Low-Resource Neural Machine Translation](#2020-12-01-5)
  - [6. A Simple and Effective Approach to Robust Unsupervised Bilingual Dictionary Induction](#2020-12-01-6)
  - [7. Machine Translation of Novels in the Age of Transformer](#2020-12-01-7)
  - [8. Multimodal Pretraining Unmasked: Unifying the Vision and Language BERTs](#2020-12-01-8)

- [Other Columns](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2020-12-01

[Return to Index](#Index)



<h2 id="2020-12-01-1">1. EdgeBERT: Optimizing On-Chip Inference for Multi-Task NLP</h2>

Title: [EdgeBERT: Optimizing On-Chip Inference for Multi-Task NLP](https://arxiv.org/abs/2011.14203)

Authors: [Thierry Tambe](https://arxiv.org/search/cs?searchtype=author&query=Tambe%2C+T), [Coleman Hooper](https://arxiv.org/search/cs?searchtype=author&query=Hooper%2C+C), [Lillian Pentecost](https://arxiv.org/search/cs?searchtype=author&query=Pentecost%2C+L), [En-Yu Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+E), [Marco Donato](https://arxiv.org/search/cs?searchtype=author&query=Donato%2C+M), [Victor Sanh](https://arxiv.org/search/cs?searchtype=author&query=Sanh%2C+V), [Alexander M. Rush](https://arxiv.org/search/cs?searchtype=author&query=Rush%2C+A+M), [David Brooks](https://arxiv.org/search/cs?searchtype=author&query=Brooks%2C+D), [Gu-Yeon Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+G)

> Transformer-based language models such as BERT provide significant accuracy improvement to a multitude of natural language processing (NLP) tasks. However, their hefty computational and memory demands make them challenging to deploy to resource-constrained edge platforms with strict latency requirements.
> We present EdgeBERT an in-depth and principled algorithm and hardware design methodology to achieve minimal latency and energy consumption on multi-task NLP inference. Compared to the ALBERT baseline, we achieve up to 2.4x and 13.4x inference latency and memory savings, respectively, with less than 1%-pt drop in accuracy on several GLUE benchmarks by employing a calibrated combination of 1) entropy-based early stopping, 2) adaptive attention span, 3) movement and magnitude pruning, and 4) floating-point quantization.
> Furthermore, in order to maximize the benefits of these algorithms in always-on and intermediate edge computing settings, we specialize a scalable hardware architecture wherein floating-point bit encodings of the shareable multi-task embedding parameters are stored in high-density non-volatile memory. Altogether, EdgeBERT enables fully on-chip inference acceleration of NLP workloads with 5.2x, and 157x lower energy than that of an un-optimized accelerator and CUDA adaptations on an Nvidia Jetson Tegra X2 mobile GPU, respectively.

| Comments: | 11 pages plus references                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Hardware Architecture (cs.AR)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2011.14203](https://arxiv.org/abs/2011.14203) [cs.AR]** |
|           | (or **[arXiv:2011.14203v1](https://arxiv.org/abs/2011.14203v1) [cs.AR]** for this version) |





<h2 id="2020-12-01-2">2. Understanding How BERT Learns to Identify Edits</h2>

Title: [Understanding How BERT Learns to Identify Edits](https://arxiv.org/abs/2011.14039)

Authors: [Samuel Stevens](https://arxiv.org/search/cs?searchtype=author&query=Stevens%2C+S), [Yu Su](https://arxiv.org/search/cs?searchtype=author&query=Su%2C+Y)

> Pre-trained transformer language models such as BERT are ubiquitous in NLP research, leading to work on understanding how and why these models work. Attention mechanisms have been proposed as a means of interpretability with varying conclusions. We propose applying BERT-based models to a sequence classification task and using the data set's labeling schema to measure each model's interpretability. We find that classification performance scores do not always correlate with interpretability. Despite this, BERT's attention weights are interpretable for over 70% of examples.

| Comments: | 8 pages, 11 figures. A work in progress                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2011.14039](https://arxiv.org/abs/2011.14039) [cs.CL]** |
|           | (or **[arXiv:2011.14039v1](https://arxiv.org/abs/2011.14039v1) [cs.CL]** for this version) |







<h2 id="2020-12-01-3">3. Using Multiple Subwords to Improve English-Esperanto Automated Literary Translation Quality</h2>

Title: [Using Multiple Subwords to Improve English-Esperanto Automated Literary Translation Quality](https://arxiv.org/abs/2011.14190)

Authors: [Alberto Poncelas](https://arxiv.org/search/cs?searchtype=author&query=Poncelas%2C+A), [Jan Buts](https://arxiv.org/search/cs?searchtype=author&query=Buts%2C+J), [James Hadley](https://arxiv.org/search/cs?searchtype=author&query=Hadley%2C+J), [Andy Way](https://arxiv.org/search/cs?searchtype=author&query=Way%2C+A)

> Building Machine Translation (MT) systems for low-resource languages remains challenging. For many language pairs, parallel data are not widely available, and in such cases MT models do not achieve results comparable to those seen with high-resource languages.
> When data are scarce, it is of paramount importance to make optimal use of the limited material available. To that end, in this paper we propose employing the same parallel sentences multiple times, only changing the way the words are split each time. For this purpose we use several Byte Pair Encoding models, with various merge operations used in their configuration.
> In our experiments, we use this technique to expand the available data and improve an MT system involving a low-resource language pair, namely English-Esperanto.
> As an additional contribution, we made available a set of English-Esperanto parallel data in the literary domain.

| Subjects:          | **Computation and Language (cs.CL)**                         |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | The 3rd Workshop on Technologies for MT of Low Resource Languages (LoResMT 2020) |
| Cite as:           | **[arXiv:2011.14190](https://arxiv.org/abs/2011.14190) [cs.CL]** |
|                    | (or **[arXiv:2011.14190v1](https://arxiv.org/abs/2011.14190v1) [cs.CL]** for this version) |







<h2 id="2020-12-01-4">4. Intrinsic Knowledge Evaluation on Chinese Language Models</h2>

Title: [Intrinsic Knowledge Evaluation on Chinese Language Models](https://arxiv.org/abs/2011.14277)

Authors: [Zhiruo Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Z), [Renfen Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+R)

> Recent NLP tasks have benefited a lot from pre-trained language models (LM) since they are able to encode knowledge of various aspects. However, current LM evaluations focus on downstream performance, hence lack to comprehensively inspect in which aspect and to what extent have they encoded knowledge. This paper addresses both queries by proposing four tasks on syntactic, semantic, commonsense, and factual knowledge, aggregating to a total of 39,308 questions covering both linguistic and world knowledge in Chinese. Throughout experiments, our probes and knowledge data prove to be a reliable benchmark for evaluating pre-trained Chinese LMs. Our work is publicly available at [this https URL](https://github.com/ZhiruoWang/ChnEval).

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2011.14277](https://arxiv.org/abs/2011.14277) [cs.CL]** |
|           | (or **[arXiv:2011.14277v1](https://arxiv.org/abs/2011.14277v1) [cs.CL]** for this version) |







<h2 id="2020-12-01-5">5. Dynamic Curriculum Learning for Low-Resource Neural Machine Translation</h2>

Title: [Dynamic Curriculum Learning for Low-Resource Neural Machine Translation](https://arxiv.org/abs/2011.14608)

Authors: [Chen Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+C), [Bojie Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+B), [Yufan Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+Y), [Kai Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+K), [Zeyang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Z), [Shen Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Qi Ju](https://arxiv.org/search/cs?searchtype=author&query=Ju%2C+Q), [Tong Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+T), [Jingbo Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+J)

> Large amounts of data has made neural machine translation (NMT) a big success in recent years. But it is still a challenge if we train these models on small-scale corpora. In this case, the way of using data appears to be more important. Here, we investigate the effective use of training data for low-resource NMT. In particular, we propose a dynamic curriculum learning (DCL) method to reorder training samples in training. Unlike previous work, we do not use a static scoring function for reordering. Instead, the order of training samples is dynamically determined in two ways - loss decline and model competence. This eases training by highlighting easy samples that the current model has enough competence to learn. We test our DCL method in a Transformer-based system. Experimental results show that DCL outperforms several strong baselines on three low-resource machine translation benchmarks and different sized data of WMT' 16 En-De.

| Comments: | COLING 2020                                                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2011.14608](https://arxiv.org/abs/2011.14608) [cs.CL]** |
|           | (or **[arXiv:2011.14608v1](https://arxiv.org/abs/2011.14608v1) [cs.CL]** for this version) |







<h2 id="2020-12-01-6">6. A Simple and Effective Approach to Robust Unsupervised Bilingual Dictionary Induction</h2>

Title: [A Simple and Effective Approach to Robust Unsupervised Bilingual Dictionary Induction](https://arxiv.org/abs/2011.14874)

Authors: [Yanyang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Yingfeng Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+Y), [Ye Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+Y), [Quan Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+Q), [Huizhen Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+H), [Shujian Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Tong Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+T), [Jingbo Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+J)

> Unsupervised Bilingual Dictionary Induction methods based on the initialization and the self-learning have achieved great success in similar language pairs, e.g., English-Spanish. But they still fail and have an accuracy of 0% in many distant language pairs, e.g., English-Japanese. In this work, we show that this failure results from the gap between the actual initialization performance and the minimum initialization performance for the self-learning to succeed. We propose Iterative Dimension Reduction to bridge this gap. Our experiments show that this simple method does not hamper the performance of similar language pairs and achieves an accuracy of 13.64~55.53% between English and four distant languages, i.e., Chinese, Japanese, Vietnamese and Thai.

| Comments: | Accepted by COLING2020                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2011.14874](https://arxiv.org/abs/2011.14874) [cs.CL]** |
|           | (or **[arXiv:2011.14874v1](https://arxiv.org/abs/2011.14874v1) [cs.CL]** for this version) |







<h2 id="2020-12-01-7">7. Machine Translation of Novels in the Age of Transformer</h2>

Title: [Machine Translation of Novels in the Age of Transformer](https://arxiv.org/abs/2011.14979)

Authors: [Antonio Toral](https://arxiv.org/search/cs?searchtype=author&query=Toral%2C+A), [Antoni Oliver](https://arxiv.org/search/cs?searchtype=author&query=Oliver%2C+A), [Pau Ribas Ballestín](https://arxiv.org/search/cs?searchtype=author&query=Ballestín%2C+P+R)

> In this chapter we build a machine translation (MT) system tailored to the literary domain, specifically to novels, based on the state-of-the-art architecture in neural MT (NMT), the Transformer (Vaswani et al., 2017), for the translation direction English-to-Catalan. Subsequently, we assess to what extent such a system can be useful by evaluating its translations, by comparing this MT system against three other systems (two domain-specific systems under the recurrent and phrase-based paradigms and a popular generic on-line system) on three evaluations. The first evaluation is automatic and uses the most-widely used automatic evaluation metric, BLEU. The two remaining evaluations are manual and they assess, respectively, preference and amount of post-editing required to make the translation error-free. As expected, the domain-specific Transformer-based system outperformed the three other systems in all the three evaluations conducted, in all cases by a large margin.

| Comments: | Chapter published in the book Maschinelle Übersetzung für Übersetzungsprofis (pp. 276-295). Jörg Porsiel (Ed.), BDÜ Fachverlag, 2020. ISBN 978-3-946702-09-2 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2011.14979](https://arxiv.org/abs/2011.14979) [cs.CL]** |
|           | (or **[arXiv:2011.14979v1](https://arxiv.org/abs/2011.14979v1) [cs.CL]** for this version) |





<h2 id="2020-12-01-8">8. Multimodal Pretraining Unmasked: Unifying the Vision and Language BERTs</h2>

Title: [Multimodal Pretraining Unmasked: Unifying the Vision and Language BERTs](https://arxiv.org/abs/2011.15124)

Authors: [Emanuele Bugliarello](https://arxiv.org/search/cs?searchtype=author&query=Bugliarello%2C+E), [Ryan Cotterell](https://arxiv.org/search/cs?searchtype=author&query=Cotterell%2C+R), [Naoaki Okazaki](https://arxiv.org/search/cs?searchtype=author&query=Okazaki%2C+N), [Desmond Elliott](https://arxiv.org/search/cs?searchtype=author&query=Elliott%2C+D)

> Large-scale pretraining and task-specific fine-tuning is now the standard methodology for many tasks in computer vision and natural language processing. Recently, a multitude of methods have been proposed for pretraining vision and language BERTs to tackle challenges at the intersection of these two key areas of AI. These models can be categorized into either single-stream or dual-stream encoders. We study the differences between these two categories, and show how they can be unified under a single theoretical framework. We then conduct controlled experiments to discern the empirical differences between five V&L BERTs. Our experiments show that training data and hyperparameters are responsible for most of the differences between the reported results, but they also reveal that the embedding layer plays a crucial role in these massive models.

| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2011.15124](https://arxiv.org/abs/2011.15124) [cs.CL]** |
|           | (or **[arXiv:2011.15124v1](https://arxiv.org/abs/2011.15124v1) [cs.CL]** for this version) |



