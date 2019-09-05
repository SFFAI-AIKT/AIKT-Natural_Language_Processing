# Daily arXiv: Machine Translation - Sep., 2019

# Index

- [2019-09-05](#2019-09-05)
  - [1. The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives](#2019-09-05-1)
  - [2. Context-Aware Monolingual Repair for Neural Machine Translation](#2019-09-05-2)
  - [3. Simpler and Faster Learning of Adaptive Policies for Simultaneous Translation](#2019-09-05-3)
  - [4. Do We Really Need Fully Unsupervised Cross-Lingual Embeddings?](#2019-09-05-4)
  - [5. SAO WMT19 Test Suite: Machine Translation of Audit Reports](#2019-09-05-5)
- [2019-09-04](#2019-09-04)
  - [1. Hybrid Data-Model Parallel Training for Sequence-to-Sequence Recurrent Neural Network Machine Translation](#2019-09-04-1)
  - [2. Handling Syntactic Divergence in Low-resource Machine Translation](#2019-09-04-2)
  - [3. Evaluating Pronominal Anaphora in Machine Translation: An Evaluation Measure and a Test Suite](#2019-09-04-3)
  - [4. Improving Back-Translation with Uncertainty-based Confidence Estimation](#2019-09-04-4)
  - [5. Explicit Cross-lingual Pre-training for Unsupervised Machine Translation](#2019-09-04-5)
  - [6. Towards Understanding Neural Machine Translation with Word Importance](#2019-09-04-6)
  - [7. One Model to Learn Both: Zero Pronoun Prediction and Translation](#2019-09-04-7)
  - [8. Evaluating the Cross-Lingual Effectiveness of Massively Multilingual Neural Machine Translation](#2019-09-04-8)
  - [9. Improving Context-aware Neural Machine Translation with Target-side Context](#2019-09-04-9)
  - [10. Enhancing Context Modeling with a Query-Guided Capsule Network for Document-level Translation](#2019-09-04-10)
  - [11. Unicoder: A Universal Language Encoder by Pre-training with Multiple Cross-lingual Tasks](#2019-09-04-11)
  - [12. Multi-agent Learning for Neural Machine Translation](#2019-09-04-12)
  - [13. Bilingual is At Least Monolingual (BALM): A Novel Translation Algorithm that Encodes Monolingual Priors](#2019-09-04-13)
- [2019-09-02](#2019-09-02)
  - [1. Latent Part-of-Speech Sequences for Neural Machine Translation](#2019-09-02-1)
  - [2. Encoders Help You Disambiguate Word Senses in Neural Machine Translation](#2019-09-02-2)
* [2019-08](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-08.md)
* [2019-07](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-07.md)
* [2019-06](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-06.md)
* [2019-05](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-05.md)
* [2019-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-04.md)
* [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)



# 2019-09-05

[Return to Index](#Index)



<h2 id="2019-09-05-1">1. The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives</h2> 

Title: [The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives](https://arxiv.org/abs/1909.01380)

Authors: [Elena Voita](https://arxiv.org/search/cs?searchtype=author&query=Voita%2C+E), [Rico Sennrich](https://arxiv.org/search/cs?searchtype=author&query=Sennrich%2C+R), [Ivan Titov](https://arxiv.org/search/cs?searchtype=author&query=Titov%2C+I)

*(Submitted on 3 Sep 2019)*

> We seek to understand how the representations of individual tokens and the structure of the learned feature space evolve between layers in deep neural networks under different learning objectives. We focus on the Transformers for our analysis as they have been shown effective on various tasks, including machine translation (MT), standard left-to-right language models (LM) and masked language modeling (MLM). Previous work used black-box probing tasks to show that the representations learned by the Transformer differ significantly depending on the objective. In this work, we use canonical correlation analysis and mutual information estimators to study how information flows across Transformer layers and how this process depends on the choice of learning objective. For example, as you go from bottom to top layers, information about the past in left-to-right language models gets vanished and predictions about the future get formed. In contrast, for MLM, representations initially acquire information about the context around the token, partially forgetting the token identity and producing a more generalized token representation. The token identity then gets recreated at the top MLM layers.

| Comments: | EMNLP 2019 (camera-ready)                            |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.01380 [cs.CL]**                         |
|           | (or **arXiv:1909.01380v1 [cs.CL]** for this version) |





<h2 id="2019-09-05-2">2. Context-Aware Monolingual Repair for Neural Machine Translation</h2> 

Title: [Context-Aware Monolingual Repair for Neural Machine Translation](https://arxiv.org/abs/1909.01383)

Authors: [Elena Voita](https://arxiv.org/search/cs?searchtype=author&query=Voita%2C+E), [Rico Sennrich](https://arxiv.org/search/cs?searchtype=author&query=Sennrich%2C+R), [Ivan Titov](https://arxiv.org/search/cs?searchtype=author&query=Titov%2C+I)

*(Submitted on 3 Sep 2019)*

> Modern sentence-level NMT systems often produce plausible translations of isolated sentences. However, when put in context, these translations may end up being inconsistent with each other. We propose a monolingual DocRepair model to correct inconsistencies between sentence-level translations. DocRepair performs automatic post-editing on a sequence of sentence-level translations, refining translations of sentences in context of each other. For training, the DocRepair model requires only monolingual document-level data in the target language. It is trained as a monolingual sequence-to-sequence model that maps inconsistent groups of sentences into consistent ones. The consistent groups come from the original training data; the inconsistent groups are obtained by sampling round-trip translations for each isolated sentence. We show that this approach successfully imitates inconsistencies we aim to fix: using contrastive evaluation, we show large improvements in the translation of several contextual phenomena in an English-Russian translation task, as well as improvements in the BLEU score. We also conduct a human evaluation and show a strong preference of the annotators to corrected translations over the baseline ones. Moreover, we analyze which discourse phenomena are hard to capture using monolingual data only.

| Comments: | EMNLP 2019 (camera-ready)                            |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.01383 [cs.CL]**                         |
|           | (or **arXiv:1909.01383v1 [cs.CL]** for this version) |





<h2 id="2019-09-05-3">3. Simpler and Faster Learning of Adaptive Policies for Simultaneous Translation</h2> 

Title: [Simpler and Faster Learning of Adaptive Policies for Simultaneous Translation](https://arxiv.org/abs/1909.01559)

Authors: [Baigong Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+B), [Renjie Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+R), [Mingbo Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+M), [Liang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+L)

*(Submitted on 4 Sep 2019)*

> Simultaneous translation is widely useful but remains challenging. Previous work falls into two main categories: (a) fixed-latency policies such as Ma et al. (2019) and (b) adaptive policies such as Gu et al. (2017). The former are simple and effective, but have to aggressively predict future content due to diverging source-target word order; the latter do not anticipate, but suffer from unstable and inefficient training. To combine the merits of both approaches, we propose a simple supervised-learning framework to learn an adaptive policy from oracle READ/WRITE sequences generated from parallel text. At each step, such an oracle sequence chooses to WRITE the next target word if the available source sentence context provides enough information to do so, otherwise READ the next source word. Experiments on German<->English show that our method, without retraining the underlying NMT model, can learn flexible policies with better BLEU scores and similar latencies compared to previous work.

| Comments: | EMNLP 2019                                           |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.01559 [cs.CL]**                         |
|           | (or **arXiv:1909.01559v1 [cs.CL]** for this version) |





<h2 id="2019-09-05-4">4. Do We Really Need Fully Unsupervised Cross-Lingual Embeddings?</h2> 

Title: [Do We Really Need Fully Unsupervised Cross-Lingual Embeddings?](https://arxiv.org/abs/1909.01638)

Authors: [Ivan Vulić](https://arxiv.org/search/cs?searchtype=author&query=Vulić%2C+I), [Goran Glavaš](https://arxiv.org/search/cs?searchtype=author&query=Glavaš%2C+G), [Roi Reichart](https://arxiv.org/search/cs?searchtype=author&query=Reichart%2C+R), [Anna Korhonen](https://arxiv.org/search/cs?searchtype=author&query=Korhonen%2C+A)

*(Submitted on 4 Sep 2019)*

> Recent efforts in cross-lingual word embedding (CLWE) learning have predominantly focused on fully unsupervised approaches that project monolingual embeddings into a shared cross-lingual space without any cross-lingual signal. The lack of any supervision makes such approaches conceptually attractive. Yet, their only core difference from (weakly) supervised projection-based CLWE methods is in the way they obtain a seed dictionary used to initialize an iterative self-learning procedure. The fully unsupervised methods have arguably become more robust, and their primary use case is CLWE induction for pairs of resource-poor and distant languages. In this paper, we question the ability of even the most robust unsupervised CLWE approaches to induce meaningful CLWEs in these more challenging settings. A series of bilingual lexicon induction (BLI) experiments with 15 diverse languages (210 language pairs) show that fully unsupervised CLWE methods still fail for a large number of language pairs (e.g., they yield zero BLI performance for 87/210 pairs). Even when they succeed, they never surpass the performance of weakly supervised methods (seeded with 500-1,000 translation pairs) using the same self-learning procedure in any BLI setup, and the gaps are often substantial. These findings call for revisiting the main motivations behind fully unsupervised CLWE methods.

| Comments: | EMNLP 2019 (Long paper)                              |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.01638 [cs.CL]**                         |
|           | (or **arXiv:1909.01638v1 [cs.CL]** for this version) |





<h2 id="2019-09-05-5">5. SAO WMT19 Test Suite: Machine Translation of Audit Reports</h2> 

Title: [SAO WMT19 Test Suite: Machine Translation of Audit Reports](https://arxiv.org/abs/1909.01701)

Authors: [Tereza Vojtěchová](https://arxiv.org/search/cs?searchtype=author&query=Vojtěchová%2C+T), [Michal Novák](https://arxiv.org/search/cs?searchtype=author&query=Novák%2C+M), [Miloš Klouček](https://arxiv.org/search/cs?searchtype=author&query=Klouček%2C+M), [Ondřej Bojar](https://arxiv.org/search/cs?searchtype=author&query=Bojar%2C+O)

*(Submitted on 4 Sep 2019)*

> This paper describes a machine translation test set of documents from the auditing domain and its use as one of the "test suites" in the WMT19 News Translation Task for translation directions involving Czech, English and German. 
> Our evaluation suggests that current MT systems optimized for the general news domain can perform quite well even in the particular domain of audit reports. The detailed manual evaluation however indicates that deep factual knowledge of the domain is necessary. For the naked eye of a non-expert, translations by many systems seem almost perfect and automatic MT evaluation with one reference is practically useless for considering these details. 
> Furthermore, we show on a sample document from the domain of agreements that even the best systems completely fail in preserving the semantics of the agreement, namely the identity of the parties.

| Comments:          | WMT19 ([this http URL](http://www.statmt.org/wmt19/))        |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**                         |
| Journal reference: | Vojt\v{e}chov\'a et al. (2019): SAO WMT19 Test Suite: Machine Translation of Audit Reports. In: Fourth Conference on Machine Translation - Proceedings of the Conference, pp. 680-692, ACL, ISBN 978-1-950737-27-7 |
| Cite as:           | **arXiv:1909.01701 [cs.CL]**                                 |
|                    | (or **arXiv:1909.01701v1 [cs.CL]** for this version)         |





# 2019-09-04

[Return to Index](#Index)



<h2 id="2019-09-04-1">1. Hybrid Data-Model Parallel Training for Sequence-to-Sequence Recurrent Neural Network Machine Translation</h2> 

Title: [Hybrid Data-Model Parallel Training for Sequence-to-Sequence Recurrent Neural Network Machine Translation](https://arxiv.org/abs/1909.00562)

Authors: [Junya Ono](https://arxiv.org/search/cs?searchtype=author&query=Ono%2C+J), [Masao Utiyama](https://arxiv.org/search/cs?searchtype=author&query=Utiyama%2C+M), [Eiichiro Sumita](https://arxiv.org/search/cs?searchtype=author&query=Sumita%2C+E)

*(Submitted on 2 Sep 2019)*

> Reduction of training time is an important issue in many tasks like patent translation involving neural networks. Data parallelism and model parallelism are two common approaches for reducing training time using multiple graphics processing units (GPUs) on one machine. In this paper, we propose a hybrid data-model parallel approach for sequence-to-sequence (Seq2Seq) recurrent neural network (RNN) machine translation. We apply a model parallel approach to the RNN encoder-decoder part of the Seq2Seq model and a data parallel approach to the attention-softmax part of the model. We achieved a speed-up of 4.13 to 4.20 times when using 4 GPUs compared with the training speed when using 1 GPU without affecting machine translation accuracy as measured in terms of BLEU scores.

| Comments: | 9 pages, 4 figures, 5 tables                                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Distributed, Parallel, and Cluster Computing (cs.DC)**; Computation and Language (cs.CL); Machine Learning (cs.LG); Neural and Evolutionary Computing (cs.NE) |
| Cite as:  | **arXiv:1909.00562 [cs.DC]**                                 |
|           | (or **arXiv:1909.00562v1 [cs.DC]** for this version)         |





<h2 id="2019-09-04-2">2. Handling Syntactic Divergence in Low-resource Machine Translation</h2> 

Title: [Handling Syntactic Divergence in Low-resource Machine Translation](https://arxiv.org/abs/1909.00040)

Authors: [Chunting Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+C), [Xuezhe Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+X), [Junjie Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+J), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G)

*(Submitted on 30 Aug 2019)*

> Despite impressive empirical successes of neural machine translation (NMT) on standard benchmarks, limited parallel data impedes the application of NMT models to many language pairs. Data augmentation methods such as back-translation make it possible to use monolingual data to help alleviate these issues, but back-translation itself fails in extreme low-resource scenarios, especially for syntactically divergent languages. In this paper, we propose a simple yet effective solution, whereby target-language sentences are re-ordered to match the order of the source and used as an additional source of training-time supervision. Experiments with simulated low-resource Japanese-to-English, and real low-resource Uyghur-to-English scenarios find significant improvements over other semi-supervised alternatives.

| Comments: | Accepted by EMNLP 2019 (short paper)                 |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.00040 [cs.CL]**                         |
|           | (or **arXiv:1909.00040v1 [cs.CL]** for this version) |







<h2 id="2019-09-04-3">3. Evaluating Pronominal Anaphora in Machine Translation: An Evaluation Measure and a Test Suite</h2> 

Title: [Evaluating Pronominal Anaphora in Machine Translation: An Evaluation Measure and a Test Suite](https://arxiv.org/abs/1909.00131)

Authors: [Prathyusha Jwalapuram](https://arxiv.org/search/cs?searchtype=author&query=Jwalapuram%2C+P), [Shafiq Joty](https://arxiv.org/search/cs?searchtype=author&query=Joty%2C+S), [Irina Temnikova](https://arxiv.org/search/cs?searchtype=author&query=Temnikova%2C+I), [Preslav Nakov](https://arxiv.org/search/cs?searchtype=author&query=Nakov%2C+P)

*(Submitted on 31 Aug 2019)*

> The ongoing neural revolution in machine translation has made it easier to model larger contexts beyond the sentence-level, which can potentially help resolve some discourse-level ambiguities such as pronominal anaphora, thus enabling better translations. Unfortunately, even when the resulting improvements are seen as substantial by humans, they remain virtually unnoticed by traditional automatic evaluation measures like BLEU, as only a few words end up being affected. Thus, specialized evaluation measures are needed. With this aim in mind, we contribute an extensive, targeted dataset that can be used as a test suite for pronoun translation, covering multiple source languages and different pronoun errors drawn from real system translations, for English. We further propose an evaluation measure to differentiate good and bad pronoun translations. We also conduct a user study to report correlations with human judgments.

| Comments: | Accepted at EMNLP 2019                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1909.00131 [cs.CL]**                                 |
|           | (or **arXiv:1909.00131v1 [cs.CL]** for this version)         |







<h2 id="2019-09-04-4">4. Improving Back-Translation with Uncertainty-based Confidence Estimation</h2> 

Title: [Improving Back-Translation with Uncertainty-based Confidence Estimation](https://arxiv.org/abs/1909.00157)

Authors: [Shuo Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Yang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Chao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Huanbo Luan](https://arxiv.org/search/cs?searchtype=author&query=Luan%2C+H), [Maosong Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+M)

*(Submitted on 31 Aug 2019)*

> While back-translation is simple and effective in exploiting abundant monolingual corpora to improve low-resource neural machine translation (NMT), the synthetic bilingual corpora generated by NMT models trained on limited authentic bilingual data are inevitably noisy. In this work, we propose to quantify the confidence of NMT model predictions based on model uncertainty. With word- and sentence-level confidence measures based on uncertainty, it is possible for back-translation to better cope with noise in synthetic bilingual corpora. Experiments on Chinese-English and English-German translation tasks show that uncertainty-based confidence estimation significantly improves the performance of back-translation.

| Comments: | EMNLP 2019                                           |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.00157 [cs.CL]**                         |
|           | (or **arXiv:1909.00157v1 [cs.CL]** for this version) |







<h2 id="2019-09-04-5">5. Explicit Cross-lingual Pre-training for Unsupervised Machine Translation</h2> 

Title: [Explicit Cross-lingual Pre-training for Unsupervised Machine Translation](https://arxiv.org/abs/1909.00180)

Authors: [Shuo Ren](https://arxiv.org/search/cs?searchtype=author&query=Ren%2C+S), [Yu Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+Y), [Shujie Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+S), [Ming Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+M), [Shuai Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+S)

*(Submitted on 31 Aug 2019)*

> Pre-training has proven to be effective in unsupervised machine translation due to its ability to model deep context information in cross-lingual scenarios. However, the cross-lingual information obtained from shared BPE spaces is inexplicit and limited. In this paper, we propose a novel cross-lingual pre-training method for unsupervised machine translation by incorporating explicit cross-lingual training signals. Specifically, we first calculate cross-lingual n-gram embeddings and infer an n-gram translation table from them. With those n-gram translation pairs, we propose a new pre-training model called Cross-lingual Masked Language Model (CMLM), which randomly chooses source n-grams in the input text stream and predicts their translation candidates at each time step. Experiments show that our method can incorporate beneficial cross-lingual information into pre-trained models. Taking pre-trained CMLM models as the encoder and decoder, we significantly improve the performance of unsupervised machine translation.

| Comments: | Accepted to EMNLP2019; 10 pages, 2 figures           |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.00180 [cs.CL]**                         |
|           | (or **arXiv:1909.00180v1 [cs.CL]** for this version) |







<h2 id="2019-09-04-6">6. Towards Understanding Neural Machine Translation with Word Importance</h2> 

Title: [Towards Understanding Neural Machine Translation with Word Importance](https://arxiv.org/abs/1909.00326)

Authors: [Shilin He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+S), [Zhaopeng Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+Z), [Xing Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Longyue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Michael R. Lyu](https://arxiv.org/search/cs?searchtype=author&query=Lyu%2C+M+R), [Shuming Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+S)

*(Submitted on 1 Sep 2019)*

> Although neural machine translation (NMT) has advanced the state-of-the-art on various language pairs, the interpretability of NMT remains unsatisfactory. In this work, we propose to address this gap by focusing on understanding the input-output behavior of NMT models. Specifically, we measure the word importance by attributing the NMT output to every input word through a gradient-based method. We validate the approach on a couple of perturbation operations, language pairs, and model architectures, demonstrating its superiority on identifying input words with higher influence on translation performance. Encouragingly, the calculated importance can serve as indicators of input words that are under-translated by NMT models. Furthermore, our analysis reveals that words of certain syntactic categories have higher importance while the categories vary across language pairs, which can inspire better design principles of NMT architectures for multi-lingual translation.

| Comments: | EMNLP 2019                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1909.00326 [cs.CL]**                                 |
|           | (or **arXiv:1909.00326v1 [cs.CL]** for this version)         |







<h2 id="2019-09-04-7">7. One Model to Learn Both: Zero Pronoun Prediction and Translation</h2> 

Title: [One Model to Learn Both: Zero Pronoun Prediction and Translation](https://arxiv.org/abs/1909.00369)

Authors: [Longyue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Zhaopeng Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+Z), [Xing Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Shuming Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+S)

*(Submitted on 1 Sep 2019)*

> Zero pronouns (ZPs) are frequently omitted in pro-drop languages, but should be recalled in non-pro-drop languages. This discourse phenomenon poses a significant challenge for machine translation (MT) when translating texts from pro-drop to non-pro-drop languages. In this paper, we propose a unified and discourse-aware ZP translation approach for neural MT models. Specifically, we jointly learn to predict and translate ZPs in an end-to-end manner, allowing both components to interact with each other. In addition, we employ hierarchical neural networks to exploit discourse-level context, which is beneficial for ZP prediction and thus translation. Experimental results on both Chinese-English and Japanese-English data show that our approach significantly and accumulatively improves both translation performance and ZP prediction accuracy over not only baseline but also previous works using external ZP prediction models. Extensive analyses confirm that the performance improvement comes from the alleviation of different kinds of errors especially caused by subjective ZPs.

| Comments: | EMNLP 2019                                           |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.00369 [cs.CL]**                         |
|           | (or **arXiv:1909.00369v1 [cs.CL]** for this version) |







<h2 id="2019-09-04-8">8. Evaluating the Cross-Lingual Effectiveness of Massively Multilingual Neural Machine Translation</h2> 

Title: [Evaluating the Cross-Lingual Effectiveness of Massively Multilingual Neural Machine Translation](https://arxiv.org/abs/1909.00437)

Authors: [Aditya Siddhant](https://arxiv.org/search/cs?searchtype=author&query=Siddhant%2C+A), [Melvin Johnson](https://arxiv.org/search/cs?searchtype=author&query=Johnson%2C+M), [Henry Tsai](https://arxiv.org/search/cs?searchtype=author&query=Tsai%2C+H), [Naveen Arivazhagan](https://arxiv.org/search/cs?searchtype=author&query=Arivazhagan%2C+N), [Jason Riesa](https://arxiv.org/search/cs?searchtype=author&query=Riesa%2C+J), [Ankur Bapna](https://arxiv.org/search/cs?searchtype=author&query=Bapna%2C+A), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O), [Karthik Raman](https://arxiv.org/search/cs?searchtype=author&query=Raman%2C+K)

*(Submitted on 1 Sep 2019)*

> The recently proposed massively multilingual neural machine translation (NMT) system has been shown to be capable of translating over 100 languages to and from English within a single model. Its improved translation performance on low resource languages hints at potential cross-lingual transfer capability for downstream tasks. In this paper, we evaluate the cross-lingual effectiveness of representations from the encoder of a massively multilingual NMT model on 5 downstream classification and sequence labeling tasks covering a diverse set of over 50 languages. We compare against a strong baseline, multilingual BERT (mBERT), in different cross-lingual transfer learning scenarios and show gains in zero-shot transfer in 4 out of these 5 tasks.

| Subjects: | **Computation and Language (cs.CL)**                 |
| --------- | ---------------------------------------------------- |
| Cite as:  | **arXiv:1909.00437 [cs.CL]**                         |
|           | (or **arXiv:1909.00437v1 [cs.CL]** for this version) |







<h2 id="2019-09-04-9">9. Improving Context-aware Neural Machine Translation with Target-side Context</h2> 

Title: [Improving Context-aware Neural Machine Translation with Target-side Context](https://arxiv.org/abs/1909.00531)

Authors: [Hayahide Yamagishi](https://arxiv.org/search/cs?searchtype=author&query=Yamagishi%2C+H), [Mamoru Komachi](https://arxiv.org/search/cs?searchtype=author&query=Komachi%2C+M)

*(Submitted on 2 Sep 2019)*

> In recent years, several studies on neural machine translation (NMT) have attempted to use document-level context by using a multi-encoder and two attention mechanisms to read the current and previous sentences to incorporate the context of the previous sentences. These studies concluded that the target-side context is less useful than the source-side context. However, we considered that the reason why the target-side context is less useful lies in the architecture used to model these contexts. 
> Therefore, in this study, we investigate how the target-side context can improve context-aware neural machine translation. We propose a weight sharing method wherein NMT saves decoder states and calculates an attention vector using the saved states when translating a current sentence. Our experiments show that the target-side context is also useful if we plug it into NMT as the decoder state when translating a previous sentence.

| Comments: | 12 pages; PACLING 2019                               |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.00531 [cs.CL]**                         |
|           | (or **arXiv:1909.00531v1 [cs.CL]** for this version) |







<h2 id="2019-09-04-10">10. Enhancing Context Modeling with a Query-Guided Capsule Network for Document-level Translation</h2> 

Title: [Enhancing Context Modeling with a Query-Guided Capsule Network for Document-level Translation](https://arxiv.org/abs/1909.00564)

Authors: [Zhengxin Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Jinchao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+J), [Fandong Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+F), [Shuhao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+S), [Yang Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+Y), [Jie Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J)

*(Submitted on 2 Sep 2019)*

> Context modeling is essential to generate coherent and consistent translation for Document-level Neural Machine Translations. The widely used method for document-level translation usually compresses the context information into a representation via hierarchical attention networks. However, this method neither considers the relationship between context words nor distinguishes the roles of context words. To address this problem, we propose a query-guided capsule networks to cluster context information into different perspectives from which the target translation may concern. Experiment results show that our method can significantly outperform strong baselines on multiple data sets of different domains.

| Comments: | 11 pages, 7 figures, 2019 Conference on Empirical Methods in Natural Language Processing |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1909.00564 [cs.CL]**                                 |
|           | (or **arXiv:1909.00564v1 [cs.CL]** for this version)         |







<h2 id="2019-09-04-11">11. Unicoder: A Universal Language Encoder by Pre-training with Multiple Cross-lingual Tasks</h2> 

Title: [Unicoder: A Universal Language Encoder by Pre-training with Multiple Cross-lingual Tasks](https://arxiv.org/abs/1909.00964)

Authors: [Haoyang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+H), [Yaobo Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+Y), [Nan Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan%2C+N), [Ming Gong](https://arxiv.org/search/cs?searchtype=author&query=Gong%2C+M), [Linjun Shou](https://arxiv.org/search/cs?searchtype=author&query=Shou%2C+L), [Daxin Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+D), [Ming Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+M)

*(Submitted on 3 Sep 2019 ([v1](https://arxiv.org/abs/1909.00964v1)), last revised 4 Sep 2019 (this version, v2))*

> We present Unicoder, a universal language encoder that is insensitive to different languages. Given an arbitrary NLP task, a model can be trained with Unicoder using training data in one language and directly applied to inputs of the same task in other languages. Comparing to similar efforts such as Multilingual BERT and XLM, three new cross-lingual pre-training tasks are proposed, including cross-lingual word recovery, cross-lingual paraphrase classification and cross-lingual masked language model. These tasks help Unicoder learn the mappings among different languages from more perspectives. We also find that doing fine-tuning on multiple languages together can bring further improvement. Experiments are performed on two tasks: cross-lingual natural language inference (XNLI) and cross-lingual question answering (XQA), where XLM is our baseline. On XNLI, 1.8% averaged accuracy improvement (on 15 languages) is obtained. On XQA, which is a new cross-lingual dataset built by us, 5.5% averaged accuracy improvement (on French and German) is obtained.

| Comments: | Accepted to EMNLP2019; 10 pages, 2 figures           |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.00964 [cs.CL]**                         |
|           | (or **arXiv:1909.00964v2 [cs.CL]** for this version) |







<h2 id="2019-09-04-12">12. Multi-agent Learning for Neural Machine Translation</h2> 

Title: [Multi-agent Learning for Neural Machine Translation](https://arxiv.org/abs/1909.01101)

Authors: [Tianchi Bi](https://arxiv.org/search/cs?searchtype=author&query=Bi%2C+T), [Hao Xiong](https://arxiv.org/search/cs?searchtype=author&query=Xiong%2C+H), [Zhongjun He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+Z), [Hua Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+H), [Haifeng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+H)

*(Submitted on 3 Sep 2019)*

> Conventional Neural Machine Translation (NMT) models benefit from the training with an additional agent, e.g., dual learning, and bidirectional decoding with one agent decoding from left to right and the other decoding in the opposite direction. In this paper, we extend the training framework to the multi-agent scenario by introducing diverse agents in an interactive updating process. At training time, each agent learns advanced knowledge from others, and they work together to improve translation quality. Experimental results on NIST Chinese-English, IWSLT 2014 German-English, WMT 2014 English-German and large-scale Chinese-English translation tasks indicate that our approach achieves absolute improvements over the strong baseline systems and shows competitive performance on all tasks.

| Comments: | Accepted by EMNLP2019                                |
| --------- | ---------------------------------------------------- |
| Subjects: | **Computation and Language (cs.CL)**                 |
| Cite as:  | **arXiv:1909.01101 [cs.CL]**                         |
|           | (or **arXiv:1909.01101v1 [cs.CL]** for this version) |







<h2 id="2019-09-04-13">13. Bilingual is At Least Monolingual (BALM): A Novel Translation Algorithm that Encodes Monolingual Priors</h2> 

Title: [Bilingual is At Least Monolingual (BALM): A Novel Translation Algorithm that Encodes Monolingual Priors](https://arxiv.org/abs/1909.01146)

Authors: [Jeffrey Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng%2C+J), [Chris Callison-Burch](https://arxiv.org/search/cs?searchtype=author&query=Callison-Burch%2C+C)

*(Submitted on 30 Aug 2019)*

> State-of-the-art machine translation (MT) models do not use knowledge of any single language's structure; this is the equivalent of asking someone to translate from English to German while knowing neither language. BALM is a framework incorporates monolingual priors into an MT pipeline; by casting input and output languages into embedded space using BERT, we can solve machine translation with much simpler models. We find that English-to-German translation on the Multi30k dataset can be solved with a simple feedforward network under the BALM framework with near-SOTA BLEU scores.

| Comments: | 15 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **arXiv:1909.01146 [cs.CL]**                                 |
|           | (or **arXiv:1909.01146v1 [cs.CL]** for this version)         |







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
