# Daily arXiv: Machine Translation - Dec., 2019

# Index

- [2019-12-03](#2019-12-03)
  - [1. Not All Attention Is Needed: Gated Attention Network for Sequence Data](#2019-12-03-1)
  - [2. Modeling Fluency and Faithfulness for Diverse Neural Machine Translation](#2019-12-03-2)
  - [3. Merging External Bilingual Pairs into Neural Machine Translation](#2019-12-03-3)
- [2019-12-02](#2019-12-02)
  - [1. DiscoTK: Using Discourse Structure for Machine Translation Evaluation](#2019-12-02-1)
  - [2. Multimodal Machine Translation through Visuals and Speech](#2019-12-02-2)
  - [3. GitHub Typo Corpus: A Large-Scale Multilingual Dataset of Misspellings and Grammatical Errors](#2019-12-02-3)
  - [4. Neural Chinese Word Segmentation as Sequence to Sequence Translation](#2019-12-02-4)
- [2019-11](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-11.md)
- [2019-10](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-10.md)
- [2019-09](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-09.md)
- [2019-08](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-08.md)
- [2019-07](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-07.md)
- [2019-06](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-06.md)
- [2019-05](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-05.md)
- [2019-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-04.md)
- [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)



# 2019-12-03

[Return to Index](#Index)



<h2 id="2019-12-03-1">1. Not All Attention Is Needed: Gated Attention Network for Sequence Data</h2>

Title: [Not All Attention Is Needed: Gated Attention Network for Sequence Data](https://arxiv.org/abs/1912.00349)

Authors: [Lanqing Xue](https://arxiv.org/search/cs?searchtype=author&query=Xue%2C+L), [Xiaopeng Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+X), [Nevin L. Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+N+L)

*(Submitted on 1 Dec 2019)*

> Although deep neural networks generally have fixed network structures, the concept of dynamic mechanism has drawn more and more attention in recent years. Attention mechanisms compute input-dependent dynamic attention weights for aggregating a sequence of hidden states. Dynamic network configuration in convolutional neural networks (CNNs) selectively activates only part of the network at a time for different inputs. In this paper, we combine the two dynamic mechanisms for text classification tasks. Traditional attention mechanisms attend to the whole sequence of hidden states for an input sentence, while in most cases not all attention is needed especially for long sequences. We propose a novel method called Gated Attention Network (GA-Net) to dynamically select a subset of elements to attend to using an auxiliary network, and compute attention weights to aggregate the selected elements. It avoids a significant amount of unnecessary computation on unattended elements, and allows the model to pay attention to important parts of the sequence. Experiments in various datasets show that the proposed method achieves better performance compared with all baseline models with global or local attention while requiring less computation and achieving better interpretability. It is also promising to extend the idea to more complex attention-based models, such as transformers and seq-to-seq models.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1912.00349](https://arxiv.org/abs/1912.00349) [cs.LG] |
|           | (or [arXiv:1912.00349v1](https://arxiv.org/abs/1912.00349v1) [cs.LG] for this version) |





<h2 id="2019-12-03-2">2. Modeling Fluency and Faithfulness for Diverse Neural Machine Translation</h2>

Title: [Modeling Fluency and Faithfulness for Diverse Neural Machine Translation](https://arxiv.org/abs/1912.00178)

Authors: [Yang Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+Y), [Wanying Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+W), [Shuhao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+S), [Chenze Shao](https://arxiv.org/search/cs?searchtype=author&query=Shao%2C+C), [Wen Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+W), [Zhengxin Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Dong Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+D)

*(Submitted on 30 Nov 2019)*

> Neural machine translation models usually adopt the teacher forcing strategy for training which requires the predicted sequence matches ground truth word by word and forces the probability of each prediction to approach a 0-1 distribution. However, the strategy casts all the portion of the distribution to the ground truth word and ignores other words in the target vocabulary even when the ground truth word cannot dominate the distribution. To address the problem of teacher forcing, we propose a method to introduce an evaluation module to guide the distribution of the prediction. The evaluation module accesses each prediction from the perspectives of fluency and faithfulness to encourage the model to generate the word which has a fluent connection with its past and future translation and meanwhile tends to form a translation equivalent in meaning to the source. The experiments on multiple translation tasks show that our method can achieve significant improvements over strong baselines.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:1912.00178](https://arxiv.org/abs/1912.00178) [cs.CL] |
|           | (or [arXiv:1912.00178v1](https://arxiv.org/abs/1912.00178v1) [cs.CL] for this version) |



<h2 id="2019-12-03-3">3. Merging External Bilingual Pairs into Neural Machine Translation</h2>

Title: [Merging External Bilingual Pairs into Neural Machine Translation](https://arxiv.org/abs/1912.00567)

Authors: [Tao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+T), [Shaohui Kuang](https://arxiv.org/search/cs?searchtype=author&query=Kuang%2C+S), [Deyi Xiong](https://arxiv.org/search/cs?searchtype=author&query=Xiong%2C+D), [António Branco](https://arxiv.org/search/cs?searchtype=author&query=Branco%2C+A)

*(Submitted on 2 Dec 2019)*

> As neural machine translation (NMT) is not easily amenable to explicit correction of errors, incorporating pre-specified translations into NMT is widely regarded as a non-trivial challenge. In this paper, we propose and explore three methods to endow NMT with pre-specified bilingual pairs. Instead, for instance, of modifying the beam search algorithm during decoding or making complex modifications to the attention mechanism --- mainstream approaches to tackling this challenge ---, we experiment with the training data being appropriately pre-processed to add information about pre-specified translations. Extra embeddings are also used to distinguish pre-specified tokens from the other tokens. Extensive experimentation and analysis indicate that over 99% of the pre-specified phrases are successfully translated (given a 85% baseline) and that there is also a substantive improvement in translation quality with the methods explored here.

| Comments:    | 7 pages, 3 figures, 5 tables                                 |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**                         |
| MSC classes: | 68T50                                                        |
| ACM classes: | I.2.7                                                        |
| Cite as:     | [arXiv:1912.00567](https://arxiv.org/abs/1912.00567) [cs.CL] |
|              | (or [arXiv:1912.00567v1](https://arxiv.org/abs/1912.00567v1) [cs.CL] for this version) |





# 2019-12-02

[Return to Index](#Index)



<h2 id="2019-12-02-1">1. DiscoTK: Using Discourse Structure for Machine Translation Evaluation</h2>
Title: [DiscoTK: Using Discourse Structure for Machine Translation Evaluation](https://arxiv.org/abs/1911.12547)

Authors: [Shafiq Joty](https://arxiv.org/search/cs?searchtype=author&query=Joty%2C+S), [Francisco Guzman](https://arxiv.org/search/cs?searchtype=author&query=Guzman%2C+F), [Lluis Marquez](https://arxiv.org/search/cs?searchtype=author&query=Marquez%2C+L), [Preslav Nakov](https://arxiv.org/search/cs?searchtype=author&query=Nakov%2C+P)

*(Submitted on 28 Nov 2019)*

> We present novel automatic metrics for machine translation evaluation that use discourse structure and convolution kernels to compare the discourse tree of an automatic translation with that of the human reference. We experiment with five transformations and augmentations of a base discourse tree representation based on the rhetorical structure theory, and we combine the kernel scores for each of them into a single score. Finally, we add other metrics from the ASIYA MT evaluation toolkit, and we tune the weights of the combination on actual human judgments. Experiments on the WMT12 and WMT13 metrics shared task datasets show correlation with human judgments that outperforms what the best systems that participated in these years achieved, both at the segment and at the system level.

| Comments:          | machine translation evaluation, machine translation, tree kernels, discourse, convolutional kernels, discourse tree, RST, rhetorical structure theory, ASIYA |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| MSC classes:       | 68T50                                                        |
| ACM classes:       | I.2.7                                                        |
| Journal reference: | WMT-2014                                                     |
| Cite as:           | [arXiv:1911.12547](https://arxiv.org/abs/1911.12547) [cs.CL] |
|                    | (or [arXiv:1911.12547v1](https://arxiv.org/abs/1911.12547v1) [cs.CL] for this version) |





<h2 id="2019-12-02-2">2. Multimodal Machine Translation through Visuals and Speech</h2>
Title: [Multimodal Machine Translation through Visuals and Speech](https://arxiv.org/abs/1911.12798)

Authors: [Umut Sulubacak](https://arxiv.org/search/cs?searchtype=author&query=Sulubacak%2C+U), [Ozan Caglayan](https://arxiv.org/search/cs?searchtype=author&query=Caglayan%2C+O), [Stig-Arne Grönroos](https://arxiv.org/search/cs?searchtype=author&query=Grönroos%2C+S), [Aku Rouhe](https://arxiv.org/search/cs?searchtype=author&query=Rouhe%2C+A), [Desmond Elliott](https://arxiv.org/search/cs?searchtype=author&query=Elliott%2C+D), [Lucia Specia](https://arxiv.org/search/cs?searchtype=author&query=Specia%2C+L), [Jörg Tiedemann](https://arxiv.org/search/cs?searchtype=author&query=Tiedemann%2C+J)

*(Submitted on 28 Nov 2019)*

> Multimodal machine translation involves drawing information from more than one modality, based on the assumption that the additional modalities will contain useful alternative views of the input data. The most prominent tasks in this area are spoken language translation, image-guided translation, and video-guided translation, which exploit audio and visual modalities, respectively. These tasks are distinguished from their monolingual counterparts of speech recognition, image captioning, and video captioning by the requirement of models to generate outputs in a different language. This survey reviews the major data resources for these tasks, the evaluation campaigns concentrated around them, the state of the art in end-to-end and pipeline approaches, and also the challenges in performance evaluation. The paper concludes with a discussion of directions for future research in these areas: the need for more expansive and challenging datasets, for targeted evaluations of model performance, and for multimodality in both the input and output space.

| Comments: | 34 pages, 4 tables, 8 figures. Submitted (Nov 2019) to the Machine Translation journal (Springer) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1911.12798](https://arxiv.org/abs/1911.12798) [cs.CL] |
|           | (or [arXiv:1911.12798v1](https://arxiv.org/abs/1911.12798v1) [cs.CL] for this version) |





<h2 id="2019-12-02-3">3. GitHub Typo Corpus: A Large-Scale Multilingual Dataset of Misspellings and Grammatical Errors</h2>
Title: [GitHub Typo Corpus: A Large-Scale Multilingual Dataset of Misspellings and Grammatical Errors](https://arxiv.org/abs/1911.12893)

Authors: [Masato Hagiwara](https://arxiv.org/search/cs?searchtype=author&query=Hagiwara%2C+M), [Masato Mita](https://arxiv.org/search/cs?searchtype=author&query=Mita%2C+M)

*(Submitted on 28 Nov 2019)*

> The lack of large-scale datasets has been a major hindrance to the development of NLP tasks such as spelling correction and grammatical error correction (GEC). As a complementary new resource for these tasks, we present the GitHub Typo Corpus, a large-scale, multilingual dataset of misspellings and grammatical errors along with their corrections harvested from GitHub, a large and popular platform for hosting and sharing git repositories. The dataset, which we have made publicly available, contains more than 350k edits and 65M characters in more than 15 languages, making it the largest dataset of misspellings to date. We also describe our process for filtering true typo edits based on learned classifiers on a small annotated subset, and demonstrate that typo edits can be identified with F1 ~ 0.9 using a very simple classifier with only three features. The detailed analyses of the dataset show that existing spelling correctors merely achieve an F-measure of approx. 0.5, suggesting that the dataset serves as a new, rich source of spelling errors that complement existing datasets.

| Comments: | Submitted at LREC 2020                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1911.12893](https://arxiv.org/abs/1911.12893) [cs.CL] |
|           | (or [arXiv:1911.12893v1](https://arxiv.org/abs/1911.12893v1) [cs.CL] for this version) |





<h2 id="2019-12-02-4">4. Neural Chinese Word Segmentation as Sequence to Sequence Translation</h2>
Title: [Neural Chinese Word Segmentation as Sequence to Sequence Translation](https://arxiv.org/abs/1911.12982)

Authors: [Xuewen Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+X), [Heyan Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+H), [Ping Jian](https://arxiv.org/search/cs?searchtype=author&query=Jian%2C+P), [Yuhang Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+Y), [Xiaochi Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+X), [Yi-Kun Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+Y)

*(Submitted on 29 Nov 2019)*

> Recently, Chinese word segmentation (CWS) methods using neural networks have made impressive progress. Most of them regard the CWS as a sequence labeling problem which construct models based on local features rather than considering global information of input sequence. In this paper, we cast the CWS as a sequence translation problem and propose a novel sequence-to-sequence CWS model with an attention-based encoder-decoder framework. The model captures the global information from the input and directly outputs the segmented sequence. It can also tackle other NLP tasks with CWS jointly in an end-to-end mode. Experiments on Weibo, PKU and MSRA benchmark datasets show that our approach has achieved competitive performances compared with state-of-the-art methods. Meanwhile, we successfully applied our proposed model to jointly learning CWS and Chinese spelling correction, which demonstrates its applicability of multi-task fusion.

| Comments: | In proceedings of SMP 2017 (Chinese National Conference on Social Media Processing) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| DOI:      | [10.1007/978-981-10-6805-8_8](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1007%2F978-981-10-6805-8_8&v=a5b3c37f) |
| Cite as:  | [arXiv:1911.12982](https://arxiv.org/abs/1911.12982) [cs.CL] |
|           | (or [arXiv:1911.12982v1](https://arxiv.org/abs/1911.12982v1) [cs.CL] for this version) |







