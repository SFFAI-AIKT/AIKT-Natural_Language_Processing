# Daily arXiv: Machine Translation - Mar., 2020

# Index

- [2020-03-04](#2020-03-04)
  - [1. Understanding the Prediction Mechanism of Sentiments by XAI Visualization](#2020-03-04-1)
  - [2. XGPT: Cross-modal Generative Pre-Training for Image Captioning](#2020-03-04-2)
- [2020-03-02](#2020-03-02)
  - [1. Robust Unsupervised Neural Machine Translation with Adversarial Training](#2020-03-02-1)
  - [2. Modeling Future Cost for Neural Machine Translation](#2020-03-02-2)
  - [3. TextBrewer: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing](#2020-03-02-3)
  - [4. Do all Roads Lead to Rome? Understanding the Role of Initialization in Iterative Back-Translation](#2020-03-02-4)
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



# 2020-03-04

[Return to Index](#Index)



<h2 id="2020-03-04-1">1. Understanding the Prediction Mechanism of Sentiments by XAI Visualization</h2>

Title: [Understanding the Prediction Mechanism of Sentiments by XAI Visualization](https://arxiv.org/abs/2003.01425)

Authors: [Chaehan So](https://arxiv.org/search/cs?searchtype=author&query=So%2C+C)

*(Submitted on 3 Mar 2020)*

> People often rely on online reviews to make purchase decisions. The present work aimed to gain an understanding of a machine learning model's prediction mechanism by visualizing the effect of sentiments extracted from online hotel reviews with explainable AI (XAI) methodology. Study 1 used the extracted sentiments as features to predict the review ratings by five machine learning algorithms (knn, CART decision trees, support vector machines, random forests, gradient boosting machines) and identified random forests as best algorithm. Study 2 analyzed the random forests model by feature importance and revealed the sentiments joy, disgust, positive and negative as the most predictive features. Furthermore, the visualization of additive variable attributions and their prediction distribution showed correct prediction in direction and effect size for the 5-star rating but partially wrong direction and insufficient effect size for the 1-star rating. These prediction details were corroborated by a what-if analysis for the four top features. In conclusion, the prediction mechanism of a machine learning model can be uncovered by visualization of particular observations. Comparing instances of contrasting ground truth values can draw a differential picture of the prediction mechanism and inform decisions for model improvement.

| Comments: | This is the author's prefinal version be published in conference proceedings: 4th International Conference on Natural Language Processing and Information Retrieval, Sejong, South Korea, 26-28 June, 2020, ACM |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Human-Computer Interaction (cs.HC)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | [arXiv:2003.01425](https://arxiv.org/abs/2003.01425) [cs.HC] |
|           | (or [arXiv:2003.01425v1](https://arxiv.org/abs/2003.01425v1) [cs.HC] for this version) |



<h2 id="2020-03-04-2">2. XGPT: Cross-modal Generative Pre-Training for Image Captioning</h2>

Title: [XGPT: Cross-modal Generative Pre-Training for Image Captioning](https://arxiv.org/abs/2003.01473)

Authors: [Qiaolin Xia](https://arxiv.org/search/cs?searchtype=author&query=Xia%2C+Q), [Haoyang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+H), [Nan Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan%2C+N), [Dongdong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+D), [Lei Ji](https://arxiv.org/search/cs?searchtype=author&query=Ji%2C+L), [Zhifang Sui](https://arxiv.org/search/cs?searchtype=author&query=Sui%2C+Z), [Edward Cui](https://arxiv.org/search/cs?searchtype=author&query=Cui%2C+E), [Taroon Bharti](https://arxiv.org/search/cs?searchtype=author&query=Bharti%2C+T), [Ming Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+M)

*(Submitted on 3 Mar 2020)*

> While many BERT-based cross-modal pre-trained models produce excellent results on downstream understanding tasks like image-text retrieval and VQA, they cannot be applied to generation tasks directly. In this paper, we propose XGPT, a new method of Cross-modal Generative Pre-Training for Image Captioning that is designed to pre-train text-to-image caption generators through three novel generation tasks, including Image-conditioned Masked Language Modeling (IMLM), Image-conditioned Denoising Autoencoding (IDA), and Text-conditioned Image Feature Generation (TIFG). As a result, the pre-trained XGPT can be fine-tuned without any task-specific architecture modifications to create state-of-the-art models for image captioning. Experiments show that XGPT obtains new state-of-the-art results on the benchmark datasets, including COCO Captions and Flickr30k Captions. We also use XGPT to generate new image captions as data augmentation for the image retrieval task and achieve significant improvement on all recall metrics.

| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2003.01473](https://arxiv.org/abs/2003.01473) [cs.CL] |
|           | (or [arXiv:2003.01473v1](https://arxiv.org/abs/2003.01473v1) [cs.CL] for this version) |





# 2020-03-02

[Return to Index](#Index)



<h2 id="2020-03-02-1">1. Robust Unsupervised Neural Machine Translation with Adversarial Training</h2>

Title: [Robust Unsupervised Neural Machine Translation with Adversarial Training](https://arxiv.org/abs/2002.12549)

Authors: [Haipeng Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+H), [Rui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+R), [Kehai Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+K), [Masao Utiyama](https://arxiv.org/search/cs?searchtype=author&query=Utiyama%2C+M), [Eiichiro Sumita](https://arxiv.org/search/cs?searchtype=author&query=Sumita%2C+E), [Tiejun Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+T)

*(Submitted on 28 Feb 2020)*

> Unsupervised neural machine translation (UNMT) has recently attracted great interest in the machine translation community, achieving only slightly worse results than supervised neural machine translation. However, in real-world scenarios, there usually exists minor noise in the input sentence and the neural translation system is sensitive to the small perturbations in the input, leading to poor performance. In this paper, we first define two types of noises and empirically show the effect of these noisy data on UNMT performance. Moreover, we propose adversarial training methods to improve the robustness of UNMT in the noisy scenario. To the best of our knowledge, this paper is the first work to explore the robustness of UNMT. Experimental results on several language pairs show that our proposed methods substantially outperform conventional UNMT systems in the noisy scenario.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2002.12549](https://arxiv.org/abs/2002.12549) [cs.CL] |
|           | (or [arXiv:2002.12549v1](https://arxiv.org/abs/2002.12549v1) [cs.CL] for this version) |





<h2 id="2020-03-02-2">2. Modeling Future Cost for Neural Machine Translation</h2>

Title: [Modeling Future Cost for Neural Machine Translation](https://arxiv.org/abs/2002.12558)

Authors: [Chaoqun Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan%2C+C), [Kehai Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+K), [Rui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+R), [Masao Utiyama](https://arxiv.org/search/cs?searchtype=author&query=Utiyama%2C+M), [Eiichiro Sumita](https://arxiv.org/search/cs?searchtype=author&query=Sumita%2C+E), [Conghui Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+C), [Tiejun Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+T)

*(Submitted on 28 Feb 2020)*

> Existing neural machine translation (NMT) systems utilize sequence-to-sequence neural networks to generate target translation word by word, and then make the generated word at each time-step and the counterpart in the references as consistent as possible. However, the trained translation model tends to focus on ensuring the accuracy of the generated target word at the current time-step and does not consider its future cost which means the expected cost of generating the subsequent target translation (i.e., the next target word). To respond to this issue, we propose a simple and effective method to model the future cost of each target word for NMT systems. In detail, a time-dependent future cost is estimated based on the current generated target word and its contextual information to boost the training of the NMT model. Furthermore, the learned future context representation at the current time-step is used to help the generation of the next target word in the decoding. Experimental results on three widely-used translation datasets, including the WMT14 German-to-English, WMT14 English-to-French, and WMT17 Chinese-to-English, show that the proposed approach achieves significant improvements over strong Transformer-based NMT baseline.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2002.12558](https://arxiv.org/abs/2002.12558) [cs.CL] |
|           | (or [arXiv:2002.12558v1](https://arxiv.org/abs/2002.12558v1) [cs.CL] for this version) |





<h2 id="2020-03-02-3">3. TextBrewer: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing</h2>

Title: [TextBrewer: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing](https://arxiv.org/abs/2002.12620)

Authors: [Ziqing Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Yiming Cui](https://arxiv.org/search/cs?searchtype=author&query=Cui%2C+Y), [Zhipeng Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Z), [Wanxiang Che](https://arxiv.org/search/cs?searchtype=author&query=Che%2C+W), [Ting Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+T), [Shijin Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Guoping Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+G)

*(Submitted on 28 Feb 2020)*

> In this paper, we introduce TextBrewer, an open-source knowledge distillation toolkit designed for natural language processing. It works with different neural network models and supports various kinds of tasks, such as text classification, reading comprehension, sequence labeling. TextBrewer provides a simple and uniform workflow that enables quick setup of distillation experiments with highly flexible configurations. It offers a set of predefined distillation methods and can be extended with custom code. As a case study, we use TextBrewer to distill BERT on several typical NLP tasks. With simple configuration, we achieve results that are comparable with or even higher than the state-of-the-art performance. Our toolkit is available through: [this http URL](http://textbrewer.hfl-rc.com/)

| Comments: | 8 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Neural and Evolutionary Computing (cs.NE) |
| Cite as:  | [arXiv:2002.12620](https://arxiv.org/abs/2002.12620) [cs.CL] |
|           | (or [arXiv:2002.12620v1](https://arxiv.org/abs/2002.12620v1) [cs.CL] for this version) |





<h2 id="2020-03-02-4">4. Do all Roads Lead to Rome? Understanding the Role of Initialization in Iterative Back-Translation</h2>

Title: [Do all Roads Lead to Rome? Understanding the Role of Initialization in Iterative Back-Translation](https://arxiv.org/abs/2002.12867)

Authors: [Mikel Artetxe](https://arxiv.org/search/cs?searchtype=author&query=Artetxe%2C+M), [Gorka Labaka](https://arxiv.org/search/cs?searchtype=author&query=Labaka%2C+G), [Noe Casas](https://arxiv.org/search/cs?searchtype=author&query=Casas%2C+N), [Eneko Agirre](https://arxiv.org/search/cs?searchtype=author&query=Agirre%2C+E)

*(Submitted on 28 Feb 2020)*

> Back-translation provides a simple yet effective approach to exploit monolingual corpora in Neural Machine Translation (NMT). Its iterative variant, where two opposite NMT models are jointly trained by alternately using a synthetic parallel corpus generated by the reverse model, plays a central role in unsupervised machine translation. In order to start producing sound translations and provide a meaningful training signal to each other, existing approaches rely on either a separate machine translation system to warm up the iterative procedure, or some form of pre-training to initialize the weights of the model. In this paper, we analyze the role that such initialization plays in iterative back-translation. Is the behavior of the final system heavily dependent on it? Or does iterative back-translation converge to a similar solution given any reasonable initialization? Through a series of empirical experiments over a diverse set of warmup systems, we show that, although the quality of the initial system does affect final performance, its effect is relatively small, as iterative back-translation has a strong tendency to convergence to a similar solution. As such, the margin of improvement left for the initialization method is narrow, suggesting that future research should focus more on improving the iterative mechanism itself.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2002.12867](https://arxiv.org/abs/2002.12867) [cs.CL] |
|           | (or [arXiv:2002.12867v1](https://arxiv.org/abs/2002.12867v1) [cs.CL] for this version) |



