# Daily arXiv: Machine Translation - May, 2021

# Index


- [2021-06-01](#2021-06-01)

  - [1. An Attention Free Transformer](#2021-06-01-1)
  - [2. LPF: A Language-Prior Feedback Objective Function for De-biased Visual Question Answering](#2021-06-01-2)
  - [3. Re-evaluating Word Mover's Distance](#2021-06-01-3)
  - [4. Memory-Efficient Differentiable Transformer Architecture Search](#2021-06-01-4)
  - [5. Why does CTC result in peaky behavior?](#2021-06-01-5)
  - [6. Grammatical Error Correction as GAN-like Sequence Labeling](#2021-06-01-6)
  - [7. Predictive Representation Learning for Language Modeling](#2021-06-01-7)
  - [8. Korean-English Machine Translation with Multiple Tokenization Strategy](#2021-06-01-8)
  - [9. Grammar Accuracy Evaluation (GAE): Quantifiable Intrinsic Evaluation of Machine Translation Models](#2021-06-01-9)
  - [10. NAS-BERT: Task-Agnostic and Adaptive-Size BERT Compression with Neural Architecture Search](#2021-06-01-10)
  - [11. Pre-training Universal Language Representation](#2021-06-01-11)
  - [12. Fast Nearest Neighbor Machine Translation](#2021-06-01-12)
  - [13. HIT: A Hierarchically Fused Deep Attention Network for Robust Code-mixed Language Representation](#2021-06-01-13)
  - [14. Attention Flows are Shapley Value Explanations](#2021-06-01-14)
  - [15. G-Transformer for Document-level Machine Translation](#2021-06-01-15)
  - [16. On Compositional Generalization of Neural Machine Translation](#2021-06-01-16)
  - [17. Transfer Learning for Sequence Generation: from Single-source to Multi-source](#2021-06-01-17)
  - [18. Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Models](#2021-06-01-18)
  - [19. Effective Batching for Recurrent Neural Network Grammars](#2021-06-01-19)
  - [20. Greedy Layer Pruning: Decreasing Inference Time of Transformer Models](#2021-06-01-20)
  - [21. Verdi: Quality Estimation and Error Detection for Bilingual](#2021-06-01-21)
  - [22. GWLAN: General Word-Level AutocompletioN for Computer-Aided Translation](#2021-06-01-22)
  - [23. Do Multilingual Neural Machine Translation Models Contain Language Pair Specific Attention Heads?](#2021-06-01-23)
  - [24. Adapting High-resource NMT Models to Translate Low-resource Related Languages without Parallel Data](#2021-06-01-24)
  - [25. Beyond Noise: Mitigating the Impact of Fine-grained Semantic Divergences on Neural Machine Translation](#2021-06-01-25)
- [Other Columns](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-06-01

[Return to Index](#Index)



<h2 id="2021-06-01-1">1. An Attention Free Transformer
</h2>

Title: [An Attention Free Transformer](https://arxiv.org/abs/2105.14103)

Authors: [Shuangfei Zhai](https://arxiv.org/search/cs?searchtype=author&query=Zhai%2C+S), [Walter Talbott](https://arxiv.org/search/cs?searchtype=author&query=Talbott%2C+W), [Nitish Srivastava](https://arxiv.org/search/cs?searchtype=author&query=Srivastava%2C+N), [Chen Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+C), [Hanlin Goh](https://arxiv.org/search/cs?searchtype=author&query=Goh%2C+H), [Ruixiang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+R), [Josh Susskind](https://arxiv.org/search/cs?searchtype=author&query=Susskind%2C+J)

> We introduce Attention Free Transformer (AFT), an efficient variant of Transformers that eliminates the need for dot product self attention. In an AFT layer, the key and value are first combined with a set of learned position biases, the result of which is multiplied with the query in an element-wise fashion. This new operation has a memory complexity linear w.r.t. both the context size and the dimension of features, making it compatible to both large input and model sizes. We also introduce AFT-local and AFT-conv, two model variants that take advantage of the idea of locality and spatial weight sharing while maintaining global connectivity. We conduct extensive experiments on two autoregressive modeling tasks (CIFAR10 and Enwik8) as well as an image recognition task (ImageNet-1K classification). We show that AFT demonstrates competitive performance on all the benchmarks, while providing excellent efficiency at the same time.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2105.14103](https://arxiv.org/abs/2105.14103) [cs.LG]** |
|           | (or **[arXiv:2105.14103v1](https://arxiv.org/abs/2105.14103v1) [cs.LG]** for this version) |



<h2 id="2021-06-01-2">2. LPF: A Language-Prior Feedback Objective Function for De-biased Visual Question Answering
</h2>

Title: [LPF: A Language-Prior Feedback Objective Function for De-biased Visual Question Answering](https://arxiv.org/abs/2105.14300)

Authors: [Zujie Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+Z), [Haifeng Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+H), [Jiaying Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+J)

> Most existing Visual Question Answering (VQA) systems tend to overly rely on language bias and hence fail to reason from the visual clue. To address this issue, we propose a novel Language-Prior Feedback (LPF) objective function, to re-balance the proportion of each answer's loss value in the total VQA loss. The LPF firstly calculates a modulating factor to determine the language bias using a question-only branch. Then, the LPF assigns a self-adaptive weight to each training sample in the training process. With this reweighting mechanism, the LPF ensures that the total VQA loss can be reshaped to a more balanced form. By this means, the samples that require certain visual information to predict will be efficiently used during training. Our method is simple to implement, model-agnostic, and end-to-end trainable. We conduct extensive experiments and the results show that the LPF (1) brings a significant improvement over various VQA models, (2) achieves competitive performance on the bias-sensitive VQA-CP v2 benchmark.

| Comments: | Accepted by ACM SIGIR 2021                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| DOI:      | [10.1145/3404835.3462981](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1145%2F3404835.3462981&v=a1843a78) |
| Cite as:  | **[arXiv:2105.14300](https://arxiv.org/abs/2105.14300) [cs.CV]** |
|           | (or **[arXiv:2105.14300v1](https://arxiv.org/abs/2105.14300v1) [cs.CV]** for this version) |





<h2 id="2021-06-01-3">3. Re-evaluating Word Mover's Distance
</h2>

Title: [Re-evaluating Word Mover's Distance](https://arxiv.org/abs/2105.14403)

Authors: [Ryoma Sato](https://arxiv.org/search/cs?searchtype=author&query=Sato%2C+R), [Makoto Yamada](https://arxiv.org/search/cs?searchtype=author&query=Yamada%2C+M), [Hisashi Kashima](https://arxiv.org/search/cs?searchtype=author&query=Kashima%2C+H)

> The word mover's distance (WMD) is a fundamental technique for measuring the similarity of two documents. As the crux of WMD, it can take advantage of the underlying geometry of the word space by employing an optimal transport formulation. The original study on WMD reported that WMD outperforms classical baselines such as bag-of-words (BOW) and TF-IDF by significant margins in various datasets. In this paper, we point out that the evaluation in the original study could be misleading. We re-evaluate the performances of WMD and the classical baselines and find that the classical baselines are competitive with WMD if we employ an appropriate preprocessing, i.e., L1 normalization. However, this result is not intuitive. WMD should be superior to BOW because WMD can take the underlying geometry into account, whereas BOW cannot. Our analysis shows that this is due to the high-dimensional nature of the underlying metric. We find that WMD in high-dimensional spaces behaves more similarly to BOW than in low-dimensional spaces due to the curse of dimensionality.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Information Retrieval (cs.IR) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2105.14403](https://arxiv.org/abs/2105.14403) [cs.LG]** |
|           | (or **[arXiv:2105.14403v1](https://arxiv.org/abs/2105.14403v1) [cs.LG]** for this version) |





<h2 id="2021-06-01-4">4. Memory-Efficient Differentiable Transformer Architecture Search
</h2>

Title: [Memory-Efficient Differentiable Transformer Architecture Search](https://arxiv.org/abs/2105.14669)

Authors: [Yuekai Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+Y), [Li Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+L), [Yelong Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+Y), [Zhihua Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F), [Weizhu Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+W)

> Differentiable architecture search (DARTS) is successfully applied in many vision tasks. However, directly using DARTS for Transformers is memory-intensive, which renders the search process infeasible. To this end, we propose a multi-split reversible network and combine it with DARTS. Specifically, we devise a backpropagation-with-reconstruction algorithm so that we only need to store the last layer's outputs. By relieving the memory burden for DARTS, it allows us to search with larger hidden size and more candidate operations. We evaluate the searched architecture on three sequence-to-sequence datasets, i.e., WMT'14 English-German, WMT'14 English-French, and WMT'14 English-Czech. Experimental results show that our network consistently outperforms standard Transformers across the tasks. Moreover, our method compares favorably with big-size Evolved Transformers, reducing search computation by an order of magnitude.

| Comments: | Accepted by Findings of ACL 2021                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2105.14669](https://arxiv.org/abs/2105.14669) [cs.LG]** |
|           | (or **[arXiv:2105.14669v1](https://arxiv.org/abs/2105.14669v1) [cs.LG]** for this version) |





<h2 id="2021-06-01-5">5. Why does CTC result in peaky behavior?
</h2>

Title: [Why does CTC result in peaky behavior?](https://arxiv.org/abs/2105.14849)

Authors: [Albert Zeyer](https://arxiv.org/search/cs?searchtype=author&query=Zeyer%2C+A), [Ralf Schlüter](https://arxiv.org/search/cs?searchtype=author&query=Schlüter%2C+R), [Hermann Ney](https://arxiv.org/search/cs?searchtype=author&query=Ney%2C+H)

> The peaky behavior of CTC models is well known experimentally. However, an understanding about why peaky behavior occurs is missing, and whether this is a good property. We provide a formal analysis of the peaky behavior and gradient descent convergence properties of the CTC loss and related training criteria. Our analysis provides a deep understanding why peaky behavior occurs and when it is suboptimal. On a simple example which should be trivial to learn for any model, we prove that a feed-forward neural network trained with CTC from uniform initialization converges towards peaky behavior with a 100% error rate. Our analysis further explains why CTC only works well together with the blank label. We further demonstrate that peaky behavior does not occur on other related losses including a label prior model, and that this improves convergence.

| Subjects: | **Machine Learning (cs.LG)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Neural and Evolutionary Computing (cs.NE); Sound (cs.SD); Audio and Speech Processing (eess.AS); Statistics Theory (math.ST) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2105.14849](https://arxiv.org/abs/2105.14849) [cs.LG]** |
|           | (or **[arXiv:2105.14849v1](https://arxiv.org/abs/2105.14849v1) [cs.LG]** for this version) |





<h2 id="2021-06-01-6">6. Grammatical Error Correction as GAN-like Sequence Labeling
</h2>

Title: [Grammatical Error Correction as GAN-like Sequence Labeling](https://arxiv.org/abs/2105.14209)

Authors: [Kevin Parnow](https://arxiv.org/search/cs?searchtype=author&query=Parnow%2C+K), [Zuchao Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Hai Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+H)

> In Grammatical Error Correction (GEC), sequence labeling models enjoy fast inference compared to sequence-to-sequence models; however, inference in sequence labeling GEC models is an iterative process, as sentences are passed to the model for multiple rounds of correction, which exposes the model to sentences with progressively fewer errors at each round. Traditional GEC models learn from sentences with fixed error rates. Coupling this with the iterative correction process causes a mismatch between training and inference that affects final performance. In order to address this mismatch, we propose a GAN-like sequence labeling model, which consists of a grammatical error detector as a discriminator and a grammatical error labeler with Gumbel-Softmax sampling as a generator. By sampling from real error distributions, our errors are more genuine compared to traditional synthesized GEC errors, thus alleviating the aforementioned mismatch and allowing for better training. Our results on several evaluation benchmarks demonstrate that our proposed approach is effective and improves the previous state-of-the-art baseline.

| Comments: | Accepted by ACL21, Findings                                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.14209](https://arxiv.org/abs/2105.14209) [cs.CL]** |
|           | (or **[arXiv:2105.14209v1](https://arxiv.org/abs/2105.14209v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-7">7. Predictive Representation Learning for Language Modeling
</h2>

Title: [Predictive Representation Learning for Language Modeling](https://arxiv.org/abs/2105.14214)

Authors: [Qingfeng Lan](https://arxiv.org/search/cs?searchtype=author&query=Lan%2C+Q), [Luke Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+L), [Martha White](https://arxiv.org/search/cs?searchtype=author&query=White%2C+M), [Alona Fyshe](https://arxiv.org/search/cs?searchtype=author&query=Fyshe%2C+A)

> To effectively perform the task of next-word prediction, long short-term memory networks (LSTMs) must keep track of many types of information. Some information is directly related to the next word's identity, but some is more secondary (e.g. discourse-level features or features of downstream words). Correlates of secondary information appear in LSTM representations even though they are not part of an \emph{explicitly} supervised prediction task. In contrast, in reinforcement learning (RL), techniques that explicitly supervise representations to predict secondary information have been shown to be beneficial. Inspired by that success, we propose Predictive Representation Learning (PRL), which explicitly constrains LSTMs to encode specific predictions, like those that might need to be learned implicitly. We show that PRL 1) significantly improves two strong language modeling methods, 2) converges more quickly, and 3) performs better when data is limited. Our work shows that explicitly encoding a simple predictive task facilitates the search for a more effective language model.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2105.14214](https://arxiv.org/abs/2105.14214) [cs.CL]** |
|           | (or **[arXiv:2105.14214v1](https://arxiv.org/abs/2105.14214v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-8">8. Korean-English Machine Translation with Multiple Tokenization Strategy
</h2>

Title: [Korean-English Machine Translation with Multiple Tokenization Strategy](https://arxiv.org/abs/2105.14274)

Authors: [Dojun Park](https://arxiv.org/search/cs?searchtype=author&query=Park%2C+D), [Youngjin Jang](https://arxiv.org/search/cs?searchtype=author&query=Jang%2C+Y), [Harksoo Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+H)

> This study was conducted to find out how tokenization methods affect the training results of machine translation models. In this work, character tokenization, morpheme tokenization, and BPE tokenization were applied to Korean as the source language and English as the target language respectively, and the comparison experiment was conducted by repeating 50,000 epochs of each 9 models using the Transformer neural network. As a result of measuring the BLEU scores of the experimental models, the model that applied BPE tokenization to Korean and morpheme tokenization to English recorded 35.73, showing the best performance.

| Comments: | KCC2021 Undergraduate/Junior Thesis Competition              |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.14274](https://arxiv.org/abs/2105.14274) [cs.CL]** |
|           | (or **[arXiv:2105.14274v1](https://arxiv.org/abs/2105.14274v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-9">9. Grammar Accuracy Evaluation (GAE): Quantifiable Intrinsic Evaluation of Machine Translation Models
</h2>

Title: [Grammar Accuracy Evaluation (GAE): Quantifiable Intrinsic Evaluation of Machine Translation Models](https://arxiv.org/abs/2105.14277)

Authors: [Dojun Park](https://arxiv.org/search/cs?searchtype=author&query=Park%2C+D), [Youngjin Jang](https://arxiv.org/search/cs?searchtype=author&query=Jang%2C+Y), [Harksoo Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+H)

> Intrinsic evaluation by humans for the performance of natural language generation models is conducted to overcome the fact that the quality of generated sentences cannot be fully represented by only extrinsic evaluation. Nevertheless, existing intrinsic evaluations have a large score deviation according to the evaluator's criteria. In this paper, we propose Grammar Accuracy Evaluation (GAE) that can provide specific evaluating criteria. As a result of analyzing the quality of machine translation by BLEU and GAE, it was confirmed that the BLEU score does not represent the absolute performance of machine translation models and that GAE compensates for the shortcomings of BLEU with a flexible evaluation on alternative synonyms and changes in sentence structure.

| Comments: | Journal of KIISE                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.14277](https://arxiv.org/abs/2105.14277) [cs.CL]** |
|           | (or **[arXiv:2105.14277v1](https://arxiv.org/abs/2105.14277v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-10">10. NAS-BERT: Task-Agnostic and Adaptive-Size BERT Compression with Neural Architecture Search
</h2>

Title: [NAS-BERT: Task-Agnostic and Adaptive-Size BERT Compression with Neural Architecture Search](https://arxiv.org/abs/2105.14444)

Authors: [Jin Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+J), [Xu Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+X), [Renqian Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+R), [Kaitao Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+K), [Jian Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J), [Tao Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+T), [Tie-Yan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+T)

> While pre-trained language models (e.g., BERT) have achieved impressive results on different natural language processing tasks, they have large numbers of parameters and suffer from big computational and memory costs, which make them difficult for real-world deployment. Therefore, model compression is necessary to reduce the computation and memory cost of pre-trained models. In this work, we aim to compress BERT and address the following two challenging practical issues: (1) The compression algorithm should be able to output multiple compressed models with different sizes and latencies, in order to support devices with different memory and latency limitations; (2) The algorithm should be downstream task agnostic, so that the compressed models are generally applicable for different downstream tasks. We leverage techniques in neural architecture search (NAS) and propose NAS-BERT, an efficient method for BERT compression. NAS-BERT trains a big supernet on a search space containing a variety of architectures and outputs multiple compressed models with adaptive sizes and latency. Furthermore, the training of NAS-BERT is conducted on standard self-supervised pre-training tasks (e.g., masked language model) and does not depend on specific downstream tasks. Thus, the compressed models can be used across various downstream tasks. The technical challenge of NAS-BERT is that training a big supernet on the pre-training task is extremely costly. We employ several techniques including block-wise search, search space pruning, and performance approximation to improve search efficiency and accuracy. Extensive experiments on GLUE and SQuAD benchmark datasets demonstrate that NAS-BERT can find lightweight models with better accuracy than previous approaches, and can be directly applied to different downstream tasks with adaptive model sizes for different requirements of memory or latency.

| Comments: | Accepted by KDD 2021                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| DOI:      | [10.1145/3447548.3467262](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1145%2F3447548.3467262&v=2e2b005f) |
| Cite as:  | **[arXiv:2105.14444](https://arxiv.org/abs/2105.14444) [cs.CL]** |
|           | (or **[arXiv:2105.14444v1](https://arxiv.org/abs/2105.14444v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-11">11. Pre-training Universal Language Representation
</h2>

Title: [Pre-training Universal Language Representation](https://arxiv.org/abs/2105.14478)

Authors: [Yian Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Hai Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+H)

> Despite the well-developed cut-edge representation learning for language, most language representation models usually focus on specific levels of linguistic units. This work introduces universal language representation learning, i.e., embeddings of different levels of linguistic units or text with quite diverse lengths in a uniform vector space. We propose the training objective MiSAD that utilizes meaningful n-grams extracted from large unlabeled corpus by a simple but effective algorithm for pre-trained language models. Then we empirically verify that well designed pre-training scheme may effectively yield universal language representation, which will bring great convenience when handling multiple layers of linguistic objects in a unified way. Especially, our model achieves the highest accuracy on analogy tasks in different language levels and significantly improves the performance on downstream tasks in the GLUE benchmark and a question answering dataset.

| Comments: | Accepted by ACL-IJCNLP 2021 main conference                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2105.14478](https://arxiv.org/abs/2105.14478) [cs.CL]** |
|           | (or **[arXiv:2105.14478v1](https://arxiv.org/abs/2105.14478v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-12">12. Fast Nearest Neighbor Machine Translation
</h2>

Title: [Fast Nearest Neighbor Machine Translation](https://arxiv.org/abs/2105.14528)

Authors: [Yuxian Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+Y), [Xiaoya Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+X), [Xiayu Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+X), [Fei Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+F), [Xiaofei Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+X), [Tianwei Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+T), [Jiwei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J)

> Though nearest neighbor Machine Translation (kNN-MT) \cite{khandelwal2020nearest} has proved to introduce significant performance boosts over standard neural MT systems, it is prohibitively slow since it uses the entire reference corpus as the datastore for the nearest neighbor search. This means each step for each beam in the beam search has to search over the entire reference corpus. kNN-MT is thus two-order slower than vanilla MT models, making it hard to be applied to real-world applications, especially online services. In this work, we propose Fast kNN-MT to address this issue. Fast kNN-MT constructs a significantly smaller datastore for the nearest neighbor search: for each word in a source sentence, Fast kNN-MT first selects its nearest token-level neighbors, which is limited to tokens that are the same as the query token. Then at each decoding step, in contrast to using the entire corpus as the datastore, the search space is limited to target tokens corresponding to the previously selected reference source tokens. This strategy avoids search through the whole datastore for nearest neighbors and drastically improves decoding efficiency. Without loss of performance, Fast kNN-MT is two-order faster than kNN-MT, and is only two times slower than the standard NMT model. Fast kNN-MT enables the practical use of kNN-MT systems in real-world MT applications.\footnote{Code is available at \url{[this https URL](https://github.com/ShannonAI/fast-knn-nmt).}}

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2105.14528](https://arxiv.org/abs/2105.14528) [cs.CL]** |
|           | (or **[arXiv:2105.14528v1](https://arxiv.org/abs/2105.14528v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-13">13. HIT: A Hierarchically Fused Deep Attention Network for Robust Code-mixed Language Representation
</h2>


Title: [HIT: A Hierarchically Fused Deep Attention Network for Robust Code-mixed Language Representation](https://arxiv.org/abs/2105.14600)

Authors: [Ayan Sengupta](https://arxiv.org/search/cs?searchtype=author&query=Sengupta%2C+A), [Sourabh Kumar Bhattacharjee](https://arxiv.org/search/cs?searchtype=author&query=Bhattacharjee%2C+S+K), [Tanmoy Chakraborty](https://arxiv.org/search/cs?searchtype=author&query=Chakraborty%2C+T), [Md Shad Akhtar](https://arxiv.org/search/cs?searchtype=author&query=Akhtar%2C+M+S)

> Understanding linguistics and morphology of resource-scarce code-mixed texts remains a key challenge in text processing. Although word embedding comes in handy to support downstream tasks for low-resource languages, there are plenty of scopes in improving the quality of language representation particularly for code-mixed languages. In this paper, we propose HIT, a robust representation learning method for code-mixed texts. HIT is a hierarchical transformer-based framework that captures the semantic relationship among words and hierarchically learns the sentence-level semantics using a fused attention mechanism. HIT incorporates two attention modules, a multi-headed self-attention and an outer product attention module, and computes their weighted sum to obtain the attention weights. Our evaluation of HIT on one European (Spanish) and five Indic (Hindi, Bengali, Tamil, Telugu, and Malayalam) languages across four NLP tasks on eleven datasets suggests significant performance improvement against various state-of-the-art systems. We further show the adaptability of learned representation across tasks in a transfer learning setup (with and without fine-tuning).

| Comments: | 15 pages, 13 tables, 6 Figures. Accepted at ACL-IJCNLP-2021 (Findings) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.14600](https://arxiv.org/abs/2105.14600) [cs.CL]** |
|           | (or **[arXiv:2105.14600v1](https://arxiv.org/abs/2105.14600v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-14">14. Attention Flows are Shapley Value Explanations
</h2>


Title: [Attention Flows are Shapley Value Explanations](https://arxiv.org/abs/2105.14652)

Authors: [Kawin Ethayarajh](https://arxiv.org/search/cs?searchtype=author&query=Ethayarajh%2C+K), [Dan Jurafsky](https://arxiv.org/search/cs?searchtype=author&query=Jurafsky%2C+D)

> Shapley Values, a solution to the credit assignment problem in cooperative game theory, are a popular type of explanation in machine learning, having been used to explain the importance of features, embeddings, and even neurons. In NLP, however, leave-one-out and attention-based explanations still predominate. Can we draw a connection between these different methods? We formally prove that -- save for the degenerate case -- attention weights and leave-one-out values cannot be Shapley Values. Attention flow is a post-processed variant of attention weights obtained by running the max-flow algorithm on the attention graph. Perhaps surprisingly, we prove that attention flows are indeed Shapley Values, at least at the layerwise level. Given the many desirable theoretical qualities of Shapley Values -- which has driven their adoption among the ML community -- we argue that NLP practitioners should, when possible, adopt attention flow explanations alongside more traditional ones.

| Comments: | ACL 2021                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.14652](https://arxiv.org/abs/2105.14652) [cs.CL]** |
|           | (or **[arXiv:2105.14652v1](https://arxiv.org/abs/2105.14652v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-15">15. G-Transformer for Document-level Machine Translation
</h2>


Title: [G-Transformer for Document-level Machine Translation](https://arxiv.org/abs/2105.14761)

Authors: [Guangsheng Bao](https://arxiv.org/search/cs?searchtype=author&query=Bao%2C+G), [Yue Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Zhiyang Teng](https://arxiv.org/search/cs?searchtype=author&query=Teng%2C+Z), [Boxing Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+B), [Weihua Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+W)

> Document-level MT models are still far from satisfactory. Existing work extend translation unit from single sentence to multiple sentences. However, study shows that when we further enlarge the translation unit to a whole document, supervised training of Transformer can fail. In this paper, we find such failure is not caused by overfitting, but by sticking around local minima during training. Our analysis shows that the increased complexity of target-to-source attention is a reason for the failure. As a solution, we propose G-Transformer, introducing locality assumption as an inductive bias into Transformer, reducing the hypothesis space of the attention from target to source. Experiments show that G-Transformer converges faster and more stably than Transformer, achieving new state-of-the-art BLEU scores for both non-pretraining and pre-training settings on three benchmark datasets.

| Comments: | Accepted by ACL2021 main track                               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2105.14761](https://arxiv.org/abs/2105.14761) [cs.CL]** |
|           | (or **[arXiv:2105.14761v1](https://arxiv.org/abs/2105.14761v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-16">16. On Compositional Generalization of Neural Machine Translation
</h2>


Title: [On Compositional Generalization of Neural Machine Translation](https://arxiv.org/abs/2105.14802)

Authors: [Yafu Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Yongjing Yin](https://arxiv.org/search/cs?searchtype=author&query=Yin%2C+Y), [Yulong Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Yue Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y)

> Modern neural machine translation (NMT) models have achieved competitive performance in standard benchmarks such as WMT. However, there still exist significant issues such as robustness, domain generalization, etc. In this paper, we study NMT models from the perspective of compositional generalization by building a benchmark dataset, CoGnition, consisting of 216k clean and consistent sentence pairs. We quantitatively analyze effects of various factors using compound translation error rate, then demonstrate that the NMT model fails badly on compositional generalization, although it performs remarkably well under traditional metrics.

| Comments: | To appear at the ACL 2021 main conference                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2105.14802](https://arxiv.org/abs/2105.14802) [cs.CL]** |
|           | (or **[arXiv:2105.14802v1](https://arxiv.org/abs/2105.14802v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-17">17. Transfer Learning for Sequence Generation: from Single-source to Multi-source
</h2>


Title: [Transfer Learning for Sequence Generation: from Single-source to Multi-source](https://arxiv.org/abs/2105.14809)

Authors: [Xuancheng Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+X), [Jingfang Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+J), [Maosong Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+M), [Yang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y)

> Multi-source sequence generation (MSG) is an important kind of sequence generation tasks that takes multiple sources, including automatic post-editing, multi-source translation, multi-document summarization, etc. As MSG tasks suffer from the data scarcity problem and recent pretrained models have been proven to be effective for low-resource downstream tasks, transferring pretrained sequence-to-sequence models to MSG tasks is essential. Although directly finetuning pretrained models on MSG tasks and concatenating multiple sources into a single long sequence is regarded as a simple method to transfer pretrained models to MSG tasks, we conjecture that the direct finetuning method leads to catastrophic forgetting and solely relying on pretrained self-attention layers to capture cross-source information is not sufficient. Therefore, we propose a two-stage finetuning method to alleviate the pretrain-finetune discrepancy and introduce a novel MSG model with a fine encoder to learn better representations in MSG tasks. Experiments show that our approach achieves new state-of-the-art results on the WMT17 APE task and multi-source translation task using the WMT14 test set. When adapted to document-level translation, our framework outperforms strong baselines significantly.

| Comments: | ACL2021 main track long paper                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2105.14809](https://arxiv.org/abs/2105.14809) [cs.CL]** |
|           | (or **[arXiv:2105.14809v1](https://arxiv.org/abs/2105.14809v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-18">18. Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Models
</h2>


Title: [Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Models](https://arxiv.org/abs/2105.14813)

Authors: [Chong Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+C), [Cenyuan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+C), [Xiaoqing Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+X), [Xuanjing Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+X)

> A sequence-to-sequence learning with neural networks has empirically proven to be an effective framework for Chinese Spelling Correction (CSC), which takes a sentence with some spelling errors as input and outputs the corrected one. However, CSC models may fail to correct spelling errors covered by the confusion sets, and also will encounter unseen ones. We propose a method, which continually identifies the weak spots of a model to generate more valuable training instances, and apply a task-specific pre-training strategy to enhance the model. The generated adversarial examples are gradually added to the training set. Experimental results show that such an adversarial training method combined with the pretraining strategy can improve both the generalization and robustness of multiple CSC models across three different datasets, achieving stateof-the-art performance for CSC task.

| Comments: | Accepted by ACL 2021                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.14813](https://arxiv.org/abs/2105.14813) [cs.CL]** |
|           | (or **[arXiv:2105.14813v1](https://arxiv.org/abs/2105.14813v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-19">19. Effective Batching for Recurrent Neural Network Grammars
</h2>


Title: [Effective Batching for Recurrent Neural Network Grammars](https://arxiv.org/abs/2105.14822)

Authors: [Hiroshi Noji](https://arxiv.org/search/cs?searchtype=author&query=Noji%2C+H), [Yohei Oseki](https://arxiv.org/search/cs?searchtype=author&query=Oseki%2C+Y)

> As a language model that integrates traditional symbolic operations and flexible neural representations, recurrent neural network grammars (RNNGs) have attracted great attention from both scientific and engineering perspectives. However, RNNGs are known to be harder to scale due to the difficulty of batched training. In this paper, we propose effective batching for RNNGs, where every operation is computed in parallel with tensors across multiple sentences. Our PyTorch implementation effectively employs a GPU and achieves x6 speedup compared to the existing C++ DyNet implementation with model-independent auto-batching. Moreover, our batched RNNG also accelerates inference and achieves x20-150 speedup for beam search depending on beam sizes. Finally, we evaluate syntactic generalization performance of the scaled RNNG against the LSTM baseline, based on the large training data of 100M tokens from English Wikipedia and the broad-coverage targeted syntactic evaluation benchmark. Our RNNG implementation is available at [this https URL](https://github.com/aistairc/rnng-pytorch/).

| Comments: | Findings of ACL: ACL-IJCNLP 2021                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.14822](https://arxiv.org/abs/2105.14822) [cs.CL]** |
|           | (or **[arXiv:2105.14822v1](https://arxiv.org/abs/2105.14822v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-20">20. Greedy Layer Pruning: Decreasing Inference Time of Transformer Models
</h2>


Title: [Greedy Layer Pruning: Decreasing Inference Time of Transformer Models](https://arxiv.org/abs/2105.14839)

Authors: [David Peer](https://arxiv.org/search/cs?searchtype=author&query=Peer%2C+D), [Sebastian Stabinger](https://arxiv.org/search/cs?searchtype=author&query=Stabinger%2C+S), [Stefan Engl](https://arxiv.org/search/cs?searchtype=author&query=Engl%2C+S), [Antonio Rodriguez-Sanchez](https://arxiv.org/search/cs?searchtype=author&query=Rodriguez-Sanchez%2C+A)

> Fine-tuning transformer models after unsupervised pre-training reaches a very high performance on many different NLP tasks. Unfortunately, transformers suffer from long inference times which greatly increases costs in production and is a limiting factor for the deployment into embedded devices. One possible solution is to use knowledge distillation, which solves this problem by transferring information from large teacher models to smaller student models, but as it needs an additional expensive pre-training phase, this solution is computationally expensive and can be financially prohibitive for smaller academic research groups. Another solution is to use layer-wise pruning methods, which reach high compression rates for transformer models and avoids the computational load of the pre-training distillation stage. The price to pay is that the performance of layer-wise pruning algorithms is not on par with state-of-the-art knowledge distillation methods. In this paper, greedy layer pruning (GLP) is introduced to (1) outperform current state-of-the-art for layer-wise pruning (2) close the performance gap when compared to knowledge distillation, while (3) using only a modest budget. More precisely, with the methodology presented it is possible to prune and evaluate competitive models on the whole GLUE benchmark with a budget of just $300. Our source code is available on [this https URL](https://github.com/deepopinion/greedy-layer-pruning).

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2105.14839](https://arxiv.org/abs/2105.14839) [cs.CL]** |
|           | (or **[arXiv:2105.14839v1](https://arxiv.org/abs/2105.14839v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-21">21. Verdi: Quality Estimation and Error Detection for Bilingual
</h2>


Title: [Verdi: Quality Estimation and Error Detection for Bilingual](https://arxiv.org/abs/2105.14878)

Authors: [Mingjun Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+M), [Haijiang Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+H), [Di Niu](https://arxiv.org/search/cs?searchtype=author&query=Niu%2C+D), [Zixuan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Z), [Xiaoli Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X)

> Translation Quality Estimation is critical to reducing post-editing efforts in machine translation and to cross-lingual corpus cleaning. As a research problem, quality estimation (QE) aims to directly estimate the quality of translation in a given pair of source and target sentences, and highlight the words that need corrections, without referencing to golden translations. In this paper, we propose Verdi, a novel framework for word-level and sentence-level post-editing effort estimation for bilingual corpora. Verdi adopts two word predictors to enable diverse features to be extracted from a pair of sentences for subsequent quality estimation, including a transformer-based neural machine translation (NMT) model and a pre-trained cross-lingual language model (XLM). We exploit the symmetric nature of bilingual corpora and apply model-level dual learning in the NMT predictor, which handles a primal task and a dual task simultaneously with weight sharing, leading to stronger context prediction ability than single-direction NMT models. By taking advantage of the dual learning scheme, we further design a novel feature to directly encode the translated target information without relying on the source context. Extensive experiments conducted on WMT20 QE tasks demonstrate that our method beats the winner of the competition and outperforms other baseline methods by a great margin. We further use the sentence-level scores provided by Verdi to clean a parallel corpus and observe benefits on both model performance and training efficiency.

| Comments: | Accepted by The Web Conference 2021                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2105.14878](https://arxiv.org/abs/2105.14878) [cs.CL]** |
|           | (or **[arXiv:2105.14878v1](https://arxiv.org/abs/2105.14878v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-22">22. GWLAN: General Word-Level AutocompletioN for Computer-Aided Translation
</h2>


Title: [GWLAN: General Word-Level AutocompletioN for Computer-Aided Translation](https://arxiv.org/abs/2105.14913)

Authors: [Huayang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+H), [Lemao Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+L), [Guoping Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+G), [Shuming Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+S)

> Computer-aided translation (CAT), the use of software to assist a human translator in the translation process, has been proven to be useful in enhancing the productivity of human translators. Autocompletion, which suggests translation results according to the text pieces provided by human translators, is a core function of CAT. There are two limitations in previous research in this line. First, most research works on this topic focus on sentence-level autocompletion (i.e., generating the whole translation as a sentence based on human input), but word-level autocompletion is under-explored so far. Second, almost no public benchmarks are available for the autocompletion task of CAT. This might be among the reasons why research progress in CAT is much slower compared to automatic MT. In this paper, we propose the task of general word-level autocompletion (GWLAN) from a real-world CAT scenario, and construct the first public benchmark to facilitate research in this topic. In addition, we propose an effective method for GWLAN and compare it with several strong baselines. Experiments demonstrate that our proposed method can give significantly more accurate predictions than the baseline methods on our benchmark datasets.

| Comments: | Accepted into the main conference of ACL 2021. arXiv admin note: text overlap with [arXiv:2105.13072](https://arxiv.org/abs/2105.13072) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.14913](https://arxiv.org/abs/2105.14913) [cs.CL]** |
|           | (or **[arXiv:2105.14913v1](https://arxiv.org/abs/2105.14913v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-23">23. Do Multilingual Neural Machine Translation Models Contain Language Pair Specific Attention Heads?
</h2>


Title: [Do Multilingual Neural Machine Translation Models Contain Language Pair Specific Attention Heads?](https://arxiv.org/abs/2105.14940)

Authors: [Zae Myung Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+Z+M), [Laurent Besacier](https://arxiv.org/search/cs?searchtype=author&query=Besacier%2C+L), [Vassilina Nikoulina](https://arxiv.org/search/cs?searchtype=author&query=Nikoulina%2C+V), [Didier Schwab](https://arxiv.org/search/cs?searchtype=author&query=Schwab%2C+D)

> Recent studies on the analysis of the multilingual representations focus on identifying whether there is an emergence of language-independent representations, or whether a multilingual model partitions its weights among different languages. While most of such work has been conducted in a "black-box" manner, this paper aims to analyze individual components of a multilingual neural translation (NMT) model. In particular, we look at the encoder self-attention and encoder-decoder attention heads (in a many-to-one NMT model) that are more specific to the translation of a certain language pair than others by (1) employing metrics that quantify some aspects of the attention weights such as "variance" or "confidence", and (2) systematically ranking the importance of attention heads with respect to translation quality. Experimental results show that surprisingly, the set of most important attention heads are very similar across the language pairs and that it is possible to remove nearly one-third of the less important heads without hurting the translation quality greatly.

| Comments: | 10 pages, accepted at Findings of ACL 2021 (short)           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2105.14940](https://arxiv.org/abs/2105.14940) [cs.CL]** |
|           | (or **[arXiv:2105.14940v1](https://arxiv.org/abs/2105.14940v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-24">24. Adapting High-resource NMT Models to Translate Low-resource Related Languages without Parallel Data
</h2>


Title: [Adapting High-resource NMT Models to Translate Low-resource Related Languages without Parallel Data](https://arxiv.org/abs/2105.15071)

Authors: [Wei-Jen Ko](https://arxiv.org/search/cs?searchtype=author&query=Ko%2C+W), [Ahmed El-Kishky](https://arxiv.org/search/cs?searchtype=author&query=El-Kishky%2C+A), [Adithya Renduchintala](https://arxiv.org/search/cs?searchtype=author&query=Renduchintala%2C+A), [Vishrav Chaudhary](https://arxiv.org/search/cs?searchtype=author&query=Chaudhary%2C+V), [Naman Goyal](https://arxiv.org/search/cs?searchtype=author&query=Goyal%2C+N), [Francisco Guzmán](https://arxiv.org/search/cs?searchtype=author&query=Guzmán%2C+F), [Pascale Fung](https://arxiv.org/search/cs?searchtype=author&query=Fung%2C+P), [Philipp Koehn](https://arxiv.org/search/cs?searchtype=author&query=Koehn%2C+P), [Mona Diab](https://arxiv.org/search/cs?searchtype=author&query=Diab%2C+M)

> The scarcity of parallel data is a major obstacle for training high-quality machine translation systems for low-resource languages. Fortunately, some low-resource languages are linguistically related or similar to high-resource languages; these related languages may share many lexical or syntactic structures. In this work, we exploit this linguistic overlap to facilitate translating to and from a low-resource language with only monolingual data, in addition to any parallel data in the related high-resource language. Our method, NMT-Adapt, combines denoising autoencoding, back-translation and adversarial objectives to utilize monolingual data for low-resource adaptation. We experiment on 7 languages from three different language families and show that our technique significantly improves translation into low-resource language compared to other translation baselines.

| Comments: | ACL 2021                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.15071](https://arxiv.org/abs/2105.15071) [cs.CL]** |
|           | (or **[arXiv:2105.15071v1](https://arxiv.org/abs/2105.15071v1) [cs.CL]** for this version) |





<h2 id="2021-06-01-25">25. Beyond Noise: Mitigating the Impact of Fine-grained Semantic Divergences on Neural Machine Translation
</h2>


Title: [Beyond Noise: Mitigating the Impact of Fine-grained Semantic Divergences on Neural Machine Translation](https://arxiv.org/abs/2105.15087)

Authors: [Eleftheria Briakou](https://arxiv.org/search/cs?searchtype=author&query=Briakou%2C+E), [Marine Carpuat](https://arxiv.org/search/cs?searchtype=author&query=Carpuat%2C+M)

> While it has been shown that Neural Machine Translation (NMT) is highly sensitive to noisy parallel training samples, prior work treats all types of mismatches between source and target as noise. As a result, it remains unclear how samples that are mostly equivalent but contain a small number of semantically divergent tokens impact NMT training. To close this gap, we analyze the impact of different types of fine-grained semantic divergences on Transformer models. We show that models trained on synthetic divergences output degenerated text more frequently and are less confident in their predictions. Based on these findings, we introduce a divergent-aware NMT framework that uses factors to help NMT recover from the degradation caused by naturally occurring divergences, improving both translation quality and model calibration on EN-FR tasks.

| Comments: | ACL 2021                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.15087](https://arxiv.org/abs/2105.15087) [cs.CL]** |
|           | (or **[arXiv:2105.15087v1](https://arxiv.org/abs/2105.15087v1) [cs.CL]** for this version) |

