# Daily arXiv: Machine Translation - October, 2020

# Index


- [2020-11-03](#2020-11-03)

  - [1. COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning](#2020-11-03-1)
  - [2. The 2020s Political Economy of Machine Translation](#2020-11-03-2)
  - [3. Streaming Simultaneous Speech Translation with Augmented Memory Transformer](#2020-11-03-3)
  - [4. Joint Masked CPC and CTC Training for ASR](#2020-11-03-4)
  - [5. Evaluating Bias In Dutch Word Embeddings](#2020-11-03-5)
  - [6. Targeted Poisoning Attacks on Black-Box Neural Machine Translation](#2020-11-03-6)
  - [7. Investigating Catastrophic Forgetting During Continual Training for Neural Machine Translation](#2020-11-03-7)
  - [8. Dual-decoder Transformer for Joint Automatic Speech Recognition and Multilingual Speech Translation](#2020-11-03-8)
  - [9. Context-Aware Cross-Attention for Non-Autoregressive Translation](#2020-11-03-9)
  - [10. Emergent Communication Pretraining for Few-Shot Machine Translation](#2020-11-03-10)
  - [11. How Far Does BERT Look At:Distance-based Clustering and Analysis of BERTś Attention](#2020-11-03-11)
  - [12. Enabling Zero-shot Multilingual Spoken Language Translation with Language-Specific Encoders and Decoders](#2020-11-03-12)
  - [13. The Devil is in the Details: Evaluating Limitations of Transformer-based Methods for Granular Tasks](#2020-11-03-13)
- [2020-11-02](#2020-11-02)
- [1. VECO: Variable Encoder-decoder Pre-training for Cross-lingual Understanding and Generation](#2020-11-02-1)
  - [2. Domain-Specific Lexical Grounding in Noisy Visual-Textual Documents](#2020-11-02-2)
- [Other Columns](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2020-11-03

[Return to Index](#Index)



<h2 id="2020-11-03-1">1. COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning</h2>

Title: [COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning](https://arxiv.org/abs/2011.00597)

Authors: [Simon Ging](https://arxiv.org/search/cs?searchtype=author&query=Ging%2C+S) (1), [Mohammadreza Zolfaghari](https://arxiv.org/search/cs?searchtype=author&query=Zolfaghari%2C+M) (1), [Hamed Pirsiavash](https://arxiv.org/search/cs?searchtype=author&query=Pirsiavash%2C+H) (2), [Thomas Brox](https://arxiv.org/search/cs?searchtype=author&query=Brox%2C+T) (1) ((1) University of Freiburg, (2) University of Maryland Baltimore County)

> Many real-world video-text tasks involve different levels of granularity, such as frames and words, clip and sentences or videos and paragraphs, each with distinct semantics. In this paper, we propose a Cooperative hierarchical Transformer (COOT) to leverage this hierarchy information and model the interactions between different levels of granularity and different modalities. The method consists of three major components: an attention-aware feature aggregation layer, which leverages the local temporal context (intra-level, e.g., within a clip), a contextual transformer to learn the interactions between low-level and high-level semantics (inter-level, e.g. clip-video, sentence-paragraph), and a cross-modal cycle-consistency loss to connect video and text. The resulting method compares favorably to the state of the art on several benchmarks while having few parameters. All code is available open-source at [this https URL](https://github.com/gingsi/coot-videotext)

| Comments:    | 27 pages, 5 figures, 19 tables. To be published in the 34th conference on Neural Information Processing Systems (NeurIPS 2020). The first two authors contributed equally to this work |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG) |
| ACM classes: | I.2.7; I.2.10                                                |
| Cite as:     | **[arXiv:2011.00597](https://arxiv.org/abs/2011.00597) [cs.CV]** |
|              | (or **[arXiv:2011.00597v1](https://arxiv.org/abs/2011.00597v1) [cs.CV]** for this version) |





<h2 id="2020-11-03-2">2. The 2020s Political Economy of Machine Translation</h2>

Title: [The 2020s Political Economy of Machine Translation](https://arxiv.org/abs/2011.01007)

Authors: [Steven Weber](https://arxiv.org/search/cs?searchtype=author&query=Weber%2C+S)

> This paper explores the hypothesis that the diversity of human languages, right now a barrier to interoperability in communication and trade, will become significantly less of a barrier as machine translation technologies are deployed over the next several years.But this new boundary-breaking technology does not reduce all boundaries equally, and it creates new challenges for the distribution of ideas and thus for innovation and economic growth.

| Comments: | 42 pages, 0 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computers and Society (cs.CY)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2011.01007](https://arxiv.org/abs/2011.01007) [cs.CY]** |
|           | (or **[arXiv:2011.01007v1](https://arxiv.org/abs/2011.01007v1) [cs.CY]** for this version) |





<h2 id="2020-11-03-3">3. Streaming Simultaneous Speech Translation with Augmented Memory Transformer</h2>

Title: [Streaming Simultaneous Speech Translation with Augmented Memory Transformer](https://arxiv.org/abs/2011.00033)

Authors: [Xutai Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+X), [Yongqiang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Mohammad Javad Dousti](https://arxiv.org/search/cs?searchtype=author&query=Dousti%2C+M+J), [Philipp Koehn](https://arxiv.org/search/cs?searchtype=author&query=Koehn%2C+P), [Juan Pino](https://arxiv.org/search/cs?searchtype=author&query=Pino%2C+J)

> Transformer-based models have achieved state-of-the-art performance on speech translation tasks. However, the model architecture is not efficient enough for streaming scenarios since self-attention is computed over an entire input sequence and the computational cost grows quadratically with the length of the input sequence. Nevertheless, most of the previous work on simultaneous speech translation, the task of generating translations from partial audio input, ignores the time spent in generating the translation when analyzing the latency. With this assumption, a system may have good latency quality trade-offs but be inapplicable in real-time scenarios. In this paper, we focus on the task of streaming simultaneous speech translation, where the systems are not only capable of translating with partial input but are also able to handle very long or continuous input. We propose an end-to-end transformer-based sequence-to-sequence model, equipped with an augmented memory transformer encoder, which has shown great success on the streaming automatic speech recognition task with hybrid or transducer-based models. We conduct an empirical evaluation of the proposed model on segment, context and memory sizes and we compare our approach to a transformer with a unidirectional mask.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2011.00033](https://arxiv.org/abs/2011.00033) [cs.CL]** |
|           | (or **[arXiv:2011.00033v1](https://arxiv.org/abs/2011.00033v1) [cs.CL]** for this version) |





<h2 id="2020-11-03-4">4. Joint Masked CPC and CTC Training for ASR</h2>

Title: [Joint Masked CPC and CTC Training for ASR](https://arxiv.org/abs/2011.00093)

Authors: [Chaitanya Talnikar](https://arxiv.org/search/cs?searchtype=author&query=Talnikar%2C+C), [Tatiana Likhomanenko](https://arxiv.org/search/cs?searchtype=author&query=Likhomanenko%2C+T), [Ronan Collobert](https://arxiv.org/search/cs?searchtype=author&query=Collobert%2C+R), [Gabriel Synnaeve](https://arxiv.org/search/cs?searchtype=author&query=Synnaeve%2C+G)

> Self-supervised learning (SSL) has shown promise in learning representations of audio that are useful for automatic speech recognition (ASR). But, training SSL models like wav2vec~2.0 requires a two-stage pipeline. In this paper we demonstrate a single-stage training of ASR models that can utilize both unlabeled and labeled data. During training, we alternately minimize two losses: an unsupervised masked Contrastive Predictive Coding (CPC) loss and the supervised audio-to-text alignment loss Connectionist Temporal Classification (CTC). We show that this joint training method directly optimizes performance for the downstream ASR task using unsupervised data while achieving similar word error rates to wav2vec~2.0 on the Librispeech 100-hour dataset. Finally, we postulate that solving the contrastive task is a regularization for the supervised CTC loss.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Sound (cs.SD) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2011.00093](https://arxiv.org/abs/2011.00093) [cs.CL]** |
|           | (or **[arXiv:2011.00093v1](https://arxiv.org/abs/2011.00093v1) [cs.CL]** for this version) |





<h2 id="2020-11-03-5">5. Evaluating Bias In Dutch Word Embeddings</h2>

Title: [Evaluating Bias In Dutch Word Embeddings](https://arxiv.org/abs/2011.00244)

Authors: [Rodrigo Alejandro Chávez Mulsa](https://arxiv.org/search/cs?searchtype=author&query=Mulsa%2C+R+A+C), [Gerasimos Spanakis](https://arxiv.org/search/cs?searchtype=author&query=Spanakis%2C+G)

> Recent research in Natural Language Processing has revealed that word embeddings can encode social biases present in the training data which can affect minorities in real world applications. This paper explores the gender bias implicit in Dutch embeddings while investigating whether English language based approaches can also be used in Dutch. We implement the Word Embeddings Association Test (WEAT), Clustering and Sentence Embeddings Association Test (SEAT) methods to quantify the gender bias in Dutch word embeddings, then we proceed to reduce the bias with Hard-Debias and Sent-Debias mitigation methods and finally we evaluate the performance of the debiased embeddings in downstream tasks. The results suggest that, among others, gender bias is present in traditional and contextualized Dutch word embeddings. We highlight how techniques used to measure and reduce bias created for English can be used in Dutch embeddings by adequately translating the data and taking into account the unique characteristics of the language. Furthermore, we analyze the effect of the debiasing techniques on downstream tasks which show a negligible impact on traditional embeddings and a 2% decrease in performance in contextualized embeddings. Finally, we release the translated Dutch datasets to the public along with the traditional embeddings with mitigated bias.

| Comments: | Accepted at GeBNLP 2020, data at [this https URL](https://github.com/Noixas/Official-Evaluating-Bias-In-Dutch) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2011.00244](https://arxiv.org/abs/2011.00244) [cs.CL]** |
|           | (or **[arXiv:2011.00244v1](https://arxiv.org/abs/2011.00244v1) [cs.CL]** for this version) |





<h2 id="2020-11-03-6">6. Targeted Poisoning Attacks on Black-Box Neural Machine Translation
</h2>

Title: [Targeted Poisoning Attacks on Black-Box Neural Machine Translation](https://arxiv.org/abs/2011.00675)

Authors: [Chang Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+C), [Jun Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J), [Yuqing Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+Y), [Francisco Guzman](https://arxiv.org/search/cs?searchtype=author&query=Guzman%2C+F), [Benjamin I. P. Rubinstein](https://arxiv.org/search/cs?searchtype=author&query=Rubinstein%2C+B+I+P), [Trevor Cohn](https://arxiv.org/search/cs?searchtype=author&query=Cohn%2C+T)

> As modern neural machine translation (NMT) systems have been widely deployed, their security vulnerabilities require close scrutiny. Most recently, NMT systems have been shown to be vulnerable to targeted attacks which cause them to produce specific, unsolicited, and even harmful translations. These attacks are usually exploited in a white-box setting, where adversarial inputs causing targeted translations are discovered for a known target system. However, this approach is less useful when the target system is black-box and unknown to the adversary (e.g., secured commercial systems). In this paper, we show that targeted attacks on black-box NMT systems are feasible, based on poisoning a small fraction of their parallel training data. We demonstrate that this attack can be realised practically via targeted corruption of web documents crawled to form the system's training data. We then analyse the effectiveness of the targeted poisoning in two common NMT training scenarios, which are the one-off training and pre-train & fine-tune paradigms. Our results are alarming: even on the state-of-the-art systems trained with massive parallel data (tens of millions), the attacks are still successful (over 50% success rate) under surprisingly low poisoning rates (e.g., 0.006%). Lastly, we discuss potential defences to counter such attacks.

| Subjects: | **Computation and Language (cs.CL)**; Cryptography and Security (cs.CR) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2011.00675](https://arxiv.org/abs/2011.00675) [cs.CL]** |
|           | (or **[arXiv:2011.00675v1](https://arxiv.org/abs/2011.00675v1) [cs.CL]** for this version) |





<h2 id="2020-11-03-7">7. Investigating Catastrophic Forgetting During Continual Training for Neural Machine Translation</h2>

Title: [Investigating Catastrophic Forgetting During Continual Training for Neural Machine Translation](https://arxiv.org/abs/2011.00678)

Authors: [Shuhao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+S), [Yang Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+Y)

> Neural machine translation (NMT) models usually suffer from catastrophic forgetting during continual training where the models tend to gradually forget previously learned knowledge and swing to fit the newly added data which may have a different distribution, e.g. a different domain. Although many methods have been proposed to solve this problem, we cannot get to know what causes this phenomenon yet. Under the background of domain adaptation, we investigate the cause of catastrophic forgetting from the perspectives of modules and parameters (neurons). The investigation on the modules of the NMT model shows that some modules have tight relation with the general-domain knowledge while some other modules are more essential in the domain adaptation. And the investigation on the parameters shows that some parameters are important for both the general-domain and in-domain translation and the great change of them during continual training brings about the performance decline in general-domain. We conduct experiments across different language pairs and domains to ensure the validity and reliability of our findings.

| Comments: | Coling2020 long paper                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2011.00678](https://arxiv.org/abs/2011.00678) [cs.CL]** |
|           | (or **[arXiv:2011.00678v1](https://arxiv.org/abs/2011.00678v1) [cs.CL]** for this version) |





<h2 id="2020-11-03-8">8. Dual-decoder Transformer for Joint Automatic Speech Recognition and Multilingual Speech Translation</h2>

Title: [Dual-decoder Transformer for Joint Automatic Speech Recognition and Multilingual Speech Translation](https://arxiv.org/abs/2011.00747)

Authors: [Hang Le](https://arxiv.org/search/cs?searchtype=author&query=Le%2C+H), [Juan Pino](https://arxiv.org/search/cs?searchtype=author&query=Pino%2C+J), [Changhan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Jiatao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+J), [Didier Schwab](https://arxiv.org/search/cs?searchtype=author&query=Schwab%2C+D), [Laurent Besacier](https://arxiv.org/search/cs?searchtype=author&query=Besacier%2C+L)

> We introduce dual-decoder Transformer, a new model architecture that jointly performs automatic speech recognition (ASR) and multilingual speech translation (ST). Our models are based on the original Transformer architecture (Vaswani et al., 2017) but consist of two decoders, each responsible for one task (ASR or ST). Our major contribution lies in how these decoders interact with each other: one decoder can attend to different information sources from the other via a dual-attention mechanism. We propose two variants of these architectures corresponding to two different levels of dependencies between the decoders, called the parallel and cross dual-decoder Transformers, respectively. Extensive experiments on the MuST-C dataset show that our models outperform the previously-reported highest translation performance in the multilingual settings, and outperform as well bilingual one-to-one results. Furthermore, our parallel models demonstrate no trade-off between ASR and ST compared to the vanilla multi-task architecture. Our code and pre-trained models are available at [this https URL](https://github.com/formiel/speech-translation).

| Comments:          | Accepted at COLING 2020 (Oral)                               |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**                         |
| Journal reference: | The 28th International Conference on Computational Linguistics (COLING 2020) |
| Cite as:           | **[arXiv:2011.00747](https://arxiv.org/abs/2011.00747) [cs.CL]** |
|                    | (or **[arXiv:2011.00747v1](https://arxiv.org/abs/2011.00747v1) [cs.CL]** for this version) |





<h2 id="2020-11-03-9">9. Context-Aware Cross-Attention for Non-Autoregressive Translation</h2>

Title: [Context-Aware Cross-Attention for Non-Autoregressive Translation](https://arxiv.org/abs/2011.00770)

Authors: [Liang Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+L), [Longyue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Di Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+D), [Dacheng Tao](https://arxiv.org/search/cs?searchtype=author&query=Tao%2C+D), [Zhaopeng Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+Z)

> Non-autoregressive translation (NAT) significantly accelerates the inference process by predicting the entire target sequence. However, due to the lack of target dependency modelling in the decoder, the conditional generation process heavily depends on the cross-attention. In this paper, we reveal a localness perception problem in NAT cross-attention, for which it is difficult to adequately capture source context. To alleviate this problem, we propose to enhance signals of neighbour source tokens into conventional cross-attention. Experimental results on several representative datasets show that our approach can consistently improve translation quality over strong NAT baselines. Extensive analyses demonstrate that the enhanced cross-attention achieves better exploitation of source contexts by leveraging both local and global information.

| Comments: | To appear in COLING 2020                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2011.00770](https://arxiv.org/abs/2011.00770) [cs.CL]** |
|           | (or **[arXiv:2011.00770v1](https://arxiv.org/abs/2011.00770v1) [cs.CL]** for this version) |





<h2 id="2020-11-03-10">10. Emergent Communication Pretraining for Few-Shot Machine Translation</h2>

Title: [Emergent Communication Pretraining for Few-Shot Machine Translation](https://arxiv.org/abs/2011.00890)

Authors: [Yaoyiran Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Edoardo M. Ponti](https://arxiv.org/search/cs?searchtype=author&query=Ponti%2C+E+M), [Ivan Vulić](https://arxiv.org/search/cs?searchtype=author&query=Vulić%2C+I), [Anna Korhonen](https://arxiv.org/search/cs?searchtype=author&query=Korhonen%2C+A)

> While state-of-the-art models that rely upon massively multilingual pretrained encoders achieve sample efficiency in downstream applications, they still require abundant amounts of unlabelled text. Nevertheless, most of the world's languages lack such resources. Hence, we investigate a more radical form of unsupervised knowledge transfer in the absence of linguistic data. In particular, for the first time we pretrain neural networks via emergent communication from referential games. Our key assumption is that grounding communication on images---as a crude approximation of real-world environments---inductively biases the model towards learning natural languages. On the one hand, we show that this substantially benefits machine translation in few-shot settings. On the other hand, this also provides an extrinsic evaluation protocol to probe the properties of emergent languages ex vitro. Intuitively, the closer they are to natural languages, the higher the gains from pretraining on them should be. For instance, in this work we measure the influence of communication success and maximum sequence length on downstream performances. Finally, we introduce a customised adapter layer and annealing strategies for the regulariser of maximum-a-posteriori inference during fine-tuning. These turn out to be crucial to facilitate knowledge transfer and prevent catastrophic forgetting. Compared to a recurrent baseline, our method yields gains of 59.0%∼147.6% in BLEU score with only 500 NMT training instances and 65.1%∼196.7% with 1,000 NMT training instances across four language pairs. These proof-of-concept results reveal the potential of emergent communication pretraining for both natural language processing tasks in resource-poor settings and extrinsic evaluation of artificial languages.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2011.00890](https://arxiv.org/abs/2011.00890) [cs.CL]** |
|           | (or **[arXiv:2011.00890v1](https://arxiv.org/abs/2011.00890v1) [cs.CL]** for this version) |





<h2 id="2020-11-03-11">11. How Far Does BERT Look At:Distance-based Clustering and Analysis of BERTś Attention</h2>

Title: [How Far Does BERT Look At:Distance-based Clustering and Analysis of BERTś Attention](https://arxiv.org/abs/2011.00943)

Authors: [Yue Guan](https://arxiv.org/search/cs?searchtype=author&query=Guan%2C+Y), [Jingwen Leng](https://arxiv.org/search/cs?searchtype=author&query=Leng%2C+J), [Chao Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+C), [Quan Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Q), [Minyi Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+M)

> Recent research on the multi-head attention mechanism, especially that in pre-trained modelssuch as BERT, has shown us heuristics and clues in analyzing various aspects of the [this http URL](http://mechanism.as/) most of the research focus on probing tasks or hidden states, previous works have found someprimitive patterns of attention head behavior by heuristic analytical methods, but a more system-atic analysis specific on the attention patterns still remains primitive. In this work, we clearlycluster the attention heatmaps into significantly different patterns through unsupervised cluster-ing on top of a set of proposed features, which corroborates with previous observations. Wefurther study their corresponding functions through analytical study. In addition, our proposedfeatures can be used to explain and calibrate different attention heads in Transformer models.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2011.00943](https://arxiv.org/abs/2011.00943) [cs.CL]** |
|           | (or **[arXiv:2011.00943v1](https://arxiv.org/abs/2011.00943v1) [cs.CL]** for this version) |



<h2 id="2020-11-03-12">12. Enabling Zero-shot Multilingual Spoken Language Translation with Language-Specific Encoders and Decoders</h2>

Title: [Enabling Zero-shot Multilingual Spoken Language Translation with Language-Specific Encoders and Decoders](https://arxiv.org/abs/2011.01097)

Authors: [Carlos Escolano](https://arxiv.org/search/cs?searchtype=author&query=Escolano%2C+C), [Marta R. Costa-jussà](https://arxiv.org/search/cs?searchtype=author&query=Costa-jussà%2C+M+R), [José A. R. Fonollosa](https://arxiv.org/search/cs?searchtype=author&query=Fonollosa%2C+J+A+R), [Carlos Segura](https://arxiv.org/search/cs?searchtype=author&query=Segura%2C+C)

> Current end-to-end approaches to Spoken Language Translation (SLT) rely on limited training resources, especially for multilingual settings. On the other hand, Multilingual Neural Machine Translation (MultiNMT) approaches rely on higher quality and more massive data sets. Our proposed method extends a MultiNMT architecture based on language-specific encoders-decoders to the task of Multilingual SLT (MultiSLT) Our experiments on four different languages show that coupling the speech encoder to the MultiNMT architecture produces similar quality translations compared to a bilingual baseline (±0.2 BLEU) while effectively allowing for zero-shot MultiSLT. Additionally, we propose using Adapter networks for SLT that produce consistent improvements of +1 BLEU points in all tested languages.

| Comments:    | Submitted to ICASSP 2021                                     |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**                         |
| ACM classes: | I.2.7                                                        |
| Cite as:     | **[arXiv:2011.01097](https://arxiv.org/abs/2011.01097) [cs.CL]** |
|              | (or **[arXiv:2011.01097v1](https://arxiv.org/abs/2011.01097v1) [cs.CL]** for this version) |



<h2 id="2020-11-03-13">13. The Devil is in the Details: Evaluating Limitations of Transformer-based Methods for Granular Tasks</h2>

Title: [The Devil is in the Details: Evaluating Limitations of Transformer-based Methods for Granular Tasks](https://arxiv.org/abs/2011.01196)

Authors: [Brihi Joshi](https://arxiv.org/search/cs?searchtype=author&query=Joshi%2C+B), [Neil Shah](https://arxiv.org/search/cs?searchtype=author&query=Shah%2C+N), [Francesco Barbieri](https://arxiv.org/search/cs?searchtype=author&query=Barbieri%2C+F), [Leonardo Neves](https://arxiv.org/search/cs?searchtype=author&query=Neves%2C+L)

> Contextual embeddings derived from transformer-based neural language models have shown state-of-the-art performance for various tasks such as question answering, sentiment analysis, and textual similarity in recent years. Extensive work shows how accurately such models can represent abstract, semantic information present in text. In this expository work, we explore a tangent direction and analyze such models' performance on tasks that require a more granular level of representation. We focus on the problem of textual similarity from two perspectives: matching documents on a granular level (requiring embeddings to capture fine-grained attributes in the text), and an abstract level (requiring embeddings to capture overall textual semantics). We empirically demonstrate, across two datasets from different domains, that despite high performance in abstract document matching as expected, contextual embeddings are consistently (and at times, vastly) outperformed by simple baselines like TF-IDF for more granular tasks. We then propose a simple but effective method to incorporate TF-IDF into models that use contextual embeddings, achieving relative improvements of up to 36% on granular tasks.

| Comments: | Accepted at COLING 2020. Code available at [this https URL](https://github.com/brihijoshi/granular-similarity-COLING-2020) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2011.01196](https://arxiv.org/abs/2011.01196) [cs.CL]** |
|           | (or **[arXiv:2011.01196v1](https://arxiv.org/abs/2011.01196v1) [cs.CL]** for this version) |







# 2020-11-02

[Return to Index](#Index)



<h2 id="2020-11-02-1">1. VECO: Variable Encoder-decoder Pre-training for Cross-lingual Understanding and Generation</h2>

Title: [VECO: Variable Encoder-decoder Pre-training for Cross-lingual Understanding and Generation](https://arxiv.org/abs/2010.16046)

Authors: [Fuli Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+F), [Wei Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+W), [Jiahao Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Yijia Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Bin Bi](https://arxiv.org/search/cs?searchtype=author&query=Bi%2C+B), [Songfang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Fei Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+F), [Luo Si](https://arxiv.org/search/cs?searchtype=author&query=Si%2C+L)

> Recent studies about learning multilingual representations have achieved significant performance gains across a wide range of downstream cross-lingual tasks. They train either an encoder-only Transformer mainly for understanding tasks, or an encoder-decoder Transformer specifically for generation tasks, ignoring the correlation between the two tasks and frameworks. In contrast, this paper presents a variable encoder-decoder (VECO) pre-training approach to unify the two mainstreams in both model architectures and pre-training tasks. VECO splits the standard Transformer block into several sub-modules trained with both inner-sequence and cross-sequence masked language modeling, and correspondingly reorganizes certain sub-modules for understanding and generation tasks during inference. Such a workflow not only ensures to train the most streamlined parameters necessary for two kinds of tasks, but also enables them to boost each other via sharing common sub-modules. As a result, VECO delivers new state-of-the-art results on various cross-lingual understanding tasks of the XTREME benchmark covering text classification, sequence labeling, question answering, and sentence retrieval. For generation tasks, VECO also outperforms all existing cross-lingual models and state-of-the-art Transformer variants on WMT14 English-to-German and English-to-French translation datasets, with gains of up to 1∼2 BLEU.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2010.16046](https://arxiv.org/abs/2010.16046) [cs.CL] |
|           | (or [arXiv:2010.16046v1](https://arxiv.org/abs/2010.16046v1) [cs.CL] for this version) |





<h2 id="2020-11-02-2">2. Domain-Specific Lexical Grounding in Noisy Visual-Textual Documents</h2>

Title: [Domain-Specific Lexical Grounding in Noisy Visual-Textual Documents](https://arxiv.org/abs/2010.16363)

Authors: [Gregory Yauney](https://arxiv.org/search/cs?searchtype=author&query=Yauney%2C+G), [Jack Hessel](https://arxiv.org/search/cs?searchtype=author&query=Hessel%2C+J), [David Mimno](https://arxiv.org/search/cs?searchtype=author&query=Mimno%2C+D)

> Images can give us insights into the contextual meanings of words, but current image-text grounding approaches require detailed annotations. Such granular annotation is rare, expensive, and unavailable in most domain-specific contexts. In contrast, unlabeled multi-image, multi-sentence documents are abundant. Can lexical grounding be learned from such documents, even though they have significant lexical and visual overlap? Working with a case study dataset of real estate listings, we demonstrate the challenge of distinguishing highly correlated grounded terms, such as "kitchen" and "bedroom", and introduce metrics to assess this document similarity. We present a simple unsupervised clustering-based method that increases precision and recall beyond object detection and image tagging baselines when evaluated on labeled subsets of the dataset. The proposed method is particularly effective for local contextual meanings of a word, for example associating "granite" with countertops in the real estate dataset and with rocky landscapes in a Wikipedia dataset.

| Subjects:          | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV) |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | Published in EMNLP 2020                                      |
| Cite as:           | [arXiv:2010.16363](https://arxiv.org/abs/2010.16363) [cs.CL] |
|                    | (or [arXiv:2010.16363v1](https://arxiv.org/abs/2010.16363v1) [cs.CL] for this version) |