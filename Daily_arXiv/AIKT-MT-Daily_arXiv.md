# Daily arXiv: Machine Translation - January, 2021

# Index


- [2021-01-01](#2021-01-01)
	- [1. Understanding and Improving Lexical Choice in Non-Autoregressive Translation](#2021-01-01-1)
  - [2. Faster Re-translation Using Non-Autoregressive Model For Simultaneous Neural Machine Translation](#2021-01-01-2)
  - [3. LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](#2021-01-01-3)
  - [4. CMV-BERT: Contrastive multi-vocab pretraining of BERT](#2021-01-01-4)
  - [5. Understanding and Improving Encoder Layer Fusion in Sequence-to-Sequence Learning](#2021-01-01-5)
  - [6. Transformer Feed-Forward Layers Are Key-Value Memories](#2021-01-01-6)
  - [7. Reservoir Transformer](#2021-01-01-7)
  - [8. Enhancing Pre-trained Language Model with Lexical Simplification](#2021-01-01-8)
  - [9. Accurate Word Representations with Universal Visual Guidance](#2021-01-01-9)
  - [10. Improving Zero-Shot Translation by Disentangling Positional Information](#2021-01-01-10)
  - [11. Improving BERT with Syntax-aware Local Attention](#2021-01-01-11)
  - [12. Synthetic Source Language Augmentation for Colloquial Neural Machine Translation](#2021-01-01-12)
  - [13. Out of Order: How important is the sequential order of words in a sentence in Natural Language Understanding tasks?](#2021-01-01-13)
  - [14. SemGloVe: Semantic Co-occurrences for GloVe from BERT](#2021-01-01-14)
  - [15. UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning](#2021-01-01-15)
  - [16. Directed Beam Search: Plug-and-Play Lexically Constrained Language Generation](#2021-01-01-16)
  - [17. Exploring Monolingual Data for Neural Machine Translation with Knowledge Distillation](#2021-01-01-17)
  - [18. CLEAR: Contrastive Learning for Sentence Representation](#2021-01-01-18)
  - [19. Seeing is Knowing! Fact-based Visual Question Answering using Knowledge Graph Embeddings](#2021-01-01-19)
  - [20. Towards Zero-Shot Knowledge Distillation for Natural Language Processing](#2021-01-01-20)
  - [21. Neural Machine Translation: A Review of Methods, Resources, and Tools](#2021-01-01-21)
  - [22. Linear-Time WordPiece Tokenization](#2021-01-01-22)
  - [23. XLM-T: Scaling up Multilingual Machine Translation with Pretrained Cross-lingual Transformer Encoders](#2021-01-01-23)
  - [24. How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models](#2021-01-01-24)
  - [25. CoCoLM: COmplex COmmonsense Enhanced Language Model](#2021-01-01-25)
  - [26. VOLT: Improving Vocabularization via Optimal Transport for Machine Translation](#2021-01-01-26)
  - [27. ERNIE-M: Enhanced Multilingual Representation by Aligning Cross-lingual Semantics with Monolingual Corpora](#2021-01-01-27)
  - [28. Revisiting Robust Neural Machine Translation: A Transformer Case Study](#2021-01-01-28)
  - [29. FDMT: A Benchmark Dataset for Fine-grained Domain Adaptation in Machine Translation](#2021-01-01-29)
  - [30. Making Pre-trained Language Models Better Few-shot Learners](#2021-01-01-30)
  - [31. Shortformer: Better Language Modeling using Shorter Inputs](#2021-01-01-31)
  - [32. Fully Non-autoregressive Neural Machine Translation: Tricks of the Trade](#2021-01-01-32)
- [Other Columns](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-01-01

[Return to Index](#Index)





<h2 id="2021-01-01-1">1. Hello World</h2>



<h2 id="2021-01-01-1">1. Understanding and Improving Lexical Choice in Non-Autoregressive Translation</h2>

Title: [Understanding and Improving Lexical Choice in Non-Autoregressive Translation](https://arxiv.org/abs/2012.14583)

Authors: [Liang Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+L), [Longyue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Xuebo Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+X), [Derek F. Wong](https://arxiv.org/search/cs?searchtype=author&query=Wong%2C+D+F), [Dacheng Tao](https://arxiv.org/search/cs?searchtype=author&query=Tao%2C+D), [Zhaopeng Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+Z)

> Knowledge distillation (KD) is essential for training non-autoregressive translation (NAT) models by reducing the complexity of the raw data with an autoregressive teacher model. In this study, we empirically show that as a side effect of this training, the lexical choice errors on low-frequency words are propagated to the NAT model from the teacher model. To alleviate this problem, we propose to expose the raw data to NAT models to restore the useful information of low-frequency words, which are missed in the distilled data. To this end, we introduce an extra Kullback-Leibler divergence term derived by comparing the lexical choice of NAT model and that embedded in the raw data. Experimental results across language pairs and model architectures demonstrate the effectiveness and universality of the proposed approach. Extensive analyses confirm our claim that our approach improves performance by reducing the lexical choice errors on low-frequency words. Encouragingly, our approach pushes the SOTA NAT performance on the WMT14 English-German and WMT16 Romanian-English datasets up to 27.8 and 33.8 BLEU points, respectively. The source code will be released.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.14583](https://arxiv.org/abs/2012.14583) [cs.CL]** |
|           | (or **[arXiv:2012.14583v1](https://arxiv.org/abs/2012.14583v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-2">2. Faster Re-translation Using Non-Autoregressive Model For Simultaneous Neural Machine Translation</h2>

Title: [Faster Re-translation Using Non-Autoregressive Model For Simultaneous Neural Machine Translation](https://arxiv.org/abs/2012.14681)

Authors: [Hyojung Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+H), [Sathish Indurthi](https://arxiv.org/search/cs?searchtype=author&query=Indurthi%2C+S), [Mohd Abbas Zaidi](https://arxiv.org/search/cs?searchtype=author&query=Zaidi%2C+M+A), [Nikhil Kumar Lakumarapu](https://arxiv.org/search/cs?searchtype=author&query=Lakumarapu%2C+N+K), [Beomseok Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+B), [Sangha Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+S), [Chanwoo Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+C), [Inchul Hwang](https://arxiv.org/search/cs?searchtype=author&query=Hwang%2C+I)

> Recently, simultaneous translation has gathered a lot of attention since it enables compelling applications such as subtitle translation for a live event or real-time video-call translation. Some of these translation applications allow editing of partial translation giving rise to re-translation approaches. The current re-translation approaches are based on autoregressive sequence generation models (ReTA), which generate tar-get tokens in the (partial) translation sequentially. The multiple re-translations with sequential generation inReTAmodelslead to an increased inference time gap between the incoming source input and the corresponding target output as the source input grows. Besides, due to the large number of inference operations involved, the ReTA models are not favorable for resource-constrained devices. In this work, we propose a faster re-translation system based on a non-autoregressive sequence generation model (FReTNA) to overcome the aforementioned limitations. We evaluate the proposed model on multiple translation tasks and our model reduces the inference times by several orders and achieves a competitive BLEUscore compared to the ReTA and streaming (Wait-k) models.The proposed model reduces the average computation time by a factor of 20 when compared to the ReTA model by incurring a small drop in the translation quality. It also outperforms the streaming-based Wait-k model both in terms of computation time (1.5 times lower) and translation quality.

| Comments: | work in progress                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2012.14681](https://arxiv.org/abs/2012.14681) [cs.CL]** |
|           | (or **[arXiv:2012.14681v1](https://arxiv.org/abs/2012.14681v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-3">3. LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding</h2>

Title: [LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/abs/2012.14740)

Authors: [Yang Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+Y), [Yiheng Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+Y), [Tengchao Lv](https://arxiv.org/search/cs?searchtype=author&query=Lv%2C+T), [Lei Cui](https://arxiv.org/search/cs?searchtype=author&query=Cui%2C+L), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F), [Guoxin Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+G), [Yijuan Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+Y), [Dinei Florencio](https://arxiv.org/search/cs?searchtype=author&query=Florencio%2C+D), [Cha Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+C), [Wanxiang Che](https://arxiv.org/search/cs?searchtype=author&query=Che%2C+W), [Min Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M), [Lidong Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+L)

> Pre-training of text and layout has proved effective in a variety of visually-rich document understanding tasks due to its effective model architecture and the advantage of large-scale unlabeled scanned/digital-born documents. In this paper, we present \textbf{LayoutLMv2} by pre-training text, layout and image in a multi-modal framework, where new model architectures and pre-training tasks are leveraged. Specifically, LayoutLMv2 not only uses the existing masked visual-language modeling task but also the new text-image alignment and text-image matching tasks in the pre-training stage, where cross-modality interaction is better learned. Meanwhile, it also integrates a spatial-aware self-attention mechanism into the Transformer architecture, so that the model can fully understand the relative positional relationship among different text blocks. Experiment results show that LayoutLMv2 outperforms strong baselines and achieves new state-of-the-art results on a wide variety of downstream visually-rich document understanding tasks, including FUNSD (0.7895 -> 0.8420), CORD (0.9493 -> 0.9601), SROIE (0.9524 -> 0.9781), Kleister-NDA (0.834 -> 0.852), RVL-CDIP (0.9443 -> 0.9564), and DocVQA (0.7295 -> 0.8672).

| Comments: | Work in progress                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2012.14740](https://arxiv.org/abs/2012.14740) [cs.CL]** |
|           | (or **[arXiv:2012.14740v1](https://arxiv.org/abs/2012.14740v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-4">4. CMV-BERT: Contrastive multi-vocab pretraining of BERT</h2>

Title: [CMV-BERT: Contrastive multi-vocab pretraining of BERT](https://arxiv.org/abs/2012.14763)

Authors: [Wei Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+W), [Daniel Cheung](https://arxiv.org/search/cs?searchtype=author&query=Cheung%2C+D)

> In this work, we represent CMV-BERT, which improves the pretraining of a language model via two ingredients: (a) contrastive learning, which is well studied in the area of computer vision; (b) multiple vocabularies, one of which is fine-grained and the other is coarse-grained. The two methods both provide different views of an original sentence, and both are shown to be beneficial. Downstream tasks demonstrate our proposed CMV-BERT are effective in improving the pretrained language models.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.14763](https://arxiv.org/abs/2012.14763) [cs.CL]** |
|           | (or **[arXiv:2012.14763v1](https://arxiv.org/abs/2012.14763v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-5">5. Understanding and Improving Encoder Layer Fusion in Sequence-to-Sequence Learning</h2>

Title: [Understanding and Improving Encoder Layer Fusion in Sequence-to-Sequence Learning](https://arxiv.org/abs/2012.14768)

Authors: [Xuebo Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+X), [Longyue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Derek F. Wong](https://arxiv.org/search/cs?searchtype=author&query=Wong%2C+D+F), [Liang Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+L), [Lidia S. Chao](https://arxiv.org/search/cs?searchtype=author&query=Chao%2C+L+S), [Zhaopeng Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+Z)

> Encoder layer fusion (EncoderFusion) is a technique to fuse all the encoder layers (instead of the uppermost layer) for sequence-to-sequence (Seq2Seq) models, which has proven effective on various NLP tasks. However, it is still not entirely clear why and when EncoderFusion should work. In this paper, our main contribution is to take a step further in understanding EncoderFusion. Many of previous studies believe that the success of EncoderFusion comes from exploiting surface and syntactic information embedded in lower encoder layers. Unlike them, we find that the encoder embedding layer is more important than other intermediate encoder layers. In addition, the uppermost decoder layer consistently pays more attention to the encoder embedding layer across NLP tasks. Based on this observation, we propose a simple fusion method, SurfaceFusion, by fusing only the encoder embedding layer for the softmax layer. Experimental results show that SurfaceFusion outperforms EncoderFusion on several NLP benchmarks, including machine translation, text summarization, and grammatical error correction. It obtains the state-of-the-art performance on WMT16 Romanian-English and WMT14 English-French translation tasks. Extensive analyses reveal that SurfaceFusion learns more expressive bilingual word embeddings by building a closer relationship between relevant source and target embeddings. The source code will be released.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.14768](https://arxiv.org/abs/2012.14768) [cs.CL]** |
|           | (or **[arXiv:2012.14768v1](https://arxiv.org/abs/2012.14768v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-6">6. Transformer Feed-Forward Layers Are Key-Value Memories</h2>

Title: [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)

Authors: [Mor Geva](https://arxiv.org/search/cs?searchtype=author&query=Geva%2C+M), [Roei Schuster](https://arxiv.org/search/cs?searchtype=author&query=Schuster%2C+R), [Jonathan Berant](https://arxiv.org/search/cs?searchtype=author&query=Berant%2C+J), [Omer Levy](https://arxiv.org/search/cs?searchtype=author&query=Levy%2C+O)

> Feed-forward layers constitute two-thirds of a transformer model's parameters, yet their role in the network remains under-explored. We show that feed-forward layers in transformer-based language models operate as key-value memories, where each key correlates with textual patterns in the training examples, and each value induces a distribution over the output vocabulary. Our experiments show that the learned patterns are human-interpretable, and that lower layers tend to capture shallow patterns, while upper layers learn more semantic ones. The values complement the keys' input patterns by inducing output distributions that concentrate probability mass on tokens likely to appear immediately after each pattern, particularly in the upper layers. Finally, we demonstrate that the output of a feed-forward layer is a composition of its memories, which is subsequently refined throughout the model's layers via residual connections to produce the final output distribution.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.14913](https://arxiv.org/abs/2012.14913) [cs.CL]** |
|           | (or **[arXiv:2012.14913v1](https://arxiv.org/abs/2012.14913v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-7">7. Reservoir Transformer</h2>

Title: [Reservoir Transformer](https://arxiv.org/abs/2012.15045)

Authors: [Sheng Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+S), [Alexei Baevski](https://arxiv.org/search/cs?searchtype=author&query=Baevski%2C+A), [Ari S. Morcos](https://arxiv.org/search/cs?searchtype=author&query=Morcos%2C+A+S), [Kurt Keutzer](https://arxiv.org/search/cs?searchtype=author&query=Keutzer%2C+K), [Michael Auli](https://arxiv.org/search/cs?searchtype=author&query=Auli%2C+M), [Douwe Kiela](https://arxiv.org/search/cs?searchtype=author&query=Kiela%2C+D)

> We demonstrate that transformers obtain impressive performance even when some of the layers are randomly initialized and never updated. Inspired by old and well-established ideas in machine learning, we explore a variety of non-linear "reservoir" layers interspersed with regular transformer layers, and show improvements in wall-clock compute time until convergence, as well as overall performance, on various machine translation and (masked) language modelling tasks.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.15045](https://arxiv.org/abs/2012.15045) [cs.CL]** |
|           | (or **[arXiv:2012.15045v1](https://arxiv.org/abs/2012.15045v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-8">8. Enhancing Pre-trained Language Model with Lexical Simplification</h2>

Title: [Enhancing Pre-trained Language Model with Lexical Simplification](https://arxiv.org/abs/2012.15070)

Authors: [Rongzhou Bao](https://arxiv.org/search/cs?searchtype=author&query=Bao%2C+R), [Jiayi Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J), [Zhuosheng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Hai Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+H)

> For both human readers and pre-trained language models (PrLMs), lexical diversity may lead to confusion and inaccuracy when understanding the underlying semantic meanings of given sentences. By substituting complex words with simple alternatives, lexical simplification (LS) is a recognized method to reduce such lexical diversity, and therefore to improve the understandability of sentences. In this paper, we leverage LS and propose a novel approach which can effectively improve the performance of PrLMs in text classification. A rule-based simplification process is applied to a given sentence. PrLMs are encouraged to predict the real label of the given sentence with auxiliary inputs from the simplified version. Using strong PrLMs (BERT and ELECTRA) as baselines, our approach can still further improve the performance in various text classification tasks.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.15070](https://arxiv.org/abs/2012.15070) [cs.CL]** |
|           | (or **[arXiv:2012.15070v1](https://arxiv.org/abs/2012.15070v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-9">9. Accurate Word Representations with Universal Visual Guidance</h2>

Title: [Accurate Word Representations with Universal Visual Guidance](https://arxiv.org/abs/2012.15086)

Authors: [Zhuosheng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Haojie Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+H), [Hai Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+H), [Rui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+R), [Masao Utiyama](https://arxiv.org/search/cs?searchtype=author&query=Utiyama%2C+M)

> Word representation is a fundamental component in neural language understanding models. Recently, pre-trained language models (PrLMs) offer a new performant method of contextualized word representations by leveraging the sequence-level context for modeling. Although the PrLMs generally give more accurate contextualized word representations than non-contextualized models do, they are still subject to a sequence of text contexts without diverse hints for word representation from multimodality. This paper thus proposes a visual representation method to explicitly enhance conventional word embedding with multiple-aspect senses from visual guidance. In detail, we build a small-scale word-image dictionary from a multimodal seed dataset where each word corresponds to diverse related images. The texts and paired images are encoded in parallel, followed by an attention layer to integrate the multimodal representations. We show that the method substantially improves the accuracy of disambiguation. Experiments on 12 natural language understanding and machine translation tasks further verify the effectiveness and the generalization capability of the proposed approach.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.15086](https://arxiv.org/abs/2012.15086) [cs.CL]** |
|           | (or **[arXiv:2012.15086v1](https://arxiv.org/abs/2012.15086v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-1">10. Improving Zero-Shot Translation by Disentangling Positional Information</h2>

Title: [Improving Zero-Shot Translation by Disentangling Positional Information](https://arxiv.org/abs/2012.15127)

Authors: [Danni Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+D), [Jan Niehues](https://arxiv.org/search/cs?searchtype=author&query=Niehues%2C+J), [James Cross](https://arxiv.org/search/cs?searchtype=author&query=Cross%2C+J), [Francisco Guzmán](https://arxiv.org/search/cs?searchtype=author&query=Guzmán%2C+F), [Xian Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+X)

> Multilingual neural machine translation has shown the capability of directly translating between language pairs unseen in training, i.e. zero-shot translation. Despite being conceptually attractive, it often suffers from low output quality. The difficulty of generalizing to new translation directions suggests the model representations are highly specific to those language pairs seen in training. We demonstrate that a main factor causing the language-specific representations is the positional correspondence to input tokens. We show that this can be easily alleviated by removing residual connections in an encoder layer. With this modification, we gain up to 18.5 BLEU points on zero-shot translation while retaining quality on supervised directions. The improvements are particularly prominent between related languages, where our proposed model outperforms pivot-based translation. Moreover, our approach allows easy integration of new languages, which substantially expands translation coverage. By thorough inspections of the hidden layer outputs, we show that our approach indeed leads to more language-independent representations.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.15127](https://arxiv.org/abs/2012.15127) [cs.CL]** |
|           | (or **[arXiv:2012.15127v1](https://arxiv.org/abs/2012.15127v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-11">11. Improving BERT with Syntax-aware Local Attention</h2>

Title: [Improving BERT with Syntax-aware Local Attention](https://arxiv.org/abs/2012.15150)

Authors: [Zhongli Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Qingyu Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+Q), [Chao Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+C), [Ke Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+K), [Yunbo Cao](https://arxiv.org/search/cs?searchtype=author&query=Cao%2C+Y)

> Pre-trained Transformer-based neural language models, such as BERT, have achieved remarkable results on varieties of NLP tasks. Recent works have shown that attention-based models can benefit from more focused attention over local regions. Most of them restrict the attention scope within a linear span, or confine to certain tasks such as machine translation and question answering. In this paper, we propose a syntax-aware local attention, where the attention scopes are restrained based on the distances in the syntactic structure. The proposed syntax-aware local attention can be integrated with pretrained language models, such as BERT, to render the model to focus on syntactically relevant words. We conduct experiments on various single-sentence benchmarks, including sentence classification and sequence labeling tasks. Experimental results show consistent gains over BERT on all benchmark datasets. The extensive studies verify that our model achieves better performance owing to more focused attention over syntactically relevant words.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.15150](https://arxiv.org/abs/2012.15150) [cs.CL]** |
|           | (or **[arXiv:2012.15150v1](https://arxiv.org/abs/2012.15150v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-12">12. Synthetic Source Language Augmentation for Colloquial Neural Machine Translation</h2>

Title: [Synthetic Source Language Augmentation for Colloquial Neural Machine Translation](https://arxiv.org/abs/2012.15178)

Authors: [Asrul Sani Ariesandy](https://arxiv.org/search/cs?searchtype=author&query=Ariesandy%2C+A+S), [Mukhlis Amien](https://arxiv.org/search/cs?searchtype=author&query=Amien%2C+M), [Alham Fikri Aji](https://arxiv.org/search/cs?searchtype=author&query=Aji%2C+A+F), [Radityo Eko Prasojo](https://arxiv.org/search/cs?searchtype=author&query=Prasojo%2C+R+E)

> Neural machine translation (NMT) is typically domain-dependent and style-dependent, and it requires lots of training data. State-of-the-art NMT models often fall short in handling colloquial variations of its source language and the lack of parallel data in this regard is a challenging hurdle in systematically improving the existing models. In this work, we develop a novel colloquial Indonesian-English test-set collected from YouTube transcript and Twitter. We perform synthetic style augmentation to the source of formal Indonesian language and show that it improves the baseline Id-En models (in BLEU) over the new test data.

| Comments:    | 5 pages                                                      |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| MSC classes: | 68T50                                                        |
| ACM classes: | I.2.7; I.2.6                                                 |
| Cite as:     | **[arXiv:2012.15178](https://arxiv.org/abs/2012.15178) [cs.CL]** |
|              | (or **[arXiv:2012.15178v1](https://arxiv.org/abs/2012.15178v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-13">13. Out of Order: How important is the sequential order of words in a sentence in Natural Language Understanding tasks?</h2>

Title: [Out of Order: How important is the sequential order of words in a sentence in Natural Language Understanding tasks?](https://arxiv.org/abs/2012.15180)

Authors: [Thang M. Pham](https://arxiv.org/search/cs?searchtype=author&query=Pham%2C+T+M), [Trung Bui](https://arxiv.org/search/cs?searchtype=author&query=Bui%2C+T), [Long Mai](https://arxiv.org/search/cs?searchtype=author&query=Mai%2C+L), [Anh Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+A)

> Do state-of-the-art natural language understanding models care about word order - one of the most important characteristics of a sequence? Not always! We found 75% to 90% of the correct predictions of BERT-based classifiers, trained on many GLUE tasks, remain constant after input words are randomly shuffled. Despite BERT embeddings are famously contextual, the contribution of each individual word to downstream tasks is almost unchanged even after the word's context is shuffled. BERT-based models are able to exploit superficial cues (e.g. the sentiment of keywords in sentiment analysis; or the word-wise similarity between sequence-pair inputs in natural language inference) to make correct decisions when tokens are arranged in random orders. Encouraging classifiers to capture word order information improves the performance on most GLUE tasks, SQuAD 2.0 and out-of-samples. Our work suggests that many GLUE tasks are not challenging machines to understand the meaning of a sentence.

| Comments: | 23 pages, 13 figures. Preprint. Work in progress             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2012.15180](https://arxiv.org/abs/2012.15180) [cs.CL]** |
|           | (or **[arXiv:2012.15180v1](https://arxiv.org/abs/2012.15180v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-14">14. SemGloVe: Semantic Co-occurrences for GloVe from BERT</h2>

Title: [SemGloVe: Semantic Co-occurrences for GloVe from BERT](https://arxiv.org/abs/2012.15197)

Authors: [Leilei Gan](https://arxiv.org/search/cs?searchtype=author&query=Gan%2C+L), [Zhiyang Teng](https://arxiv.org/search/cs?searchtype=author&query=Teng%2C+Z), [Yue Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Linchao Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+L), [Fei Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+F), [Yi Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Y)

> GloVe learns word embeddings by leveraging statistical information from word co-occurrence matrices. However, word pairs in the matrices are extracted from a predefined local context window, which might lead to limited word pairs and potentially semantic irrelevant word pairs. In this paper, we propose SemGloVe, which distills semantic co-occurrences from BERT into static GloVe word embeddings. Particularly, we propose two models to extract co-occurrence statistics based on either the masked language model or the multi-head attention weights of BERT. Our methods can extract word pairs without limiting by the local window assumption and can define the co-occurrence weights by directly considering the semantic distance between word pairs. Experiments on several word similarity datasets and four external tasks show that SemGloVe can outperform GloVe.

| Comments: | 10 pages, 3 figures, 5 tables                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2012.15197](https://arxiv.org/abs/2012.15197) [cs.CL]** |
|           | (or **[arXiv:2012.15197v1](https://arxiv.org/abs/2012.15197v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-15">15. UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning</h2>

Title: [UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning](https://arxiv.org/abs/2012.15409)

Authors: [Wei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+W), [Can Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+C), [Guocheng Niu](https://arxiv.org/search/cs?searchtype=author&query=Niu%2C+G), [Xinyan Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+X), [Hao Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+H), [Jiachen Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Hua Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+H), [Haifeng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+H)

> Existed pre-training methods either focus on single-modal tasks or multi-modal tasks, and cannot effectively adapt to each other. They can only utilize single-modal data (i.e. text or image) or limited multi-modal data (i.e. image-text pairs). In this work, we propose a unified-modal pre-training architecture, namely UNIMO, which can effectively adapt to both single-modal and multi-modal understanding and generation tasks. Large scale of free text corpus and image collections can be utilized to improve the capability of visual and textual understanding, and cross-modal contrastive learning (CMCL) is leveraged to align the textual and visual information into a unified semantic space over a corpus of image-text pairs. As the non-paired single-modal data is very rich, our model can utilize much larger scale of data to learn more generalizable representations. Moreover, the textual knowledge and visual knowledge can enhance each other in the unified semantic space. The experimental results show that UNIMO significantly improves the performance of several single-modal and multi-modal downstream tasks.

| Comments: | 11 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2012.15409](https://arxiv.org/abs/2012.15409) [cs.CL]** |
|           | (or **[arXiv:2012.15409v1](https://arxiv.org/abs/2012.15409v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-16">16. Directed Beam Search: Plug-and-Play Lexically Constrained Language Generation</h2>

Title: [Directed Beam Search: Plug-and-Play Lexically Constrained Language Generation](https://arxiv.org/abs/2012.15416)

Authors: [Damian Pascual](https://arxiv.org/search/cs?searchtype=author&query=Pascual%2C+D), [Beni Egressy](https://arxiv.org/search/cs?searchtype=author&query=Egressy%2C+B), [Florian Bolli](https://arxiv.org/search/cs?searchtype=author&query=Bolli%2C+F), [Roger Wattenhofer](https://arxiv.org/search/cs?searchtype=author&query=Wattenhofer%2C+R)

> Large pre-trained language models are capable of generating realistic text. However, controlling these models so that the generated text satisfies lexical constraints, i.e., contains specific words, is a challenging problem. Given that state-of-the-art language models are too large to be trained from scratch in a manageable time, it is desirable to control these models without re-training them. Methods capable of doing this are called plug-and-play. Recent plug-and-play methods have been successful in constraining small bidirectional language models as well as forward models in tasks with a restricted search space, e.g., machine translation. However, controlling large transformer-based models to meet lexical constraints without re-training them remains a challenge. In this work, we propose Directed Beam Search (DBS), a plug-and-play method for lexically constrained language generation. Our method can be applied to any language model, is easy to implement and can be used for general language generation. In our experiments we use DBS to control GPT-2. We demonstrate its performance on keyword-to-phrase generation and we obtain comparable results as a state-of-the-art non-plug-and-play model for lexically constrained story generation.

| Comments: | Preprint. Work in progress                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2012.15416](https://arxiv.org/abs/2012.15416) [cs.CL]** |
|           | (or **[arXiv:2012.15416v1](https://arxiv.org/abs/2012.15416v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-17">17. Exploring Monolingual Data for Neural Machine Translation with Knowledge Distillation</h2>

Title: [Exploring Monolingual Data for Neural Machine Translation with Knowledge Distillation](https://arxiv.org/abs/2012.15455)

Authors: [Alham Fikri Aji](https://arxiv.org/search/cs?searchtype=author&query=Aji%2C+A+F), [Kenneth Heafield](https://arxiv.org/search/cs?searchtype=author&query=Heafield%2C+K)

> We explore two types of monolingual data that can be included in knowledge distillation training for neural machine translation (NMT). The first is the source-side monolingual data. Second, is the target-side monolingual data that is used as back-translation data. Both datasets are (forward-)translated by a teacher model from source-language to target-language, which are then combined into a dataset for smaller student models. We find that source-side monolingual data improves model performance when evaluated by test-set originated from source-side. Likewise, target-side data has a positive effect on the test-set in the opposite direction. We also show that it is not required to train the student model with the same data used by the teacher, as long as the domains are the same. Finally, we find that combining source-side and target-side yields in better performance than relying on just one side of the monolingual data.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.15455](https://arxiv.org/abs/2012.15455) [cs.CL]** |
|           | (or **[arXiv:2012.15455v1](https://arxiv.org/abs/2012.15455v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-18">18. CLEAR: Contrastive Learning for Sentence Representation</h2>

Title: [CLEAR: Contrastive Learning for Sentence Representation](https://arxiv.org/abs/2012.15466)

Authors: [Zhuofeng Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+Z), [Sinong Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Jiatao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+J), [Madian Khabsa](https://arxiv.org/search/cs?searchtype=author&query=Khabsa%2C+M), [Fei Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+F), [Hao Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+H)

> Pre-trained language models have proven their unique powers in capturing implicit language features. However, most pre-training approaches focus on the word-level training objective, while sentence-level objectives are rarely studied. In this paper, we propose Contrastive LEArning for sentence Representation (CLEAR), which employs multiple sentence-level augmentation strategies in order to learn a noise-invariant sentence representation. These augmentations include word and span deletion, reordering, and substitution. Furthermore, we investigate the key reasons that make contrastive learning effective through numerous experiments. We observe that different sentence augmentations during pre-training lead to different performance improvements on various downstream tasks. Our approach is shown to outperform multiple existing methods on both SentEval and GLUE benchmarks.

| Comments: | 10 pages, 2 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2012.15466](https://arxiv.org/abs/2012.15466) [cs.CL]** |
|           | (or **[arXiv:2012.15466v1](https://arxiv.org/abs/2012.15466v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-19">19. Seeing is Knowing! Fact-based Visual Question Answering using Knowledge Graph Embeddings</h2>

Title: [Seeing is Knowing! Fact-based Visual Question Answering using Knowledge Graph Embeddings](https://arxiv.org/abs/2012.15484)

Authors: [Kiran Ramnath](https://arxiv.org/search/cs?searchtype=author&query=Ramnath%2C+K), [Mark Hasegawa-Johnson](https://arxiv.org/search/cs?searchtype=author&query=Hasegawa-Johnson%2C+M)

> Fact-based Visual Question Answering (FVQA), a challenging variant of VQA, requires a QA-system to include facts from a diverse knowledge graph (KG) in its reasoning process to produce an answer. Large KGs, especially common-sense KGs, are known to be incomplete, i.e. not all non-existent facts are always incorrect. Therefore, being able to reason over incomplete KGs for QA is a critical requirement in real-world applications that has not been addressed extensively in the literature. We develop a novel QA architecture that allows us to reason over incomplete KGs, something current FVQA state-of-the-art (SOTA) approaches lack.We use KG Embeddings, a technique widely used for KG completion, for the downstream task of FVQA. We also employ a new image representation technique we call "Image-as-Knowledge" to enable this capability, alongside a simple one-step co-Attention mechanism to attend to text and image during QA. Our FVQA architecture is faster during inference time, being O(m), as opposed to existing FVQA SOTA methods which are O(N logN), where m is number of vertices, N is number of edges (which is O(m^2)). We observe that our architecture performs comparably in the standard answer-retrieval baseline with existing methods; while for missing-edge reasoning, our KG representation outperforms the SOTA representation by 25%, and image representation outperforms the SOTA representation by 2.6%.

| Comments: | 9 pages, 10 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2012.15484](https://arxiv.org/abs/2012.15484) [cs.CL]** |
|           | (or **[arXiv:2012.15484v1](https://arxiv.org/abs/2012.15484v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-20">20. Towards Zero-Shot Knowledge Distillation for Natural Language Processing</h2>

Title: [Towards Zero-Shot Knowledge Distillation for Natural Language Processing](https://arxiv.org/abs/2012.15495)

Authors: [Ahmad Rashid](https://arxiv.org/search/cs?searchtype=author&query=Rashid%2C+A), [Vasileios Lioutas](https://arxiv.org/search/cs?searchtype=author&query=Lioutas%2C+V), [Abbas Ghaddar](https://arxiv.org/search/cs?searchtype=author&query=Ghaddar%2C+A), [Mehdi Rezagholizadeh](https://arxiv.org/search/cs?searchtype=author&query=Rezagholizadeh%2C+M)

> Knowledge Distillation (KD) is a common knowledge transfer algorithm used for model compression across a variety of deep learning based natural language processing (NLP) solutions. In its regular manifestations, KD requires access to the teacher's training data for knowledge transfer to the student network. However, privacy concerns, data regulations and proprietary reasons may prevent access to such data. We present, to the best of our knowledge, the first work on Zero-Shot Knowledge Distillation for NLP, where the student learns from the much larger teacher without any task specific data. Our solution combines out of domain data and adversarial training to learn the teacher's output distribution. We investigate six tasks from the GLUE benchmark and demonstrate that we can achieve between 75% and 92% of the teacher's classification score (accuracy or F1) while compressing the model 30 times.

| Comments: | 13 pages, 8 tables, 2 algorithms and 1 figure                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2012.15495](https://arxiv.org/abs/2012.15495) [cs.CL]** |
|           | (or **[arXiv:2012.15495v1](https://arxiv.org/abs/2012.15495v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-21">21. Neural Machine Translation: A Review of Methods, Resources, and Tools</h2>

Title: [Neural Machine Translation: A Review of Methods, Resources, and Tools](https://arxiv.org/abs/2012.15515)

Authors: [Zhixing Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+Z), [Shuo Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Zonghan Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Gang Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+G), [Xuancheng Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+X), [Maosong Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+M), [Yang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y)

> Machine translation (MT) is an important sub-field of natural language processing that aims to translate natural languages using computers. In recent years, end-to-end neural machine translation (NMT) has achieved great success and has become the new mainstream method in practical MT systems. In this article, we first provide a broad review of the methods for NMT and focus on methods relating to architectures, decoding, and data augmentation. Then we summarize the resources and tools that are useful for researchers. Finally, we conclude with a discussion of possible future research directions.

| Comments: | Accepted by AI Open                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2012.15515](https://arxiv.org/abs/2012.15515) [cs.CL]** |
|           | (or **[arXiv:2012.15515v1](https://arxiv.org/abs/2012.15515v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-22">22. Linear-Time WordPiece Tokenization</h2>

Title: [Linear-Time WordPiece Tokenization](https://arxiv.org/abs/2012.15524)

Authors: [Xinying Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+X), [Alex Salcianu](https://arxiv.org/search/cs?searchtype=author&query=Salcianu%2C+A), [Yang Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+Y), [Dave Dopson](https://arxiv.org/search/cs?searchtype=author&query=Dopson%2C+D), [Denny Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+D)

> WordPiece tokenization is a subword-based tokenization schema adopted by BERT: it segments the input text via a longest-match-first tokenization strategy, known as Maximum Matching or MaxMatch. To the best of our knowledge, all published MaxMatch algorithms are quadratic (or higher). In this paper, we propose LinMaxMatch, a novel linear-time algorithm for MaxMatch and WordPiece tokenization. Inspired by the Aho-Corasick algorithm, we introduce additional linkages on top of the trie built from the vocabulary, allowing smart transitions when the trie matching cannot continue. Experimental results show that our algorithm is 3x faster on average than two production systems by HuggingFace and TensorFlow Text. Regarding long-tail inputs, our algorithm is 4.5x faster at the 95 percentile. This work has immediate practical value (reducing inference latency, saving compute resources, etc.) and is of theoretical interest by providing an optimal complexity solution to the decades-old MaxMatch problem.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.15524](https://arxiv.org/abs/2012.15524) [cs.CL]** |
|           | (or **[arXiv:2012.15524v1](https://arxiv.org/abs/2012.15524v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-23">23. XLM-T: Scaling up Multilingual Machine Translation with Pretrained Cross-lingual Transformer Encoders</h2>

Title: [XLM-T: Scaling up Multilingual Machine Translation with Pretrained Cross-lingual Transformer Encoders](https://arxiv.org/abs/2012.15547)

Authors: [Shuming Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+S), [Jian Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+J), [Haoyang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+H), [Zewen Chi](https://arxiv.org/search/cs?searchtype=author&query=Chi%2C+Z), [Li Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+L), [Dongdong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+D), [Hany Hassan Awadalla](https://arxiv.org/search/cs?searchtype=author&query=Awadalla%2C+H+H), [Alexandre Muzio](https://arxiv.org/search/cs?searchtype=author&query=Muzio%2C+A), [Akiko Eriguchi](https://arxiv.org/search/cs?searchtype=author&query=Eriguchi%2C+A), [Saksham Singhal](https://arxiv.org/search/cs?searchtype=author&query=Singhal%2C+S), [Xia Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+X), [Arul Menezes](https://arxiv.org/search/cs?searchtype=author&query=Menezes%2C+A), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F)

> Multilingual machine translation enables a single model to translate between different languages. Most existing multilingual machine translation systems adopt a randomly initialized Transformer backbone. In this work, inspired by the recent success of language model pre-training, we present XLM-T, which initializes the model with an off-the-shelf pretrained cross-lingual Transformer encoder and fine-tunes it with multilingual parallel data. This simple method achieves significant improvements on a WMT dataset with 10 language pairs and the OPUS-100 corpus with 94 pairs. Surprisingly, the method is also effective even upon the strong baseline with back-translation. Moreover, extensive analysis of XLM-T on unsupervised syntactic parsing, word alignment, and multilingual classification explains its effectiveness for machine translation. The code will be at [this https URL](https://aka.ms/xlm-t).

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.15547](https://arxiv.org/abs/2012.15547) [cs.CL]** |
|           | (or **[arXiv:2012.15547v1](https://arxiv.org/abs/2012.15547v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-24">24. How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models</h2>

Title: [How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models](https://arxiv.org/abs/2012.15613)

Authors: [Phillip Rust](https://arxiv.org/search/cs?searchtype=author&query=Rust%2C+P), [Jonas Pfeiffer](https://arxiv.org/search/cs?searchtype=author&query=Pfeiffer%2C+J), [Ivan Vulić](https://arxiv.org/search/cs?searchtype=author&query=Vulić%2C+I), [Sebastian Ruder](https://arxiv.org/search/cs?searchtype=author&query=Ruder%2C+S), [Iryna Gurevych](https://arxiv.org/search/cs?searchtype=author&query=Gurevych%2C+I)

> In this work we provide a \textit{systematic empirical comparison} of pretrained multilingual language models versus their monolingual counterparts with regard to their monolingual task performance. We study a set of nine typologically diverse languages with readily available pretrained monolingual models on a set of five diverse monolingual downstream tasks. We first establish if a gap between the multilingual and the corresponding monolingual representation of that language exists, and subsequently investigate the reason for a performance difference. To disentangle the impacting variables, we train new monolingual models on the same data, but with different tokenizers, both the monolingual and the multilingual version. We find that while the pretraining data size is an important factor, the designated tokenizer of the monolingual model plays an equally important role in the downstream performance. Our results show that languages which are adequately represented in the multilingual model's vocabulary exhibit negligible performance decreases over their monolingual counterparts. We further find that replacing the original multilingual tokenizer with the specialized monolingual tokenizer improves the downstream performance of the multilingual model for almost every task and language.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.15613](https://arxiv.org/abs/2012.15613) [cs.CL]** |
|           | (or **[arXiv:2012.15613v1](https://arxiv.org/abs/2012.15613v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-25">25. CoCoLM: COmplex COmmonsense Enhanced Language Model</h2>

Title: [CoCoLM: COmplex COmmonsense Enhanced Language Model](https://arxiv.org/abs/2012.15643)

Authors: [Changlong Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+C), [Hongming Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+H), [Yangqiu Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+Y), [Wilfred Ng](https://arxiv.org/search/cs?searchtype=author&query=Ng%2C+W)

> Large-scale pre-trained language models have demonstrated strong knowledge representation ability. However, recent studies suggest that even though these giant models contains rich simple commonsense knowledge (e.g., bird can fly and fish can swim.), they often struggle with the complex commonsense knowledge that involves multiple eventualities (verb-centric phrases, e.g., identifying the relationship between ``Jim yells at Bob'' and ``Bob is upset'').To address this problem, in this paper, we propose to help pre-trained language models better incorporate complex commonsense knowledge. Different from existing fine-tuning approaches, we do not focus on a specific task and propose a general language model named CoCoLM. Through the careful training over a large-scale eventuality knowledge graphs ASER, we successfully teach pre-trained language models (i.e., BERT and RoBERTa) rich complex commonsense knowledge among eventualities. Experiments on multiple downstream commonsense tasks that requires the correct understanding of eventualities demonstrate the effectiveness of CoCoLM.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.15643](https://arxiv.org/abs/2012.15643) [cs.CL]** |
|           | (or **[arXiv:2012.15643v1](https://arxiv.org/abs/2012.15643v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-26">26. VOLT: Improving Vocabularization via Optimal Transport for Machine Translation</h2>

Title: [VOLT: Improving Vocabularization via Optimal Transport for Machine Translation](https://arxiv.org/abs/2012.15671)

Authors: [Jingjing Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+J), [Hao Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+H), [Chun Gan](https://arxiv.org/search/cs?searchtype=author&query=Gan%2C+C), [Zaixiang Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+Z), [Lei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L)

> It is well accepted that the choice of token vocabulary largely affects the performance of machine translation. However, due to expensive trial costs, most studies only conduct simple trials with dominant approaches (e.g BPE) and commonly used vocabulary sizes. In this paper, we find an exciting relation between an information-theoretic feature and BLEU scores. With this observation, we formulate the quest of vocabularization -- finding the best token dictionary with a proper size -- as an optimal transport problem. We then propose VOLT, a simple and efficient vocabularization solution without the full and costly trial training. We evaluate our approach on multiple machine translation tasks, including WMT-14 English-German translation, TED bilingual translation, and TED multilingual translation. Empirical results show that VOLT beats widely-used vocabularies on diverse scenarios. For example, VOLT achieves 70% vocabulary size reduction and 0.6 BLEU gain on English-German translation. Also, one advantage of VOLT lies in its low resource consumption. Compared to naive BPE-search, VOLT reduces the search time from 288 GPU hours to 0.5 CPU hours.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.15671](https://arxiv.org/abs/2012.15671) [cs.CL]** |
|           | (or **[arXiv:2012.15671v1](https://arxiv.org/abs/2012.15671v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-27">27. ERNIE-M: Enhanced Multilingual Representation by Aligning Cross-lingual Semantics with Monolingual Corpora</h2>

Title: [ERNIE-M: Enhanced Multilingual Representation by Aligning Cross-lingual Semantics with Monolingual Corpora](https://arxiv.org/abs/2012.15674)

Authors: [Xuan Ouyang](https://arxiv.org/search/cs?searchtype=author&query=Ouyang%2C+X), [Shuohuan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Chao Pang](https://arxiv.org/search/cs?searchtype=author&query=Pang%2C+C), [Yu Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+Y), [Hao Tian](https://arxiv.org/search/cs?searchtype=author&query=Tian%2C+H), [Hua Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+H), [Haifeng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+H)

> Recent studies have demonstrated that pre-trained cross-lingual models achieve impressive performance on downstream cross-lingual tasks. This improvement stems from the learning of a large amount of monolingual and parallel corpora. While it is generally acknowledged that parallel corpora are critical for improving the model performance, existing methods are often constrained by the size of parallel corpora, especially for the low-resource languages. In this paper, we propose ERNIE-M, a new training method that encourages the model to align the representation of multiple languages with monolingual corpora, to break the constraint of parallel corpus size on the model performance. Our key insight is to integrate the idea of back translation in the pre-training process. We generate pseudo-parallel sentences pairs on a monolingual corpus to enable the learning of semantic alignment between different languages, which enhances the semantic modeling of cross-lingual models. Experimental results show that ERNIE-M outperforms existing cross-lingual models and delivers new state-of-the-art results on various cross-lingual downstream tasks. The codes and pre-trained models will be made publicly available.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.15674](https://arxiv.org/abs/2012.15674) [cs.CL]** |
|           | (or **[arXiv:2012.15674v1](https://arxiv.org/abs/2012.15674v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-28">28. Revisiting Robust Neural Machine Translation: A Transformer Case Study</h2>

Title: [Revisiting Robust Neural Machine Translation: A Transformer Case Study](https://arxiv.org/abs/2012.15710)

Authors: [Peyman Passban](https://arxiv.org/search/cs?searchtype=author&query=Passban%2C+P), [Puneeth S.M. Saladi](https://arxiv.org/search/cs?searchtype=author&query=Saladi%2C+P+S), [Qun Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Q)

> Transformers (Vaswani et al., 2017) have brought a remarkable improvement in the performance of neural machine translation (NMT) systems, but they could be surprisingly vulnerable to noise. Accordingly, we tried to investigate how noise breaks Transformers and if there exist solutions to deal with such issues. There is a large body of work in the NMT literature on analyzing the behaviour of conventional models for the problem of noise but it seems Transformers are understudied in this context.
> Therefore, we introduce a novel data-driven technique to incorporate noise during training. This idea is comparable to the well-known fine-tuning strategy. Moreover, we propose two new extensions to the original Transformer, that modify the neural architecture as well as the training process to handle noise. We evaluated our techniques to translate the English--German pair in both directions. Experimental results show that our models have a higher tolerance to noise. More specifically, they perform with no deterioration where up to 10% of entire test words are infected by noise.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.15710](https://arxiv.org/abs/2012.15710) [cs.CL]** |
|           | (or **[arXiv:2012.15710v1](https://arxiv.org/abs/2012.15710v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-29">29. FDMT: A Benchmark Dataset for Fine-grained Domain Adaptation in Machine Translation</h2>

Title: [FDMT: A Benchmark Dataset for Fine-grained Domain Adaptation in Machine Translation](https://arxiv.org/abs/2012.15717)

Authors: [Wenhao Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+W), [Shujian Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Tong Pu](https://arxiv.org/search/cs?searchtype=author&query=Pu%2C+T), [Xu Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+X), [Jian Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+J), [Wei Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+W), [Yanfeng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Jiajun Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+J)

> Previous domain adaptation research usually neglect the diversity in translation within a same domain, which is a core problem for adapting a general neural machine translation (NMT) model into a specific domain in real-world scenarios. One representative of such challenging scenarios is to deploy a translation system for a conference with a specific topic, e.g. computer networks or natural language processing, where there is usually extremely less resources due to the limited time schedule. To motivate a wide investigation in such settings, we present a real-world fine-grained domain adaptation task in machine translation (FDMT). The FDMT dataset (Zh-En) consists of four sub-domains of information technology: autonomous vehicles, AI education, real-time networks and smart phone. To be closer to reality, FDMT does not employ any in-domain bilingual training data. Instead, each sub-domain is equipped with monolingual data, bilingual dictionary and knowledge base, to encourage in-depth exploration of these available resources. Corresponding development set and test set are provided for evaluation purpose. We make quantitative experiments and deep analyses in this new setting, which benchmarks the fine-grained domain adaptation task and reveals several challenging problems that need to be addressed.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.15717](https://arxiv.org/abs/2012.15717) [cs.CL]** |
|           | (or **[arXiv:2012.15717v1](https://arxiv.org/abs/2012.15717v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-30">30. Making Pre-trained Language Models Better Few-shot Learners</h2>

Title: [Making Pre-trained Language Models Better Few-shot Learners](https://arxiv.org/abs/2012.15723)

Authors: [Tianyu Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+T), [Adam Fisch](https://arxiv.org/search/cs?searchtype=author&query=Fisch%2C+A), [Danqi Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+D)

> The recent GPT-3 model (Brown et al., 2020) achieves remarkable few-shot performance solely by leveraging a natural-language prompt and a few task demonstrations as input context. Inspired by their findings, we study few-shot learning in a more practical scenario, where we use smaller language models for which fine-tuning is computationally efficient. We present LM-BFF--better few-shot fine-tuning of language models--a suite of simple and complementary techniques for fine-tuning language models on a small number of annotated examples. Our approach includes (1) prompt-based fine-tuning together with a novel pipeline for automating prompt generation; and (2) a refined strategy for dynamically and selectively incorporating demonstrations into each context. Finally, we present a systematic evaluation for analyzing few-shot performance on a range of NLP tasks, including classification and regression. Our experiments demonstrate that our methods combine to dramatically outperform standard fine-tuning procedures in this low resource setting, achieving up to 30% absolute improvement, and 11% on average across all tasks. Our approach makes minimal assumptions on task resources and domain expertise, and hence constitutes a strong task-agnostic method for few-shot learning.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.15723](https://arxiv.org/abs/2012.15723) [cs.CL]** |
|           | (or **[arXiv:2012.15723v1](https://arxiv.org/abs/2012.15723v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-31">31. Shortformer: Better Language Modeling using Shorter Inputs</h2>

Title: [Shortformer: Better Language Modeling using Shorter Inputs](https://arxiv.org/abs/2012.15832)

Authors: [Ofir Press](https://arxiv.org/search/cs?searchtype=author&query=Press%2C+O), [Noah A. Smith](https://arxiv.org/search/cs?searchtype=author&query=Smith%2C+N+A), [Mike Lewis](https://arxiv.org/search/cs?searchtype=author&query=Lewis%2C+M)

> We explore the benefits of decreasing the input length of transformers. First, we show that initially training the model on short subsequences, before moving on to longer ones, both reduces overall training time and, surprisingly, gives a large improvement in perplexity. We then show how to improve the efficiency of recurrence methods in transformers, which let models condition on previously processed tokens (when generating sequences that are larger than the maximal length that the transformer can handle at once). Existing methods require computationally expensive relative position embeddings; we introduce a simple alternative of adding absolute position embeddings to queries and keys instead of to word embeddings, which efficiently produces superior results. By combining these techniques, we increase training speed by 65%, make generation nine times faster, and substantially improve perplexity on WikiText-103, without adding any parameters.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2012.15832](https://arxiv.org/abs/2012.15832) [cs.CL]** |
|           | (or **[arXiv:2012.15832v1](https://arxiv.org/abs/2012.15832v1) [cs.CL]** for this version) |





<h2 id="2021-01-01-32">32. Fully Non-autoregressive Neural Machine Translation: Tricks of the Trade</h2>

Title: [Fully Non-autoregressive Neural Machine Translation: Tricks of the Trade](https://arxiv.org/abs/2012.15833)

Authors: [Jiatao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+J), [Xiang Kong](https://arxiv.org/search/cs?searchtype=author&query=Kong%2C+X)

> Fully non-autoregressive neural machine translation (NAT) is proposed to simultaneously predict tokens with single forward of neural networks, which significantly reduces the inference latency at the expense of quality drop compared to the Transformer baseline. In this work, we target on closing the performance gap while maintaining the latency advantage. We first inspect the fundamental issues of fully NAT models, and adopt dependency reduction in the learning space of output tokens as the basic guidance. Then, we revisit methods in four different aspects that have been proven effective for improving NAT models, and carefully combine these techniques with necessary modifications. Our extensive experiments on three translation benchmarks show that the proposed system achieves the new state-of-the-art results for fully NAT models, and obtains comparable performance with the autoregressive and iterative NAT systems. For instance, one of the proposed models achieves 27.49 BLEU points on WMT14 En-De with approximately 16.5X speed up at inference time.

| Comments: | 9 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2012.15833](https://arxiv.org/abs/2012.15833) [cs.CL]** |
|           | (or **[arXiv:2012.15833v1](https://arxiv.org/abs/2012.15833v1) [cs.CL]** for this version) |

