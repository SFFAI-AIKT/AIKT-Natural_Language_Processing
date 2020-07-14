# Daily arXiv: Machine Translation - July, 2020

# Index


- [2020-07-14](#2020-07-14)

  - [1. TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech](#2020-07-14-1)
  - [2. Fine-grained Language Identification with Multilingual CapsNet Model](#2020-07-14-2)
  - [3. Is Machine Learning Speaking my Language? A Critical Look at the NLP-Pipeline Across 8 Human Languages](#2020-07-14-3)
  - [4. Do You Have the Right Scissors? Tailoring Pre-trained Language Models via Monte-Carlo Methods](#2020-07-14-4)
  - [5. Generating Fluent Adversarial Examples for Natural Languages](#2020-07-14-5)
  - [6. Transformer with Depth-Wise LSTM](#2020-07-14-6)
- [2020-07-13](#2020-07-13)

  - [1. Pragmatic information in translation: a corpus-based study of tense and mood in English and German](#2020-07-13-1)
  - [2. Learn to Use Future Information in Simultaneous Translation](#2020-07-13-2)
- [2020-07-10](#2020-07-10)

  - [1. Principal Word Vectors](#2020-07-10-1)
  - [2. Targeting the Benchmark: On Methodology in Current Natural Language Processing Research](#2020-07-10-2)
- [2020-07-09](#2020-07-09)

  - [1. Learning Speech Representations from Raw Audio by Joint Audiovisual Self-Supervision](2020-07-09-1)
  - [2. Best-First Beam Search](2020-07-09-2)
  - [3. A Survey on Transfer Learning in Natural Language Processing](2020-07-09-3)
- [2020-07-08](#2020-07-08-1)

  - [1. Do Transformers Need Deep Long-Range Memory](#2020-07-08-1)
  - [2. The Go Transformer: Natural Language Modeling for Game Play](#2020-07-08-2)
  - [3. scb-mt-en-th-2020: A Large English-Thai Parallel Corpus](#2020-07-08-3)
- [2020-06](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-06.md)
- [2020-05](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-05.md)
- [2020-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-04.md)
- [2020-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2020-03.md)
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



# 2020-07-14

[Return to Index](#Index)



<h2 id="2020-07-14-1">1. TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech</h2>

Title: [TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech](https://arxiv.org/abs/2007.06028)

Authors: [Andy T. Liu](https://arxiv.org/search/eess?searchtype=author&query=Liu%2C+A+T), [Shang-Wen Li](https://arxiv.org/search/eess?searchtype=author&query=Li%2C+S), [Hung-yi Lee](https://arxiv.org/search/eess?searchtype=author&query=Lee%2C+H)

> We introduce a self-supervised speech pre-training method called TERA, which stands for Transformer Encoder Representations from Alteration. Recent approaches often learn through the formulation of a single auxiliary task like contrastive prediction, autoregressive prediction, or masked reconstruction. Unlike previous approaches, we use a multi-target auxiliary task to pre-train Transformer Encoders on a large amount of unlabeled speech. The model learns through the reconstruction of acoustic frames from its altered counterpart, where we use a stochastic policy to alter along three dimensions: temporal, channel, and magnitude. TERA can be used to extract speech representations or fine-tune with downstream models. We evaluate TERA on several downstream tasks, including phoneme classification, speaker recognition, and speech recognition. TERA achieved strong performance on these tasks by improving upon surface features and outperforming previous methods. In our experiments, we show that through alteration along different dimensions, the model learns to encode distinct aspects of speech. We explore different knowledge transfer methods to incorporate the pre-trained model with downstream models. Furthermore, we show that the proposed method can be easily transferred to another dataset not used in pre-training.

| Comments: | Submitted to IEEE TASLP, currently under review              |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2007.06028](https://arxiv.org/abs/2007.06028) [eess.AS]** |
|           | (or **[arXiv:2007.06028v1](https://arxiv.org/abs/2007.06028v1) [eess.AS]** for this version) |





<h2 id="2020-07-14-2">2. Fine-grained Language Identification with Multilingual CapsNet Model</h2>

Title: [Fine-grained Language Identification with Multilingual CapsNet Model](https://arxiv.org/abs/2007.06078)

Authors: [Mudit Verma](https://arxiv.org/search/eess?searchtype=author&query=Verma%2C+M), [Arun Balaji Buduru](https://arxiv.org/search/eess?searchtype=author&query=Buduru%2C+A+B)

> Due to a drastic improvement in the quality of internet services worldwide, there is an explosion of multilingual content generation and consumption. This is especially prevalent in countries with large multilingual audience, who are increasingly consuming media outside their linguistic familiarity/preference. Hence, there is an increasing need for real-time and fine-grained content analysis services, including language identification, content transcription, and analysis. Accurate and fine-grained spoken language detection is an essential first step for all the subsequent content analysis algorithms. Current techniques in spoken language detection may lack on one of these fronts: accuracy, fine-grained detection, data requirements, manual effort in data collection \& pre-processing. Hence in this work, a real-time language detection approach to detect spoken language from 5 seconds' audio clips with an accuracy of 91.8\% is presented with exiguous data requirements and minimal pre-processing. Novel architectures for Capsule Networks is proposed which operates on spectrogram images of the provided audio snippets. We use previous approaches based on Recurrent Neural Networks and iVectors to present the results. Finally we show a ``Non-Class'' analysis to further stress on why CapsNet architecture works for LID task.

| Comments: | 5 pages, 6 figures                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL); Sound (cs.SD) |
| Cite as:  | **[arXiv:2007.06078](https://arxiv.org/abs/2007.06078) [eess.AS]** |
|           | (or **[arXiv:2007.06078v1](https://arxiv.org/abs/2007.06078v1) [eess.AS]** for this version) |





<h2 id="2020-07-14-3">3. Is Machine Learning Speaking my Language? A Critical Look at the NLP-Pipeline Across 8 Human Languages</h2>

Title: [Is Machine Learning Speaking my Language? A Critical Look at the NLP-Pipeline Across 8 Human Languages](https://arxiv.org/abs/2007.05872)

Authors: [Esma Wali](https://arxiv.org/search/cs?searchtype=author&query=Wali%2C+E), [Yan Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Christopher Mahoney](https://arxiv.org/search/cs?searchtype=author&query=Mahoney%2C+C), [Thomas Middleton](https://arxiv.org/search/cs?searchtype=author&query=Middleton%2C+T), [Marzieh Babaeianjelodar](https://arxiv.org/search/cs?searchtype=author&query=Babaeianjelodar%2C+M), [Mariama Njie](https://arxiv.org/search/cs?searchtype=author&query=Njie%2C+M), [Jeanna Neefe Matthews](https://arxiv.org/search/cs?searchtype=author&query=Matthews%2C+J+N)

> Natural Language Processing (NLP) is increasingly used as a key ingredient in critical decision-making systems such as resume parsers used in sorting a list of job candidates. NLP systems often ingest large corpora of human text, attempting to learn from past human behavior and decisions in order to produce systems that will make recommendations about our future world. Over 7000 human languages are being spoken today and the typical NLP pipeline underrepresents speakers of most of them while amplifying the voices of speakers of other languages. In this paper, a team including speakers of 8 languages - English, Chinese, Urdu, Farsi, Arabic, French, Spanish, and Wolof - takes a critical look at the typical NLP pipeline and how even when a language is technically supported, substantial caveats remain to prevent full participation. Despite huge and admirable investments in multilingual support in many tools and resources, we are still making NLP-guided decisions that systematically and dramatically underrepresent the voices of much of the world.

| Comments:    | Participatory Approaches to Machine Learning Workshop, 37th International Conference on Machine Learning |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| ACM classes: | I.2.7                                                        |
| Cite as:     | **[arXiv:2007.05872](https://arxiv.org/abs/2007.05872) [cs.CL]** |
|              | (or **[arXiv:2007.05872v1](https://arxiv.org/abs/2007.05872v1) [cs.CL]** for this version) |





<h2 id="2020-07-14-4">4. Do You Have the Right Scissors? Tailoring Pre-trained Language Models via Monte-Carlo Methods</h2>

Title: [Do You Have the Right Scissors? Tailoring Pre-trained Language Models via Monte-Carlo Methods](https://arxiv.org/abs/2007.06162)

Authors: [Ning Miao](https://arxiv.org/search/cs?searchtype=author&query=Miao%2C+N), [Yuxuan Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+Y), [Hao Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+H), [Lei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L)

> It has been a common approach to pre-train a language model on a large corpus and fine-tune it on task-specific data. In practice, we observe that fine-tuning a pre-trained model on a small dataset may lead to over- and/or under-estimation problem. In this paper, we propose MC-Tailor, a novel method to alleviate the above issue in text generation tasks by truncating and transferring the probability mass from over-estimated regions to under-estimated ones. Experiments on a variety of text generation datasets show that MC-Tailor consistently and significantly outperforms the fine-tuning approach. Our code is available at this url.

| Comments: | Accepted by ACL 2020                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2007.06162](https://arxiv.org/abs/2007.06162) [cs.CL]** |
|           | (or **[arXiv:2007.06162v1](https://arxiv.org/abs/2007.06162v1) [cs.CL]** for this version) |





<h2 id="2020-07-14-5">5. Generating Fluent Adversarial Examples for Natural Languages</h2>

Title: [Generating Fluent Adversarial Examples for Natural Languages](https://arxiv.org/abs/2007.06174)

Authors: [Huangzhao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+H), [Hao Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+H), [Ning Miao](https://arxiv.org/search/cs?searchtype=author&query=Miao%2C+N), [Lei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L)

> Efficiently building an adversarial attacker for natural language processing (NLP) tasks is a real challenge. Firstly, as the sentence space is discrete, it is difficult to make small perturbations along the direction of gradients. Secondly, the fluency of the generated examples cannot be guaranteed. In this paper, we propose MHA, which addresses both problems by performing Metropolis-Hastings sampling, whose proposal is designed with the guidance of gradients. Experiments on IMDB and SNLI show that our proposed MHA outperforms the baseline model on attacking capability. Adversarial training with MAH also leads to better robustness and performance.

| Comments: | Accepted by ACL 2019                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2007.06174](https://arxiv.org/abs/2007.06174) [cs.CL]** |
|           | (or **[arXiv:2007.06174v1](https://arxiv.org/abs/2007.06174v1) [cs.CL]** for this version) |





<h2 id="2020-07-14-6">6. Transformer with Depth-Wise LSTM</h2>

Title: [Transformer with Depth-Wise LSTM](https://arxiv.org/abs/2007.06257)

Authors: [Hongfei Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+H), [Qiuhui Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Q), [Deyi Xiong](https://arxiv.org/search/cs?searchtype=author&query=Xiong%2C+D), [Josef van Genabith](https://arxiv.org/search/cs?searchtype=author&query=van+Genabith%2C+J)

> Increasing the depth of models allows neural models to model complicated functions but may also lead to optimization issues. The Transformer translation model employs the residual connection to ensure its convergence. In this paper, we suggest that the residual connection has its drawbacks, and propose to train Transformers with the depth-wise LSTM which regards outputs of layers as steps in time series instead of residual connections, under the motivation that the vanishing gradient problem suffered by deep networks is the same as recurrent networks applied to long sequences, while LSTM (Hochreiter and Schmidhuber, 1997) has been proven of good capability in capturing long-distance relationship, and its design may alleviate some drawbacks of residual connections while ensuring the convergence. We integrate the computation of multi-head attention networks and feed-forward networks with the depth-wise LSTM for the Transformer, which shows how to utilize the depth-wise LSTM like the residual connection. Our experiment with the 6-layer Transformer shows that our approach can bring about significant BLEU improvements in both WMT 14 English-German and English-French tasks, and our deep Transformer experiment demonstrates the effectiveness of the depth-wise LSTM on the convergence of deep Transformers. Additionally, we propose to measure the impacts of the layer's non-linearity on the performance by distilling the analyzing layer of the trained model into a linear transformation and observing the performance degradation with the replacement. Our analysis results support the more efficient use of per-layer non-linearity with depth-wise LSTM than with residual connections.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2007.06257](https://arxiv.org/abs/2007.06257) [cs.CL]** |
|           | (or **[arXiv:2007.06257v1](https://arxiv.org/abs/2007.06257v1) [cs.CL]** for this version) |





# 2020-07-13

[Return to Index](#Index)



<h2 id="2020-07-13-1">1. Pragmatic information in translation: a corpus-based study of tense and mood in English and German
</h2>

Title: [Pragmatic information in translation: a corpus-based study of tense and mood in English and German](https://arxiv.org/abs/2007.05234)

Authors: [Anita Ramm](https://arxiv.org/search/cs?searchtype=author&query=Ramm%2C+A), [Ekaterina Lapshinova-Koltunski](https://arxiv.org/search/cs?searchtype=author&query=Lapshinova-Koltunski%2C+E), [Alexander Fraser](https://arxiv.org/search/cs?searchtype=author&query=Fraser%2C+A)

> Grammatical tense and mood are important linguistic phenomena to consider in natural language processing (NLP) research. We consider the correspondence between English and German tense and mood in translation. Human translators do not find this correspondence easy, and as we will show through careful analysis, there are no simplistic ways to map tense and mood from one language to another. Our observations about the challenges of human translation of tense and mood have important implications for multilingual NLP. Of particular importance is the challenge of modeling tense and mood in rule-based, phrase-based statistical and neural machine translation.

| Comments:    | Technical Report of CIS, LMU Munich. September 19th, 2019    |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| ACM classes: | I.1.2                                                        |
| Cite as:     | **[arXiv:2007.05234](https://arxiv.org/abs/2007.05234) [cs.CL]** |
|              | (or **[arXiv:2007.05234v1](https://arxiv.org/abs/2007.05234v1) [cs.CL]** for this version) |





<h2 id="2020-07-13-2">2. Learn to Use Future Information in Simultaneous Translation</h2>

Title: [Learn to Use Future Information in Simultaneous Translation](https://arxiv.org/abs/2007.05290)

Authors: [Xueqing Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+X), [Yingce Xia](https://arxiv.org/search/cs?searchtype=author&query=Xia%2C+Y), [Lijun Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+L), [Shufang Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+S), [Weiqing Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+W), [Jiang Bian](https://arxiv.org/search/cs?searchtype=author&query=Bian%2C+J), [Tao Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+T), [Tie-Yan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+T)

> Simultaneous neural machine translation (briefly, NMT) has attracted much attention recently. In contrast to standard NMT, where the NMT system can utilize the full input sentence, simultaneous NMT is formulated as a prefix-to-prefix problem, where the system can only utilize the prefix of the input sentence and more uncertainty is introduced to decoding. Wait-k is a simple yet effective strategy for simultaneous NMT, where the decoder generates the output sequence k words behind the input words. We observed that training simultaneous NMT systems with future information (i.e., trained with a larger k) generally outperforms the standard ones (i.e., trained with the given k). Based on this observation, we propose a framework that automatically learns how much future information to use in training for simultaneous NMT. We first build a series of tasks where each one is associated with a different k, and then learn a model on these tasks guided by a controller. The controller is jointly trained with the translation model through bi-level optimization. We conduct experiments on four datasets to demonstrate the effectiveness of our method.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2007.05290](https://arxiv.org/abs/2007.05290) [cs.CL]** |
|           | (or **[arXiv:2007.05290v1](https://arxiv.org/abs/2007.05290v1) [cs.CL]** for this version) |



# 2020-07-10

[Return to Index](#Index)



<h2 id="2020-07-10-1">1. Principal Word Vectors</h2>

Title: [Principal Word Vectors](https://arxiv.org/abs/2007.04629)

Authors: [Ali Basirat](https://arxiv.org/search/cs?searchtype=author&query=Basirat%2C+A), [Christian Hardmeier](https://arxiv.org/search/cs?searchtype=author&query=Hardmeier%2C+C), [Joakim Nivre](https://arxiv.org/search/cs?searchtype=author&query=Nivre%2C+J)

> We generalize principal component analysis for embedding words into a vector space. The generalization is made in two major levels. The first is to generalize the concept of the corpus as a counting process which is defined by three key elements vocabulary set, feature (annotation) set, and context. This generalization enables the principal word embedding method to generate word vectors with regard to different types of contexts and different types of annotations provided for a corpus. The second is to generalize the transformation step used in most of the word embedding methods. To this end, we define two levels of transformations. The first is a quadratic transformation, which accounts for different types of weighting over the vocabulary units and contextual features. Second is an adaptive non-linear transformation, which reshapes the data distribution to be meaningful to principal component analysis. The effect of these generalizations on the word vectors is intrinsically studied with regard to the spread and the discriminability of the word vectors. We also provide an extrinsic evaluation of the contribution of the principal word vectors on a word similarity benchmark and the task of dependency parsing. Our experiments are finalized by a comparison between the principal word vectors and other sets of word vectors generated with popular word embedding methods. The results obtained from our intrinsic evaluation metrics show that the spread and the discriminability of the principal word vectors are higher than that of other word embedding methods. The results obtained from the extrinsic evaluation metrics show that the principal word vectors are better than some of the word embedding methods and on par with popular methods of word embedding.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2007.04629](https://arxiv.org/abs/2007.04629) [cs.CL]** |
|           | (or **[arXiv:2007.04629v1](https://arxiv.org/abs/2007.04629v1) [cs.CL]** for this version) |





<h2 id="2020-07-10-2">2. Targeting the Benchmark: On Methodology in Current Natural Language Processing Research</h2>

Title: [Targeting the Benchmark: On Methodology in Current Natural Language Processing Research](https://arxiv.org/abs/2007.04792)

Authors: [David Schlangen](https://arxiv.org/search/cs?searchtype=author&query=Schlangen%2C+D)

> It has become a common pattern in our field: One group introduces a language task, exemplified by a dataset, which they argue is challenging enough to serve as a benchmark. They also provide a baseline model for it, which then soon is improved upon by other groups. Often, research efforts then move on, and the pattern repeats itself. What is typically left implicit is the argumentation for why this constitutes progress, and progress towards what. In this paper, we try to step back for a moment from this pattern and work out possible argumentations and their parts.

| Comments: | arXiv admin note: text overlap with [arXiv:1908.10747](https://arxiv.org/abs/1908.10747) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2007.04792](https://arxiv.org/abs/2007.04792) [cs.CL]** |
|           | (or **[arXiv:2007.04792v1](https://arxiv.org/abs/2007.04792v1) [cs.CL]** for this version) |



# 2020-07-09

[Return to Index](#Index)



<h2 id="2020-07-09-1">1. Learning Speech Representations from Raw Audio by Joint Audiovisual Self-Supervision</h2>

Title: [Learning Speech Representations from Raw Audio by Joint Audiovisual Self-Supervision](https://arxiv.org/abs/2007.04134)

Authors: [Abhinav Shukla](https://arxiv.org/search/eess?searchtype=author&query=Shukla%2C+A), [Stavros Petridis](https://arxiv.org/search/eess?searchtype=author&query=Petridis%2C+S), [Maja Pantic](https://arxiv.org/search/eess?searchtype=author&query=Pantic%2C+M)

> The intuitive interaction between the audio and visual modalities is valuable for cross-modal self-supervised learning. This concept has been demonstrated for generic audiovisual tasks like video action recognition and acoustic scene classification. However, self-supervision remains under-explored for audiovisual speech. We propose a method to learn self-supervised speech representations from the raw audio waveform. We train a raw audio encoder by combining audio-only self-supervision (by predicting informative audio attributes) with visual self-supervision (by generating talking faces from audio). The visual pretext task drives the audio representations to capture information related to lip movements. This enriches the audio encoder with visual information and the encoder can be used for evaluation without the visual modality. Our method attains competitive performance with respect to existing self-supervised audio features on established isolated word classification benchmarks, and significantly outperforms other methods at learning from fewer labels. Notably, our method also outperforms fully supervised training, thus providing a strong initialization for speech related tasks. Our results demonstrate the potential of multimodal self-supervision in audiovisual speech for learning good audio representations.

| Comments: | Accepted at the Workshop on Self-supervision in Audio and Speech at ICML 2020 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG); Sound (cs.SD) |
| Cite as:  | **[arXiv:2007.04134](https://arxiv.org/abs/2007.04134) [eess.AS]** |
|           | (or **[arXiv:2007.04134v1](https://arxiv.org/abs/2007.04134v1) [eess.AS]** for this version) |





<h2 id="2020-07-09-2">2. Best-First Beam Search</h2>

Title: [Best-First Beam Search](https://arxiv.org/abs/2007.03909)

Authors: [Clara Meister](https://arxiv.org/search/cs?searchtype=author&query=Meister%2C+C), [Tim Vieira](https://arxiv.org/search/cs?searchtype=author&query=Vieira%2C+T), [Ryan Cotterell](https://arxiv.org/search/cs?searchtype=author&query=Cotterell%2C+R)

> Decoding for many NLP tasks requires a heuristic algorithm for approximating exact search since the full search space is often intractable if not simply too large to traverse efficiently. The default algorithm for this job is beam search--a pruned version of breadth-first search--which in practice, returns better results than exact inference due to beneficial search bias. In this work, we show that standard beam search is a computationally inefficient choice for many decoding tasks; specifically, when the scoring function is a monotonic function in sequence length, other search algorithms can be used to reduce the number of calls to the scoring function (e.g., a neural network), which is often the bottleneck computation. We propose best-first beam search, an algorithm that provably returns the same set of results as standard beam search, albeit in the minimum number of scoring function calls to guarantee optimality (modulo beam size). We show that best-first beam search can be used with length normalization and mutual information decoding, among other rescoring functions. Lastly, we propose a memory-reduced variant of best-first beam search, which has a similar search bias in terms of downstream performance, but runs in a fraction of the time.

| Comments: | TACL 2020                                                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Data Structures and Algorithms (cs.DS) |
| Cite as:  | **[arXiv:2007.03909](https://arxiv.org/abs/2007.03909) [cs.CL]** |
|           | (or **[arXiv:2007.03909v2](https://arxiv.org/abs/2007.03909v2) [cs.CL]** for this version) |





<h2 id="2020-07-09-3">3. A Survey on Transfer Learning in Natural Language Processing</h2>

Title: [A Survey on Transfer Learning in Natural Language Processing](https://arxiv.org/abs/2007.04239)

Authors: [Zaid Alyafeai](https://arxiv.org/search/cs?searchtype=author&query=Alyafeai%2C+Z), [Maged Saeed AlShaibani](https://arxiv.org/search/cs?searchtype=author&query=AlShaibani%2C+M+S), [Irfan Ahmad](https://arxiv.org/search/cs?searchtype=author&query=Ahmad%2C+I)

> Deep learning models usually require a huge amount of data. However, these large datasets are not always attainable. This is common in many challenging NLP tasks. Consider Neural Machine Translation, for instance, where curating such large datasets may not be possible specially for low resource languages. Another limitation of deep learning models is the demand for huge computing resources. These obstacles motivate research to question the possibility of knowledge transfer using large trained models. The demand for transfer learning is increasing as many large models are emerging. In this survey, we feature the recent transfer learning advances in the field of NLP. We also provide a taxonomy for categorizing different transfer learning approaches from the literature.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2007.04239](https://arxiv.org/abs/2007.04239) [cs.CL]** |
|           | (or **[arXiv:2007.04239v1](https://arxiv.org/abs/2007.04239v1) [cs.CL]** for this version) |



# 2020-07-08

[Return to Index](#Index)



<h2 id="2020-07-08-1">1. Do Transformers Need Deep Long-Range Memory</h2>

Title: [Do Transformers Need Deep Long-Range Memory](https://arxiv.org/abs/2007.03356)

Authors: [Jack W. Rae](https://arxiv.org/search/cs?searchtype=author&query=Rae%2C+J+W), [Ali Razavi](https://arxiv.org/search/cs?searchtype=author&query=Razavi%2C+A)

> Deep attention models have advanced the modelling of sequential data across many domains. For language modelling in particular, the Transformer-XL -- a Transformer augmented with a long-range memory of past activations -- has been shown to be state-of-the-art across a variety of well-studied benchmarks. The Transformer-XL incorporates a long-range memory at every layer of the network, which renders its state to be thousands of times larger than RNN predecessors. However it is unclear whether this is necessary. We perform a set of interventions to show that comparable performance can be obtained with 6X fewer long range memories and better performance can be obtained by limiting the range of attention in lower layers of the network.

| Comments: | published at 58th Annual Meeting of the Association for Computational Linguistics. 6 pages, 4 figures, 1 table |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Machine Learning (stat.ML) |
| Cite as:  | **[arXiv:2007.03356](https://arxiv.org/abs/2007.03356) [cs.LG]** |
|           | (or **[arXiv:2007.03356v1](https://arxiv.org/abs/2007.03356v1) [cs.LG]** for this version) |





<h2 id="2020-07-08-2">2. The Go Transformer: Natural Language Modeling for Game Play</h2>

Title: [The Go Transformer: Natural Language Modeling for Game Play](https://arxiv.org/abs/2007.03500)

Authors: [David Noever](https://arxiv.org/search/cs?searchtype=author&query=Noever%2C+D), [Matthew Ciolino](https://arxiv.org/search/cs?searchtype=author&query=Ciolino%2C+M), [Josh Kalin](https://arxiv.org/search/cs?searchtype=author&query=Kalin%2C+J)

> This work applies natural language modeling to generate plausible strategic moves in the ancient game of Go. We train the Generative Pretrained Transformer (GPT-2) to mimic the style of Go champions as archived in Smart Game Format (SGF), which offers a text description of move sequences. The trained model further generates valid but previously unseen strategies for Go. Because GPT-2 preserves punctuation and spacing, the raw output of the text generator provides inputs to game visualization and creative patterns, such as the Sabaki project's (2020) game engine using auto-replays. Results demonstrate that language modeling can capture both the sequencing format of championship Go games and their strategic formations. Compared to random game boards, the GPT-2 fine-tuning shows efficient opening move sequences favoring corner play over less advantageous center and side play. Game generation as a language modeling task offers novel approaches to more than 40 other board games where historical text annotation provides training data (e.g., Amazons & Connect 4/6).

| Comments: | 8 Pages, 5 Figures, 1 Table                                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2007.03500](https://arxiv.org/abs/2007.03500) [cs.CL]** |
|           | (or **[arXiv:2007.03500v1](https://arxiv.org/abs/2007.03500v1) [cs.CL]** for this version) |







<h2 id="2020-07-08-3">3. scb-mt-en-th-2020: A Large English-Thai Parallel Corpus</h2>

Title: [scb-mt-en-th-2020: A Large English-Thai Parallel Corpus](https://arxiv.org/abs/2007.03541)

Authors: [Lalita Lowphansirikul](https://arxiv.org/search/cs?searchtype=author&query=Lowphansirikul%2C+L), [Charin Polpanumas](https://arxiv.org/search/cs?searchtype=author&query=Polpanumas%2C+C), [Attapol T. Rutherford](https://arxiv.org/search/cs?searchtype=author&query=Rutherford%2C+A+T), [Sarana Nutanong](https://arxiv.org/search/cs?searchtype=author&query=Nutanong%2C+S)

> The primary objective of our work is to build a large-scale English-Thai dataset for machine translation. We construct an English-Thai machine translation dataset with over 1 million segment pairs, curated from various sources, namely news, Wikipedia articles, SMS messages, task-based dialogs, web-crawled data and government documents. Methodology for gathering data, building parallel texts and removing noisy sentence pairs are presented in a reproducible manner. We train machine translation models based on this dataset. Our models' performance are comparable to that of Google Translation API (as of May 2020) for Thai-English and outperform Google when the Open Parallel Corpus (OPUS) is included in the training data for both Thai-English and English-Thai translation. The dataset, pre-trained models, and source code to reproduce our work are available for public use.

| Comments: | 35 pages, 4 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2007.03541](https://arxiv.org/abs/2007.03541) [cs.CL]** |
|           | (or **[arXiv:2007.03541v1](https://arxiv.org/abs/2007.03541v1) [cs.CL]** for this version) |



