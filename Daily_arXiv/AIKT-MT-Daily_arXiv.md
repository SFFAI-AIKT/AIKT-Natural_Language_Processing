# Daily arXiv: Machine Translation - July, 2020

# Index


- [2020-07-14](#2020-07-14)

  - [1. TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech](#2020-07-14-1)
  - [2. Fine-grained Language Identification with Multilingual CapsNet Model](#2020-07-14-2)
  - [3. Is Machine Learning Speaking my Language? A Critical Look at the NLP-Pipeline Across 8 Human Languages](#2020-07-14-3)
  - [4. Do You Have the Right Scissors? Tailoring Pre-trained Language Models via Monte-Carlo Methods](#2020-07-14-4)
  - [5. Generating Fluent Adversarial Examples for Natural Languages](#2020-07-14-5)
  - [6. Transformer with Depth-Wise LSTM](#2020-07-14-6)

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

