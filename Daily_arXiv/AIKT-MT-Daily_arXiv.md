# Daily arXiv: Machine Translation - May, 2021

# Index


- [2021-05-06](#2021-05-06)
  
  - [1. Data Augmentation by Concatenation for Low-Resource Translation: A Mystery and a Solution](#2021-05-06-1)
  - [2. Full-Sentence Models Perform Better in Simultaneous Translation Using the Information Enhanced Decoding Strategy](#2021-05-06-2)
- [2021-05-04](#2021-05-04)	
  - [1. AlloST: Low-resource Speech Translation without Source Transcription](#2021-05-04-1)
  - [2. Larger-Scale Transformers for Multilingual Masked Language Modeling](#2021-05-04-2)
  - [3. Transformers: "The End of History" for NLP?](#2021-05-04-3)
  - [4. BERT memorisation and pitfalls in low-resource scenarios](#2021-05-04-4)
  - [5. Natural Language Generation Using Link Grammar for General Conversational Intelligence](#2021-05-04-5)
- [Other Columns](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-05-04

[Return to Index](#Index)



<h2 id="2021-05-04-1">1. Data Augmentation by Concatenation for Low-Resource Translation: A Mystery and a Solution

</h2>

Title: [Data Augmentation by Concatenation for Low-Resource Translation: A Mystery and a Solution](https://arxiv.org/abs/2105.01691)

Authors: [Toan Q. Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+T+Q), [Kenton Murray](https://arxiv.org/search/cs?searchtype=author&query=Murray%2C+K), [David Chiang](https://arxiv.org/search/cs?searchtype=author&query=Chiang%2C+D)

> In this paper, we investigate the driving factors behind concatenation, a simple but effective data augmentation method for low-resource neural machine translation. Our experiments suggest that discourse context is unlikely the cause for the improvement of about +1 BLEU across four language pairs. Instead, we demonstrate that the improvement comes from three other factors unrelated to discourse: context diversity, length diversity, and (to a lesser extent) position shifting.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2105.01691](https://arxiv.org/abs/2105.01691) [cs.CL]** |
|           | (or **[arXiv:2105.01691v1](https://arxiv.org/abs/2105.01691v1) [cs.CL]** for this version) |



<h2 id="2021-05-04-2">2. Full-Sentence Models Perform Better in Simultaneous Translation Using the Information Enhanced Decoding Strategy
</h2>

Title: [Full-Sentence Models Perform Better in Simultaneous Translation Using the Information Enhanced Decoding Strategy](https://arxiv.org/abs/2105.01893)

Authors: [Zhengxin Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z)

> Simultaneous translation, which starts translating each sentence after receiving only a few words in source sentence, has a vital role in many scenarios. Although the previous prefix-to-prefix framework is considered suitable for simultaneous translation and achieves good performance, it still has two inevitable drawbacks: the high computational resource costs caused by the need to train a separate model for each latency k and the insufficient ability to encode information because each target token can only attend to a specific source prefix. We propose a novel framework that adopts a simple but effective decoding strategy which is designed for full-sentence models. Within this framework, training a single full-sentence model can achieve arbitrary given latency and save computational resources. Besides, with the competence of the full-sentence model to encode the whole sentence, our decoding strategy can enhance the information maintained in the decoded states in real time. Experimental results show that our method achieves better translation quality than baselines on 4 directions: Zh→En, En→Ro and En↔De.

| Comments: | 8 pages, 5 figures                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2105.01893](https://arxiv.org/abs/2105.01893) [cs.CL]** |
|           | (or **[arXiv:2105.01893v1](https://arxiv.org/abs/2105.01893v1) [cs.CL]** for this version) |









# 2021-05-04

[Return to Index](#Index)



<h2 id="2021-05-04-1">1. AlloST: Low-resource Speech Translation without Source Transcription
</h2>


Title: [AlloST: Low-resource Speech Translation without Source Transcription](https://arxiv.org/abs/2105.00171)

Authors: [Yao-Fei Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng%2C+Y), [Hung-Shin Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+H), [Hsin-Min Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+H)

> The end-to-end architecture has made promising progress in speech translation (ST). However, the ST task is still challenging under low-resource conditions. Most ST models have shown unsatisfactory results, especially in the absence of word information from the source speech utterance. In this study, we survey methods to improve ST performance without using source transcription, and propose a learning framework that utilizes a language-independent universal phone recognizer. The framework is based on an attention-based sequence-to-sequence model, where the encoder generates the phonetic embeddings and phone-aware acoustic representations, and the decoder controls the fusion of the two embedding streams to produce the target token sequence. In addition to investigating different fusion strategies, we explore the specific usage of byte pair encoding (BPE), which compresses a phone sequence into a syllable-like segmented sequence with semantic information. Experiments conducted on the Fisher Spanish-English and Taigi-Mandarin drama corpora show that our method outperforms the conformer-based baseline, and the performance is close to that of the existing best method using source transcription.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2105.00171](https://arxiv.org/abs/2105.00171) [cs.CL]** |
|           | (or **[arXiv:2105.00171v1](https://arxiv.org/abs/2105.00171v1) [cs.CL]** for this version) |



<h2 id="2021-05-04-2">2. Larger-Scale Transformers for Multilingual Masked Language Modeling
</h2>


Title: [Larger-Scale Transformers for Multilingual Masked Language Modeling](https://arxiv.org/abs/2105.00572)

Authors: [Naman Goyal](https://arxiv.org/search/cs?searchtype=author&query=Goyal%2C+N), [Jingfei Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+J), [Myle Ott](https://arxiv.org/search/cs?searchtype=author&query=Ott%2C+M), [Giri Anantharaman](https://arxiv.org/search/cs?searchtype=author&query=Anantharaman%2C+G), [Alexis Conneau](https://arxiv.org/search/cs?searchtype=author&query=Conneau%2C+A)

> Recent work has demonstrated the effectiveness of cross-lingual language model pretraining for cross-lingual understanding. In this study, we present the results of two larger multilingual masked language models, with 3.5B and 10.7B parameters. Our two new models dubbed XLM-R XL and XLM-R XXL outperform XLM-R by 1.8% and 2.4% average accuracy on XNLI. Our model also outperforms the RoBERTa-Large model on several English tasks of the GLUE benchmark by 0.3% on average while handling 99 more languages. This suggests pretrained models with larger capacity may obtain both strong performance on high-resource languages while greatly improving low-resource languages. We make our code and models publicly available.

| Comments: | 4 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.00572](https://arxiv.org/abs/2105.00572) [cs.CL]** |
|           | (or **[arXiv:2105.00572v1](https://arxiv.org/abs/2105.00572v1) [cs.CL]** for this version) |





<h2 id="2021-05-04-3">3. Transformers: "The End of History" for NLP?
</h2>


Title: [Transformers: "The End of History" for NLP?](https://arxiv.org/abs/2105.00813)

Authors: [Anton Chernyavskiy](https://arxiv.org/search/cs?searchtype=author&query=Chernyavskiy%2C+A), [Dmitry Ilvovsky](https://arxiv.org/search/cs?searchtype=author&query=Ilvovsky%2C+D), [Preslav Nakov](https://arxiv.org/search/cs?searchtype=author&query=Nakov%2C+P)

> Recent advances in neural architectures, such as the Transformer, coupled with the emergence of large-scale pre-trained models such as BERT, have revolutionized the field of Natural Language Processing (NLP), pushing the state-of-the-art for a number of NLP tasks. A rich family of variations of these models has been proposed, such as RoBERTa, ALBERT, and XLNet, but fundamentally, they all remain limited in their ability to model certain kinds of information, and they cannot cope with certain information sources, which was easy for pre-existing models. Thus, here we aim to shed some light on some important theoretical limitations of pre-trained BERT-style models that are inherent in the general Transformer architecture. First, we demonstrate in practice on two general types of tasks -- segmentation and segment labeling -- and four datasets that these limitations are indeed harmful and that addressing them, even in some very simple and naive ways, can yield sizable improvements over vanilla RoBERTa and XLNet. Then, we offer a more general discussion on desiderata for future additions to the Transformer architecture that would increase its expressiveness, which we hope could help in the design of the next generation of deep NLP architectures.

| Comments:    | Transformers, NLP, BERT, RoBERTa, XLNet                      |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**; Information Retrieval (cs.IR); Machine Learning (cs.LG) |
| MSC classes: | 68T50                                                        |
| ACM classes: | I.2.7                                                        |
| Cite as:     | **[arXiv:2105.00813](https://arxiv.org/abs/2105.00813) [cs.CL]** |
|              | (or **[arXiv:2105.00813v1](https://arxiv.org/abs/2105.00813v1) [cs.CL]** for this version) |





<h2 id="2021-05-04-4">4. BERT memorisation and pitfalls in low-resource scenarios
</h2>


Title: [BERT memorisation and pitfalls in low-resource scenarios](https://arxiv.org/abs/2105.00828)

Authors: [Michael Tänzer](https://arxiv.org/search/cs?searchtype=author&query=Tänzer%2C+M), [Sebastian Ruder](https://arxiv.org/search/cs?searchtype=author&query=Ruder%2C+S), [Marek Rei](https://arxiv.org/search/cs?searchtype=author&query=Rei%2C+M)

> State-of-the-art pre-trained models have been shown to memorise facts and perform well with limited amounts of training data. To gain a better understanding of how these models learn, we study their generalisation and memorisation capabilities in noisy and low-resource scenarios. We find that the training of these models is almost unaffected by label noise and that it is possible to reach near-optimal performances even on extremely noisy datasets. Conversely, we also find that they completely fail when tested on low-resource tasks such as few-shot learning and rare entity recognition. To mitigate such limitations, we propose a novel architecture based on BERT and prototypical networks that improves performance in low-resource named entity recognition tasks.

| Comments: | 14 pages, 24 figures                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2105.00828](https://arxiv.org/abs/2105.00828) [cs.CL]** |
|           | (or **[arXiv:2105.00828v1](https://arxiv.org/abs/2105.00828v1) [cs.CL]** for this version) |





<h2 id="2021-05-04-5">5. Natural Language Generation Using Link Grammar for General Conversational Intelligence
</h2>


Title: [Natural Language Generation Using Link Grammar for General Conversational Intelligence](https://arxiv.org/abs/2105.00830)

Authors: [Vignav Ramesh](https://arxiv.org/search/cs?searchtype=author&query=Ramesh%2C+V), [Anton Kolonin](https://arxiv.org/search/cs?searchtype=author&query=Kolonin%2C+A)

> Many current artificial general intelligence (AGI) and natural language processing (NLP) architectures do not possess general conversational intelligence--that is, they either do not deal with language or are unable to convey knowledge in a form similar to the human language without manual, labor-intensive methods such as template-based customization. In this paper, we propose a new technique to automatically generate grammatically valid sentences using the Link Grammar database. This natural language generation method far outperforms current state-of-the-art baselines and may serve as the final component in a proto-AGI question answering pipeline that understandably handles natural language material.

| Comments: | 17 pages, 5 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2105.00830](https://arxiv.org/abs/2105.00830) [cs.CL]** |
|           | (or **[arXiv:2105.00830v1](https://arxiv.org/abs/2105.00830v1) [cs.CL]** for this version) |

