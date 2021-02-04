# Daily arXiv: Machine Translation - February, 2021

# Index


- [2021-02-04](#2021-02-04)

  - [1. The Multilingual TEDx Corpus for Speech Recognition and Translation](#2021-02-04-1)
  - [2. Memorization vs. Generalization: Quantifying Data Leakage in NLP Performance Evaluation](#2021-02-04-2)
  - [3. When Can Models Learn From Explanations? A Formal Framework for Understanding the Roles of Explanation Data](#2021-02-04-3)
- [2021-02-03](#2021-02-03)

  - [1. Two Demonstrations of the Machine Translation Applications to Historical Documents](#2021-02-03-1)
  - [2. CTC-based Compression for Direct Speech Translation](#2021-02-03-2)
  - [3. The GEM Benchmark: Natural Language Generation, its Evaluation and Metrics](#2021-02-03-3)
- [2021-02-02](#2021-02-02)

  - [1. Speech Recognition by Simply Fine-tuning BERT](#2021-02-02-1)
  - [2. Phoneme-BERT: Joint Language Modelling of Phoneme Sequence and ASR Transcript](#2021-02-02-2)
  - [3. Machine Translationese: Effects of Algorithmic Bias on Linguistic Complexity in Machine Translation](#2021-02-02-3)
  - [4. Decoupling the Role of Data, Attention, and Losses in Multimodal Transformers](#2021-02-02-4)
  - [5. Neural OCR Post-Hoc Correction of Historical Corpora](#2021-02-02-5)
  - [6. GTAE: Graph-Transformer based Auto-Encoders for Linguistic-Constrained Text Style Transfer](#2021-02-02-6)
  - [7. Multilingual LAMA: Investigating Knowledge in Multilingual Pretrained Language Models](#2021-02-02-7)
  - [8. End2End Acoustic to Semantic Transduction](#2021-02-02-8)
  - [9. Measuring and Improving Consistency in Pretrained Language Models](#2021-02-02-9)
- [2021-02-01](#2021-02-01)

  - [1. Combining pre-trained language models and structured knowledge](#2021-02-01-1)
  - [2. Few-Shot Domain Adaptation for Grammatical Error Correction via Meta-Learning](#2021-02-01-2)
  - [3. Synthesizing Monolingual Data for Neural Machine Translation](#2021-02-01-3)
  - [4. Transition based Graph Decoder for Neural Machine Translation](#2021-02-01-4)
- [Other Columns](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)





# 2021-02-04

[Return to Index](#Index)



<h2 id="2021-02-04-1">1. The Multilingual TEDx Corpus for Speech Recognition and Translation</h2>

Title: [The Multilingual TEDx Corpus for Speech Recognition and Translation](https://arxiv.org/abs/2102.01757)

Authors: [Elizabeth Salesky](https://arxiv.org/search/cs?searchtype=author&query=Salesky%2C+E), [Matthew Wiesner](https://arxiv.org/search/cs?searchtype=author&query=Wiesner%2C+M), [Jacob Bremerman](https://arxiv.org/search/cs?searchtype=author&query=Bremerman%2C+J), [Roldano Cattoni](https://arxiv.org/search/cs?searchtype=author&query=Cattoni%2C+R), [Matteo Negri](https://arxiv.org/search/cs?searchtype=author&query=Negri%2C+M), [Marco Turchi](https://arxiv.org/search/cs?searchtype=author&query=Turchi%2C+M), [Douglas W. Oard](https://arxiv.org/search/cs?searchtype=author&query=Oard%2C+D+W), [Matt Post](https://arxiv.org/search/cs?searchtype=author&query=Post%2C+M)

> We present the Multilingual TEDx corpus, built to support speech recognition (ASR) and speech translation (ST) research across many non-English source languages. The corpus is a collection of audio recordings from TEDx talks in 8 source languages. We segment transcripts into sentences and align them to the source-language audio and target-language translations. The corpus is released along with open-sourced code enabling extension to new talks and languages as they become available. Our corpus creation methodology can be applied to more languages than previous work, and creates multi-way parallel evaluation sets. We provide baselines in multiple ASR and ST settings, including multilingual models to improve translation performance for low-resource language pairs.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2102.01757](https://arxiv.org/abs/2102.01757) [cs.CL]** |
|           | (or **[arXiv:2102.01757v1](https://arxiv.org/abs/2102.01757v1) [cs.CL]** for this version) |



<h2 id="2021-02-04-2">2. Memorization vs. Generalization: Quantifying Data Leakage in NLP Performance Evaluation</h2>

Title: [Memorization vs. Generalization: Quantifying Data Leakage in NLP Performance Evaluation](https://arxiv.org/abs/2102.01818)

Authors: [Aparna Elangovan](https://arxiv.org/search/cs?searchtype=author&query=Elangovan%2C+A), [Jiayuan He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+J), [Karin Verspoor](https://arxiv.org/search/cs?searchtype=author&query=Verspoor%2C+K)

> Public datasets are often used to evaluate the efficacy and generalizability of state-of-the-art methods for many tasks in natural language processing (NLP). However, the presence of overlap between the train and test datasets can lead to inflated results, inadvertently evaluating the model's ability to memorize and interpreting it as the ability to generalize. In addition, such data sets may not provide an effective indicator of the performance of these methods in real world scenarios. We identify leakage of training data into test data on several publicly available datasets used to evaluate NLP tasks, including named entity recognition and relation extraction, and study them to assess the impact of that leakage on the model's ability to memorize versus generalize.

| Comments: | To appear EACL 2021                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2102.01818](https://arxiv.org/abs/2102.01818) [cs.CL]** |
|           | (or **[arXiv:2102.01818v1](https://arxiv.org/abs/2102.01818v1) [cs.CL]** for this version) |





<h2 id="2021-02-04-3">3. When Can Models Learn From Explanations? A Formal Framework for Understanding the Roles of Explanation Data</h2>

Title: [When Can Models Learn From Explanations? A Formal Framework for Understanding the Roles of Explanation Data](https://arxiv.org/abs/2102.02201)

Authors: [Peter Hase](https://arxiv.org/search/cs?searchtype=author&query=Hase%2C+P), [Mohit Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+M)

> Many methods now exist for conditioning model outputs on task instructions, retrieved documents, and user-provided explanations and feedback. Rather than relying solely on examples of task inputs and outputs, these approaches allow for valuable additional data to be used in modeling with the purpose of improving model correctness and aligning learned models with human priors. Meanwhile, a growing body of evidence suggests that some language models can (1) store a large amount of knowledge in their parameters, and (2) perform inference over tasks in unstructured text to solve new tasks at test time. These results raise the possibility that, for some tasks, humans cannot explain to a model any more about the task than it already knows or could infer on its own. In this paper, we study the circumstances under which explanations of individual data points can (or cannot) improve modeling performance. In order to carefully control important properties of the data and explanations, we introduce a synthetic dataset for experiments, and we also make use of three existing datasets with explanations: e-SNLI, TACRED, SemEval. We first give a formal framework for the available modeling approaches, in which explanation data can be used as model inputs, as labels, or as a prior. After arguing that the most promising role for explanation data is as model inputs, we propose to use a retrieval-based method and show that it solves our synthetic task with accuracies upwards of 95%, while baselines without explanation data achieve below 65% accuracy. We then identify properties of datasets for which retrieval-based modeling fails. With the three existing datasets, we find no improvements from explanation retrieval. Drawing on our findings from our synthetic task, we suggest that at least one of six preconditions for successful modeling fails to hold with these datasets.

| Comments: | 25 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2102.02201](https://arxiv.org/abs/2102.02201) [cs.CL]** |
|           | (or **[arXiv:2102.02201v1](https://arxiv.org/abs/2102.02201v1) [cs.CL]** for this version) |





# 2021-02-03

[Return to Index](#Index)



<h2 id="2021-02-03-1">1. Two Demonstrations of the Machine Translation Applications to Historical Documents</h2>

Title: [Two Demonstrations of the Machine Translation Applications to Historical Documents](https://arxiv.org/abs/2102.01417)

Authors:[Miguel Domingo](https://arxiv.org/search/cs?searchtype=author&query=Domingo%2C+M), [Francisco Casacuberta](https://arxiv.org/search/cs?searchtype=author&query=Casacuberta%2C+F)

> We present our demonstration of two machine translation applications to historical documents. The first task consists in generating a new version of a historical document, written in the modern version of its original language. The second application is limited to a document's orthography. It adapts the document's spelling to modern standards in order to achieve an orthography consistency and accounting for the lack of spelling conventions. We followed an interactive, adaptive framework that allows the user to introduce corrections to the system's hypothesis. The system reacts to these corrections by generating a new hypothesis that takes them into account. Once the user is satisfied with the system's hypothesis and validates it, the system adapts its model following an online learning strategy. This system is implemented following a client-server architecture. We developed a website which communicates with the neural models. All code is open-source and publicly available. The demonstration is hosted at [this http URL](http://demosmt.prhlt.upv.es/mthd/).

| Comments: | Presented at the Demos session of ICPR 2020: [this https URL](https://www.micc.unifi.it/icpr2020/index.php/demos/) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2102.01417](https://arxiv.org/abs/2102.01417) [cs.CL]** |
|           | (or **[arXiv:2102.01417v1](https://arxiv.org/abs/2102.01417v1) [cs.CL]** for this version) |





<h2 id="2021-02-03-2">2. CTC-based Compression for Direct Speech Translation</h2>

Title: [CTC-based Compression for Direct Speech Translation](https://arxiv.org/abs/2102.01578)

Authors:[Marco Gaido](https://arxiv.org/search/cs?searchtype=author&query=Gaido%2C+M), [Mauro Cettolo](https://arxiv.org/search/cs?searchtype=author&query=Cettolo%2C+M), [Matteo Negri](https://arxiv.org/search/cs?searchtype=author&query=Negri%2C+M), [Marco Turchi](https://arxiv.org/search/cs?searchtype=author&query=Turchi%2C+M)

> Previous studies demonstrated that a dynamic phone-informed compression of the input audio is beneficial for speech translation (ST). However, they required a dedicated model for phone recognition and did not test this solution for direct ST, in which a single model translates the input audio into the target language without intermediate representations. In this work, we propose the first method able to perform a dynamic compression of the input indirect ST models. In particular, we exploit the Connectionist Temporal Classification (CTC) to compress the input sequence according to its phonetic characteristics. Our experiments demonstrate that our solution brings a 1.3-1.5 BLEU improvement over a strong baseline on two language pairs (English-Italian and English-German), contextually reducing the memory footprint by more than 10%.

| Comments: | Accepted at EACL2021                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2102.01578](https://arxiv.org/abs/2102.01578) [cs.CL]** |
|           | (or **[arXiv:2102.01578v1](https://arxiv.org/abs/2102.01578v1) [cs.CL]** for this version) |





<h2 id="2021-02-03-3">3. The GEM Benchmark: Natural Language Generation, its Evaluation and Metrics</h2>

Title: [The GEM Benchmark: Natural Language Generation, its Evaluation and Metrics](https://arxiv.org/abs/2102.01672)

Authors:[Sebastian Gehrmann](https://arxiv.org/search/cs?searchtype=author&query=Gehrmann%2C+S), [Tosin Adewumi](https://arxiv.org/search/cs?searchtype=author&query=Adewumi%2C+T), [Karmanya Aggarwal](https://arxiv.org/search/cs?searchtype=author&query=Aggarwal%2C+K), [Pawan Sasanka Ammanamanchi](https://arxiv.org/search/cs?searchtype=author&query=Ammanamanchi%2C+P+S), [Aremu Anuoluwapo](https://arxiv.org/search/cs?searchtype=author&query=Anuoluwapo%2C+A), [Antoine Bosselut](https://arxiv.org/search/cs?searchtype=author&query=Bosselut%2C+A), [Khyathi Raghavi Chandu](https://arxiv.org/search/cs?searchtype=author&query=Chandu%2C+K+R), [Miruna Clinciu](https://arxiv.org/search/cs?searchtype=author&query=Clinciu%2C+M), [Dipanjan Das](https://arxiv.org/search/cs?searchtype=author&query=Das%2C+D), [Kaustubh D. Dhole](https://arxiv.org/search/cs?searchtype=author&query=Dhole%2C+K+D), [Wanyu Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+W), [Esin Durmus](https://arxiv.org/search/cs?searchtype=author&query=Durmus%2C+E), [Ondřej Dušek](https://arxiv.org/search/cs?searchtype=author&query=Dušek%2C+O), [Chris Emezue](https://arxiv.org/search/cs?searchtype=author&query=Emezue%2C+C), [Varun Gangal](https://arxiv.org/search/cs?searchtype=author&query=Gangal%2C+V), [Cristina Garbacea](https://arxiv.org/search/cs?searchtype=author&query=Garbacea%2C+C), [Tatsunori Hashimoto](https://arxiv.org/search/cs?searchtype=author&query=Hashimoto%2C+T), [Yufang Hou](https://arxiv.org/search/cs?searchtype=author&query=Hou%2C+Y), [Yacine Jernite](https://arxiv.org/search/cs?searchtype=author&query=Jernite%2C+Y), [Harsh Jhamtani](https://arxiv.org/search/cs?searchtype=author&query=Jhamtani%2C+H), [Yangfeng Ji](https://arxiv.org/search/cs?searchtype=author&query=Ji%2C+Y), [Shailza Jolly](https://arxiv.org/search/cs?searchtype=author&query=Jolly%2C+S), [Dhruv Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+D), [Faisal Ladhak](https://arxiv.org/search/cs?searchtype=author&query=Ladhak%2C+F), [Aman Madaan](https://arxiv.org/search/cs?searchtype=author&query=Madaan%2C+A), [Mounica Maddela](https://arxiv.org/search/cs?searchtype=author&query=Maddela%2C+M), [Khyati Mahajan](https://arxiv.org/search/cs?searchtype=author&query=Mahajan%2C+K), [Saad Mahamood](https://arxiv.org/search/cs?searchtype=author&query=Mahamood%2C+S), [Bodhisattwa Prasad Majumder](https://arxiv.org/search/cs?searchtype=author&query=Majumder%2C+B+P), [Pedro Henrique Martins](https://arxiv.org/search/cs?searchtype=author&query=Martins%2C+P+H), [Angelina McMillan-Major](https://arxiv.org/search/cs?searchtype=author&query=McMillan-Major%2C+A), [Simon Mille](https://arxiv.org/search/cs?searchtype=author&query=Mille%2C+S), [Emiel van Miltenburg](https://arxiv.org/search/cs?searchtype=author&query=van+Miltenburg%2C+E), [Moin Nadeem](https://arxiv.org/search/cs?searchtype=author&query=Nadeem%2C+M), [Shashi Narayan](https://arxiv.org/search/cs?searchtype=author&query=Narayan%2C+S), [Vitaly Nikolaev](https://arxiv.org/search/cs?searchtype=author&query=Nikolaev%2C+V), [Rubungo Andre Niyongabo](https://arxiv.org/search/cs?searchtype=author&query=Niyongabo%2C+R+A), [Salomey Osei](https://arxiv.org/search/cs?searchtype=author&query=Osei%2C+S), [Ankur Parikh](https://arxiv.org/search/cs?searchtype=author&query=Parikh%2C+A), [Laura Perez-Beltrachini](https://arxiv.org/search/cs?searchtype=author&query=Perez-Beltrachini%2C+L), [Niranjan Ramesh Rao](https://arxiv.org/search/cs?searchtype=author&query=Rao%2C+N+R), [Vikas Raunak](https://arxiv.org/search/cs?searchtype=author&query=Raunak%2C+V), [Juan Diego Rodriguez](https://arxiv.org/search/cs?searchtype=author&query=Rodriguez%2C+J+D), [Sashank Santhanam](https://arxiv.org/search/cs?searchtype=author&query=Santhanam%2C+S), [João Sedoc](https://arxiv.org/search/cs?searchtype=author&query=Sedoc%2C+J), [Thibault Sellam](https://arxiv.org/search/cs?searchtype=author&query=Sellam%2C+T), [Samira Shaikh](https://arxiv.org/search/cs?searchtype=author&query=Shaikh%2C+S), [Anastasia Shimorina](https://arxiv.org/search/cs?searchtype=author&query=Shimorina%2C+A), [Marco Antonio Sobrevilla Cabezudo](https://arxiv.org/search/cs?searchtype=author&query=Cabezudo%2C+M+A+S), [Hendrik Strobelt](https://arxiv.org/search/cs?searchtype=author&query=Strobelt%2C+H), [Nishant Subramani](https://arxiv.org/search/cs?searchtype=author&query=Subramani%2C+N), [Wei Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+W), [Diyi Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+D), [Akhila Yerukola](https://arxiv.org/search/cs?searchtype=author&query=Yerukola%2C+A), [Jiawei Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J)

> We introduce GEM, a living benchmark for natural language Generation (NLG), its Evaluation, and Metrics. Measuring progress in NLG relies on a constantly evolving ecosystem of automated metrics, datasets, and human evaluation standards. However, due to this moving target, new models often still evaluate on divergent anglo-centric corpora with well-established, but flawed, metrics. This disconnect makes it challenging to identify the limitations of current models and opportunities for progress. Addressing this limitation, GEM provides an environment in which models can easily be applied to a wide set of corpora and evaluation strategies can be tested. Regular updates to the benchmark will help NLG research become more multilingual and evolve the challenge alongside models.
> This paper serves as the description of the initial release for which we are organizing a shared task at our ACL 2021 Workshop and to which we invite the entire NLG community to participate.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2102.01672](https://arxiv.org/abs/2102.01672) [cs.CL]** |
|           | (or **[arXiv:2102.01672v2](https://arxiv.org/abs/2102.01672v2) [cs.CL]** for this version) |





# 2021-02-02

[Return to Index](#Index)



<h2 id="2021-02-02-1">1. Speech Recognition by Simply Fine-tuning BERT</h2>

Title: [Speech Recognition by Simply Fine-tuning BERT](https://arxiv.org/abs/2102.00291)

Authors: [Wen-Chin Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+W), [Chia-Hua Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+C), [Shang-Bao Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+S), [Kuan-Yu Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+K), [Hsin-Min Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+H), [Tomoki Toda](https://arxiv.org/search/cs?searchtype=author&query=Toda%2C+T)

> We propose a simple method for automatic speech recognition (ASR) by fine-tuning BERT, which is a language model (LM) trained on large-scale unlabeled text data and can generate rich contextual representations. Our assumption is that given a history context sequence, a powerful LM can narrow the range of possible choices and the speech signal can be used as a simple clue. Hence, comparing to conventional ASR systems that train a powerful acoustic model (AM) from scratch, we believe that speech recognition is possible by simply fine-tuning a BERT model. As an initial study, we demonstrate the effectiveness of the proposed idea on the AISHELL dataset and show that stacking a very simple AM on top of BERT can yield reasonable performance.

| Comments: | Accepted to ICASSP 2021                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Sound (cs.SD)**; Computation and Language (cs.CL); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2102.00291](https://arxiv.org/abs/2102.00291) [cs.SD]** |
|           | (or **[arXiv:2102.00291v1](https://arxiv.org/abs/2102.00291v1) [cs.SD]** for this version) |





<h2 id="2021-02-02-2">2. Phoneme-BERT: Joint Language Modelling of Phoneme Sequence and ASR Transcript</h2>

Title: [Phoneme-BERT: Joint Language Modelling of Phoneme Sequence and ASR Transcript](https://arxiv.org/abs/2102.00804)

Authors: [Mukuntha Narayanan Sundararaman](https://arxiv.org/search/eess?searchtype=author&query=Sundararaman%2C+M+N), [Ayush Kumar](https://arxiv.org/search/eess?searchtype=author&query=Kumar%2C+A), [Jithendra Vepa](https://arxiv.org/search/eess?searchtype=author&query=Vepa%2C+J)

> Recent years have witnessed significant improvement in ASR systems to recognize spoken utterances. However, it is still a challenging task for noisy and out-of-domain data, where substitution and deletion errors are prevalent in the transcribed text. These errors significantly degrade the performance of downstream tasks. In this work, we propose a BERT-style language model, referred to as PhonemeBERT, that learns a joint language model with phoneme sequence and ASR transcript to learn phonetic-aware representations that are robust to ASR errors. We show that PhonemeBERT can be used on downstream tasks using phoneme sequences as additional features, and also in low-resource setup where we only have ASR-transcripts for the downstream tasks with no phoneme information available. We evaluate our approach extensively by generating noisy data for three benchmark datasets - Stanford Sentiment Treebank, TREC and ATIS for sentiment, question and intent classification tasks respectively. The results of the proposed approach beats the state-of-the-art baselines comprehensively on each dataset.

| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2102.00804](https://arxiv.org/abs/2102.00804) [eess.AS]** |
|           | (or **[arXiv:2102.00804v1](https://arxiv.org/abs/2102.00804v1) [eess.AS]** for this version) |





<h2 id="2021-02-02-3">3. Machine Translationese: Effects of Algorithmic Bias on Linguistic Complexity in Machine Translation</h2>

Title: [Machine Translationese: Effects of Algorithmic Bias on Linguistic Complexity in Machine Translation](https://arxiv.org/abs/2102.00287)

Authors: [Eva Vanmassenhove](https://arxiv.org/search/cs?searchtype=author&query=Vanmassenhove%2C+E), [Dimitar Shterionov](https://arxiv.org/search/cs?searchtype=author&query=Shterionov%2C+D), [Matthew Gwilliam](https://arxiv.org/search/cs?searchtype=author&query=Gwilliam%2C+M)

> Recent studies in the field of Machine Translation (MT) and Natural Language Processing (NLP) have shown that existing models amplify biases observed in the training data. The amplification of biases in language technology has mainly been examined with respect to specific phenomena, such as gender bias. In this work, we go beyond the study of gender in MT and investigate how bias amplification might affect language in a broader sense. We hypothesize that the 'algorithmic bias', i.e. an exacerbation of frequently observed patterns in combination with a loss of less frequent ones, not only exacerbates societal biases present in current datasets but could also lead to an artificially impoverished language: 'machine translationese'. We assess the linguistic richness (on a lexical and morphological level) of translations created by different data-driven MT paradigms - phrase-based statistical (PB-SMT) and neural MT (NMT). Our experiments show that there is a loss of lexical and morphological richness in the translations produced by all investigated MT paradigms for two language pairs (EN<=>FR and EN<=>ES).

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computers and Society (cs.CY) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2102.00287](https://arxiv.org/abs/2102.00287) [cs.CL]** |
|           | (or **[arXiv:2102.00287v1](https://arxiv.org/abs/2102.00287v1) [cs.CL]** for this version) |





<h2 id="2021-02-02-4">4. Decoupling the Role of Data, Attention, and Losses in Multimodal Transformers</h2>

Title: [Decoupling the Role of Data, Attention, and Losses in Multimodal Transformers](https://arxiv.org/abs/2102.00529)

Authors: [Lisa Anne Hendricks](https://arxiv.org/search/cs?searchtype=author&query=Hendricks%2C+L+A), [John Mellor](https://arxiv.org/search/cs?searchtype=author&query=Mellor%2C+J), [Rosalia Schneider](https://arxiv.org/search/cs?searchtype=author&query=Schneider%2C+R), [Jean-Baptiste Alayrac](https://arxiv.org/search/cs?searchtype=author&query=Alayrac%2C+J), [Aida Nematzadeh](https://arxiv.org/search/cs?searchtype=author&query=Nematzadeh%2C+A)

> Recently multimodal transformer models have gained popularity because their performance on language and vision tasks suggest they learn rich visual-linguistic representations. Focusing on zero-shot image retrieval tasks, we study three important factors which can impact the quality of learned representations: pretraining data, the attention mechanism, and loss functions. By pretraining models on six datasets, we observe that dataset noise and language similarity to our downstream task are important indicators of model performance. Through architectural analysis, we learn that models with a multimodal attention mechanism can outperform deeper models with modality specific attention mechanisms. Finally, we show that successful contrastive losses used in the self-supervised learning literature do not yield similar performance gains when used in multimodal transformers

| Comments: | pre-print of MIT Press Publication version                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2102.00529](https://arxiv.org/abs/2102.00529) [cs.CL]** |
|           | (or **[arXiv:2102.00529v1](https://arxiv.org/abs/2102.00529v1) [cs.CL]** for this version) |







<h2 id="2021-02-02-5">5. Neural OCR Post-Hoc Correction of Historical Corpora</h2>

Title: [Neural OCR Post-Hoc Correction of Historical Corpora](https://arxiv.org/abs/2102.00583)

Authors: [Lijun Lyu](https://arxiv.org/search/cs?searchtype=author&query=Lyu%2C+L), [Maria Koutraki](https://arxiv.org/search/cs?searchtype=author&query=Koutraki%2C+M), [Martin Krickl](https://arxiv.org/search/cs?searchtype=author&query=Krickl%2C+M), [Besnik Fetahu](https://arxiv.org/search/cs?searchtype=author&query=Fetahu%2C+B)

> Optical character recognition (OCR) is crucial for a deeper access to historical collections. OCR needs to account for orthographic variations, typefaces, or language evolution (i.e., new letters, word spellings), as the main source of character, word, or word segmentation transcription errors. For digital corpora of historical prints, the errors are further exacerbated due to low scan quality and lack of language standardization.
> For the task of OCR post-hoc correction, we propose a neural approach based on a combination of recurrent (RNN) and deep convolutional network (ConvNet) to correct OCR transcription errors. At character level we flexibly capture errors, and decode the corrected output based on a novel attention mechanism. Accounting for the input and output similarity, we propose a new loss function that rewards the model's correcting behavior.
> Evaluation on a historical book corpus in German language shows that our models are robust in capturing diverse OCR transcription errors and reduce the word error rate of 32.3% by more than 89%.

| Comments: | To appear at TACL                                            |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2102.00583](https://arxiv.org/abs/2102.00583) [cs.CL]** |
|           | (or **[arXiv:2102.00583v1](https://arxiv.org/abs/2102.00583v1) [cs.CL]** for this version) |





<h2 id="2021-02-02-6">6. GTAE: Graph-Transformer based Auto-Encoders for Linguistic-Constrained Text Style Transfer</h2>

Title: [GTAE: Graph-Transformer based Auto-Encoders for Linguistic-Constrained Text Style Transfer](https://arxiv.org/abs/2102.00769)

Authors: [Yukai Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+Y), [Sen Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+S), [Chenxing Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+C), [Xiaodan Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+X), [Xiaojun Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+X), [Liang Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+L)

> Non-parallel text style transfer has attracted increasing research interests in recent years. Despite successes in transferring the style based on the encoder-decoder framework, current approaches still lack the ability to preserve the content and even logic of original sentences, mainly due to the large unconstrained model space or too simplified assumptions on latent embedding space. Since language itself is an intelligent product of humans with certain grammars and has a limited rule-based model space by its nature, relieving this problem requires reconciling the model capacity of deep neural networks with the intrinsic model constraints from human linguistic rules. To this end, we propose a method called Graph Transformer based Auto Encoder (GTAE), which models a sentence as a linguistic graph and performs feature extraction and style transfer at the graph level, to maximally retain the content and the linguistic structure of original sentences. Quantitative experiment results on three non-parallel text style transfer tasks show that our model outperforms state-of-the-art methods in content preservation, while achieving comparable performance on transfer accuracy and sentence naturalness.

| Comments: | The first two authors share equal-authorship; Code:[this https URL](https://github.com/SenZHANG-GitHub/graph-text-style-transfer) ; benchmark: [this https URL](https://github.com/ykshi/text-style-transfer-benchmark) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2102.00769](https://arxiv.org/abs/2102.00769) [cs.CL]** |
|           | (or **[arXiv:2102.00769v1](https://arxiv.org/abs/2102.00769v1) [cs.CL]** for this version) |





<h2 id="2021-02-02-7">7. Multilingual LAMA: Investigating Knowledge in Multilingual Pretrained Language Models</h2>

Title: [Multilingual LAMA: Investigating Knowledge in Multilingual Pretrained Language Models](https://arxiv.org/abs/2102.00894)

Authors: [Nora Kassner](https://arxiv.org/search/cs?searchtype=author&query=Kassner%2C+N), [Philipp Dufter](https://arxiv.org/search/cs?searchtype=author&query=Dufter%2C+P), [Hinrich Schütze](https://arxiv.org/search/cs?searchtype=author&query=Schütze%2C+H)

> Recently, it has been found that monolingual English language models can be used as knowledge bases. Instead of structural knowledge base queries, masked sentences such as "Paris is the capital of [MASK]" are used as probes. We translate the established benchmarks TREx and GoogleRE into 53 languages. Working with mBERT, we investigate three questions. (i) Can mBERT be used as a multilingual knowledge base? Most prior work only considers English. Extending research to multiple languages is important for diversity and accessibility. (ii) Is mBERT's performance as knowledge base language-independent or does it vary from language to language? (iii) A multilingual model is trained on more text, e.g., mBERT is trained on 104 Wikipedias. Can mBERT leverage this for better performance? We find that using mBERT as a knowledge base yields varying performance across languages and pooling predictions across languages improves performance. Conversely, mBERT exhibits a language bias; e.g., when queried in Italian, it tends to predict Italy as the country of origin.

| Comments: | Accepted to EACL 2021                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2102.00894](https://arxiv.org/abs/2102.00894) [cs.CL]** |
|           | (or **[arXiv:2102.00894v1](https://arxiv.org/abs/2102.00894v1) [cs.CL]** for this version) |





<h2 id="2021-02-02-8">8. End2End Acoustic to Semantic Transduction</h2>

Title: [End2End Acoustic to Semantic Transduction](https://arxiv.org/abs/2102.01013)

Authors: [Valentin Pelloin](https://arxiv.org/search/cs?searchtype=author&query=Pelloin%2C+V), [Nathalie Camelin](https://arxiv.org/search/cs?searchtype=author&query=Camelin%2C+N), [Antoine Laurent](https://arxiv.org/search/cs?searchtype=author&query=Laurent%2C+A), [Renato De Mori](https://arxiv.org/search/cs?searchtype=author&query=De+Mori%2C+R), [Antoine Caubrière](https://arxiv.org/search/cs?searchtype=author&query=Caubrière%2C+A), [Yannick Estève](https://arxiv.org/search/cs?searchtype=author&query=Estève%2C+Y), [Sylvain Meignier](https://arxiv.org/search/cs?searchtype=author&query=Meignier%2C+S)

> In this paper, we propose a novel end-to-end sequence-to-sequence spoken language understanding model using an attention mechanism. It reliably selects contextual acoustic features in order to hypothesize semantic contents. An initial architecture capable of extracting all pronounced words and concepts from acoustic spans is designed and tested. With a shallow fusion language model, this system reaches a 13.6 concept error rate (CER) and an 18.5 concept value error rate (CVER) on the French MEDIA corpus, achieving an absolute 2.8 points reduction compared to the state-of-the-art. Then, an original model is proposed for hypothesizing concepts and their values. This transduction reaches a 15.4 CER and a 21.6 CVER without any new type of context.

| Comments: | Accepted at IEEE ICASSP 2021                                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2102.01013](https://arxiv.org/abs/2102.01013) [cs.CL]** |
|           | (or **[arXiv:2102.01013v1](https://arxiv.org/abs/2102.01013v1) [cs.CL]** for this version) |





<h2 id="2021-02-02-9">9. Measuring and Improving Consistency in Pretrained Language Models</h2>

Title: [Measuring and Improving Consistency in Pretrained Language Models](https://arxiv.org/abs/2102.01017)

Authors: [Yanai Elazar](https://arxiv.org/search/cs?searchtype=author&query=Elazar%2C+Y), [Nora Kassner](https://arxiv.org/search/cs?searchtype=author&query=Kassner%2C+N), [Shauli Ravfogel](https://arxiv.org/search/cs?searchtype=author&query=Ravfogel%2C+S), [Abhilasha Ravichander](https://arxiv.org/search/cs?searchtype=author&query=Ravichander%2C+A), [Eduard Hovy](https://arxiv.org/search/cs?searchtype=author&query=Hovy%2C+E), [Hinrich Schütze](https://arxiv.org/search/cs?searchtype=author&query=Schütze%2C+H), [Yoav Goldberg](https://arxiv.org/search/cs?searchtype=author&query=Goldberg%2C+Y)

> Consistency of a model -- that is, the invariance of its behavior under meaning-preserving alternations in its input -- is a highly desirable property in natural language processing. In this paper we study the question: Are Pretrained Language Models (PLMs) consistent with respect to factual knowledge? To this end, we create ParaRel, a high-quality resource of cloze-style query English paraphrases. It contains a total of 328 paraphrases for thirty-eight relations. Using ParaRel, we show that the consistency of all PLMs we experiment with is poor -- though with high variance between relations. Our analysis of the representational spaces of PLMs suggests that they have a poor structure and are currently not suitable for representing knowledge in a robust way. Finally, we propose a method for improving model consistency and experimentally demonstrate its effectiveness.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2102.01017](https://arxiv.org/abs/2102.01017) [cs.CL]** |
|           | (or **[arXiv:2102.01017v1](https://arxiv.org/abs/2102.01017v1) [cs.CL]** for this version) |











# 2021-02-01

[Return to Index](#Index)



<h2 id="2021-02-01-1">1. Combining pre-trained language models and structured knowledge</h2>

Title: [Combining pre-trained language models and structured knowledge](https://arxiv.org/abs/2101.12294)

Authors: [Pedro Colon-Hernandez](https://arxiv.org/search/cs?searchtype=author&query=Colon-Hernandez%2C+P), [Catherine Havasi](https://arxiv.org/search/cs?searchtype=author&query=Havasi%2C+C), [Jason Alonso](https://arxiv.org/search/cs?searchtype=author&query=Alonso%2C+J), [Matthew Huggins](https://arxiv.org/search/cs?searchtype=author&query=Huggins%2C+M), [Cynthia Breazeal](https://arxiv.org/search/cs?searchtype=author&query=Breazeal%2C+C)

> In recent years, transformer-based language models have achieved state of the art performance in various NLP benchmarks. These models are able to extract mostly distributional information with some semantics from unstructured text, however it has proven challenging to integrate structured information, such as knowledge graphs into these models. We examine a variety of approaches to integrate structured knowledge into current language models and determine challenges, and possible opportunities to leverage both structured and unstructured information sources. From our survey, we find that there are still opportunities at exploiting adapter-based injections and that it may be possible to further combine various of the explored approaches into one system.

| Comments: | Initial Submission                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2101.12294](https://arxiv.org/abs/2101.12294) [cs.CL]** |
|           | (or **[arXiv:2101.12294v1](https://arxiv.org/abs/2101.12294v1) [cs.CL]** for this version) |





<h2 id="2021-02-01-2">2. Few-Shot Domain Adaptation for Grammatical Error Correction via Meta-Learning</h2>

Title: [Few-Shot Domain Adaptation for Grammatical Error Correction via Meta-Learning](https://arxiv.org/abs/2101.12409)

Authors: [Shengsheng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+S), [Yaping Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+Y), [Yun Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Liner Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+L), [Chencheng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Erhong Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+E)

> Most existing Grammatical Error Correction (GEC) methods based on sequence-to-sequence mainly focus on how to generate more pseudo data to obtain better performance. Few work addresses few-shot GEC domain adaptation. In this paper, we treat different GEC domains as different GEC tasks and propose to extend meta-learning to few-shot GEC domain adaptation without using any pseudo data. We exploit a set of data-rich source domains to learn the initialization of model parameters that facilitates fast adaptation on new resource-poor target domains. We adapt GEC model to the first language (L1) of the second language learner. To evaluate the proposed method, we use nine L1s as source domains and five L1s as target domains. Experiment results on the L1 GEC domain adaptation dataset demonstrate that the proposed approach outperforms the multi-task transfer learning baseline by 0.50 F0.5 score on average and enables us to effectively adapt to a new L1 domain with only 200 parallel sentences.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2101.12409](https://arxiv.org/abs/2101.12409) [cs.CL]** |
|           | (or **[arXiv:2101.12409v1](https://arxiv.org/abs/2101.12409v1) [cs.CL]** for this version) |





<h2 id="2021-02-01-3">3. Synthesizing Monolingual Data for Neural Machine Translation</h2>

Title: [Synthesizing Monolingual Data for Neural Machine Translation](https://arxiv.org/abs/2101.12462)

Authors: [Benjamin Marie](https://arxiv.org/search/cs?searchtype=author&query=Marie%2C+B), [Atsushi Fujita](https://arxiv.org/search/cs?searchtype=author&query=Fujita%2C+A)

> In neural machine translation (NMT), monolingual data in the target language are usually exploited through a method so-called "back-translation" to synthesize additional training parallel data. The synthetic data have been shown helpful to train better NMT, especially for low-resource language pairs and domains. Nonetheless, large monolingual data in the target domains or languages are not always available to generate large synthetic parallel data. In this work, we propose a new method to generate large synthetic parallel data leveraging very small monolingual data in a specific domain. We fine-tune a pre-trained GPT-2 model on such small in-domain monolingual data and use the resulting model to generate a large amount of synthetic in-domain monolingual data. Then, we perform back-translation, or forward translation, to generate synthetic in-domain parallel data. Our preliminary experiments on three language pairs and five domains show the effectiveness of our method to generate fully synthetic but useful in-domain parallel data for improving NMT in all configurations. We also show promising results in extreme adaptation for personalized NMT.

| Comments: | Preliminary work                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2101.12462](https://arxiv.org/abs/2101.12462) [cs.CL]** |
|           | (or **[arXiv:2101.12462v1](https://arxiv.org/abs/2101.12462v1) [cs.CL]** for this version) |





<h2 id="2021-02-01-4">4. Transition based Graph Decoder for Neural Machine Translation</h2>

Title: [Transition based Graph Decoder for Neural Machine Translation](https://arxiv.org/abs/2101.12640)

Authors: [Leshem Choshen](https://arxiv.org/search/cs?searchtype=author&query=Choshen%2C+L), [Omri Abend](https://arxiv.org/search/cs?searchtype=author&query=Abend%2C+O)

> While a number of works showed gains from incorporating source-side symbolic syntactic and semantic structure into neural machine translation (NMT), much fewer works addressed the decoding of such structure.
> We propose a general Transformer-based approach for tree and graph decoding based on generating a sequence of transitions, inspired by a similar approach that uses RNNs by Dyer (2016).
> Experiments with using the proposed decoder with Universal Dependencies syntax on English-German, German-English and English-Russian show improved performance over the standard Transformer decoder, as well as over ablated versions of the model.\tacltxt{\footnote{All code implementing the presented models will be released upon acceptance.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2101.12640](https://arxiv.org/abs/2101.12640) [cs.CL]** |
|           | (or **[arXiv:2101.12640v1](https://arxiv.org/abs/2101.12640v1) [cs.CL]** for this version) |