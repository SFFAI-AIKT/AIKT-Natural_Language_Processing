# Daily arXiv: Machine Translation - April, 2021

# Index

- [2021-04-01](#2021-04-01)
  - [1. An Exploration of Data Augmentation Techniques for Improving English to Tigrinya Translation](#2021-04-01-1)
- [2. Few-shot learning through contextual data augmentation](#2021-04-01-2)
  - [3. UA-GEC: Grammatical Error Correction and Fluency Corpus for the Ukrainian Language](#2021-04-01-3)
  - [4. Divide and Rule: Training Context-Aware Multi-Encoder Translation Models with Little Resources](#2021-04-01-4)
  - [5. Leveraging Neural Machine Translation for Word Alignment](#2021-04-01-5)
- [2021-03-31](#2021-03-31)	
  - [1. Diagnosing Vision-and-Language Navigation: What Really Matters](#2021-03-31-1)
- [Other Columns](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)





# 2021-04-01

[Return to Index](#Index)



<h2 id="2021-04-01-1">1. An Exploration of Data Augmentation Techniques for Improving English to Tigrinya Translation
</h2>

Title: [An Exploration of Data Augmentation Techniques for Improving English to Tigrinya Translation](https://arxiv.org/abs/2103.16789)

Authors:[Lidia Kidane](https://arxiv.org/search/cs?searchtype=author&query=Kidane%2C+L), [Sachin Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+S), [Yulia Tsvetkov](https://arxiv.org/search/cs?searchtype=author&query=Tsvetkov%2C+Y)

> It has been shown that the performance of neural machine translation (NMT) drops starkly in low-resource conditions, often requiring large amounts of auxiliary data to achieve competitive results. An effective method of generating auxiliary data is back-translation of target language sentences. In this work, we present a case study of Tigrinya where we investigate several back-translation methods to generate synthetic source sentences. We find that in low-resource conditions, back-translation by pivoting through a higher-resource language related to the target language proves most effective resulting in substantial improvements over baselines.

| Comments: | Accepted at AfricaNLP Workshop, EACL 2021                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2103.16789](https://arxiv.org/abs/2103.16789) [cs.CL]** |
|           | (or **[arXiv:2103.16789v1](https://arxiv.org/abs/2103.16789v1) [cs.CL]** for this version) |





<h2 id="2021-04-01-2">2. Few-shot learning through contextual data augmentation
</h2>

Title: [Few-shot learning through contextual data augmentation](https://arxiv.org/abs/2103.16911)

Authors:[Farid Arthaud](https://arxiv.org/search/cs?searchtype=author&query=Arthaud%2C+F), [Rachel Bawden](https://arxiv.org/search/cs?searchtype=author&query=Bawden%2C+R), [Alexandra Birch](https://arxiv.org/search/cs?searchtype=author&query=Birch%2C+A)

> Machine translation (MT) models used in industries with constantly changing topics, such as translation or news agencies, need to adapt to new data to maintain their performance over time. Our aim is to teach a pre-trained MT model to translate previously unseen words accurately, based on very few examples. We propose (i) an experimental setup allowing us to simulate novel vocabulary appearing in human-submitted translations, and (ii) corresponding evaluation metrics to compare our approaches. We extend a data augmentation approach using a pre-trained language model to create training examples with similar contexts for novel words. We compare different fine-tuning and data augmentation approaches and show that adaptation on the scale of one to five examples is possible. Combining data augmentation with randomly selected training sentences leads to the highest BLEU score and accuracy improvements. Impressively, with only 1 to 5 examples, our model reports better accuracy scores than a reference system trained with on average 313 parallel examples.

| Comments: | 14 pages includince 3 of appendices                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2103.16911](https://arxiv.org/abs/2103.16911) [cs.CL]** |
|           | (or **[arXiv:2103.16911v1](https://arxiv.org/abs/2103.16911v1) [cs.CL]** for this version) |





<h2 id="2021-04-01-3">3. UA-GEC: Grammatical Error Correction and Fluency Corpus for the Ukrainian Language
</h2>

Title: [UA-GEC: Grammatical Error Correction and Fluency Corpus for the Ukrainian Language](https://arxiv.org/abs/2103.16997)

Authors:[Oleksiy Syvokon](https://arxiv.org/search/cs?searchtype=author&query=Syvokon%2C+O), [Olena Nahorna](https://arxiv.org/search/cs?searchtype=author&query=Nahorna%2C+O)

> We present a corpus professionally annotated for grammatical error correction (GEC) and fluency edits in the Ukrainian language. To the best of our knowledge, this is the first GEC corpus for the Ukrainian language. We collected texts with errors (20,715 sentences) from a diverse pool of contributors, including both native and non-native speakers. The data cover a wide variety of writing domains, from text chats and essays to formal writing. Professional proofreaders corrected and annotated the corpus for errors relating to fluency, grammar, punctuation, and spelling. This corpus can be used for developing and evaluating GEC systems in Ukrainian. More generally, it can be used for researching multilingual and low-resource NLP, morphologically rich languages, document-level GEC, and fluency correction. The corpus is publicly available at [this https URL](https://github.com/grammarly/ua-gec)

| Comments: | See [this https URL](https://github.com/grammarly/ua-gec) for the dataset. Version 2 of the data is in progress |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2103.16997](https://arxiv.org/abs/2103.16997) [cs.CL]** |
|           | (or **[arXiv:2103.16997v1](https://arxiv.org/abs/2103.16997v1) [cs.CL]** for this version) |





<h2 id="2021-04-01-4">4. Divide and Rule: Training Context-Aware Multi-Encoder Translation Models with Little Resources
</h2>

Title: [Divide and Rule: Training Context-Aware Multi-Encoder Translation Models with Little Resources](https://arxiv.org/abs/2103.17151)

Authors:[Lorenzo Lupo](https://arxiv.org/search/cs?searchtype=author&query=Lupo%2C+L), [Marco Dinarelli](https://arxiv.org/search/cs?searchtype=author&query=Dinarelli%2C+M), [Laurent Besacier](https://arxiv.org/search/cs?searchtype=author&query=Besacier%2C+L)

> Multi-encoder models are a broad family of context-aware Neural Machine Translation (NMT) systems that aim to improve translation quality by encoding document-level contextual information alongside the current sentence. The context encoding is undertaken by contextual parameters, trained on document-level data. In this work, we show that training these parameters takes large amount of data, since the contextual training signal is sparse. We propose an efficient alternative, based on splitting sentence pairs, that allows to enrich the training signal of a set of parallel sentences by breaking intra-sentential syntactic links, and thus frequently pushing the model to search the context for disambiguating clues. We evaluate our approach with BLEU and contrastive test sets, showing that it allows multi-encoder models to achieve comparable performances to a setting where they are trained with ×10 document-level data. We also show that our approach is a viable option to context-aware NMT for language pairs with zero document-level parallel data.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2103.17151](https://arxiv.org/abs/2103.17151) [cs.CL]** |
|           | (or **[arXiv:2103.17151v1](https://arxiv.org/abs/2103.17151v1) [cs.CL]** for this version) |





<h2 id="2021-04-01-5">5. Leveraging Neural Machine Translation for Word Alignment
</h2>

Title: [Leveraging Neural Machine Translation for Word Alignment](https://arxiv.org/abs/2103.17250)

Authors:[Vilém Zouhar](https://arxiv.org/search/cs?searchtype=author&query=Zouhar%2C+V), [Daria Pylypenko](https://arxiv.org/search/cs?searchtype=author&query=Pylypenko%2C+D)

> The most common tools for word-alignment rely on a large amount of parallel sentences, which are then usually processed according to one of the IBM model algorithms. The training data is, however, the same as for machine translation (MT) systems, especially for neural MT (NMT), which itself is able to produce word-alignments using the trained attention heads. This is convenient because word-alignment is theoretically a viable byproduct of any attention-based NMT, which is also able to provide decoder scores for a translated sentence pair.
> We summarize different approaches on how word-alignment can be extracted from alignment scores and then explore ways in which scores can be extracted from NMT, focusing on inferring the word-alignment scores based on output sentence and token probabilities. We compare this to the extraction of alignment scores from attention. We conclude with aggregating all of the sources of alignment scores into a simple feed-forward network which achieves the best results when combined alignment extractors are used.

| Comments: | 16 pages (without references). To be published in PBML 116   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2103.17250](https://arxiv.org/abs/2103.17250) [cs.CL]** |
|           | (or **[arXiv:2103.17250v1](https://arxiv.org/abs/2103.17250v1) [cs.CL]** for this version) |






