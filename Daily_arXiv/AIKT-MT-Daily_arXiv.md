# Daily arXiv: Machine Translation - Apr., 2019

### Index

- [2019-05-07](#2019-05-07)
  - [1. BVS Corpus: A Multilingual Parallel Corpus of Biomedical Scientific Texts](#2019-05-07-1)
  - [2. A Parallel Corpus of Theses and Dissertations Abstracts](#2019-05-07-2)
  - [3. A Large Parallel Corpus of Full-Text Scientific Articles](#2019-05-07-3)
  - [4. UFRGS Participation on the WMT Biomedical Translation Shared Task](#2019-05-07-4)

* [2019-04](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-04.md)
* [2019-03](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-03.md)
* [2019-02](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-02.md)



# 2019-05-07

[Return to Index](#Index)

<h2 id="2019-05-07-1">1. BVS Corpus: A Multilingual Parallel Corpus of Biomedical Scientific Texts</h2>

Title: [BVS Corpus: A Multilingual Parallel Corpus of Biomedical Scientific Texts](https://arxiv.org/abs/1905.01712)

Authors: [Felipe Soares](https://arxiv.org/search/cs?searchtype=author&query=Soares%2C+F), [Martin Krallinger](https://arxiv.org/search/cs?searchtype=author&query=Krallinger%2C+M)

*(Submitted on 5 May 2019)*

> The BVS database (Health Virtual Library) is a centralized source of biomedical information for Latin America and Carib, created in 1998 and coordinated by BIREME (Biblioteca Regional de Medicina) in agreement with the Pan American Health Organization (OPAS). Abstracts are available in English, Spanish, and Portuguese, with a subset in more than one language, thus being a possible source of parallel corpora. In this article, we present the development of parallel corpora from BVS in three languages: English, Portuguese, and Spanish. Sentences were automatically aligned using the Hunalign algorithm for EN/ES and EN/PT language pairs, and for a subset of trilingual articles also. We demonstrate the capabilities of our corpus by training a Neural Machine Translation (OpenNMT) system for each language pair, which outperformed related works on scientific biomedical articles. Sentence alignment was also manually evaluated, presenting an average 96% of correctly aligned sentences across all languages. Our parallel corpus is freely available, with complementary information regarding article metadata.

| Comments: | Accepted at the Copora conference. arXiv admin note: text overlap with [arXiv:1905.01715](https://arxiv.org/abs/1905.01715) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Information Retrieval (cs.IR) |
| Cite as:  | **arXiv:1905.01712 [cs.CL]**                                 |
|           | (or **arXiv:1905.01712v1 [cs.CL]** for this version)         |



<h2 id="2019-05-07-2">2. A Parallel Corpus of Theses and Dissertations Abstracts
</h2>

Title: [A Parallel Corpus of Theses and Dissertations Abstracts](https://arxiv.org/abs/1905.01715)

Authors: [Felipe Soares](https://arxiv.org/search/cs?searchtype=author&query=Soares%2C+F), [Gabrielli Harumi Yamashita](https://arxiv.org/search/cs?searchtype=author&query=Yamashita%2C+G+H), [Michel Jose Anzanello](https://arxiv.org/search/cs?searchtype=author&query=Anzanello%2C+M+J)

*(Submitted on 5 May 2019)*

> In Brazil, the governmental body responsible for overseeing and coordinating post-graduate programs, CAPES, keeps records of all theses and dissertations presented in the country. Information regarding such documents can be accessed online in the Theses and Dissertations Catalog (TDC), which contains abstracts in Portuguese and English, and additional metadata. Thus, this database can be a potential source of parallel corpora for the Portuguese and English languages. In this article, we present the development of a parallel corpus from TDC, which is made available by CAPES under the open data initiative. Approximately 240,000 documents were collected and aligned using the Hunalign tool. We demonstrate the capability of our developed corpus by training Statistical Machine Translation (SMT) and Neural Machine Translation (NMT) models for both language directions, followed by a comparison with Google Translate (GT). Both translation models presented better BLEU scores than GT, with NMT system being the most accurate one. Sentence alignment was also manually evaluated, presenting an average of 82.30% correctly aligned sentences. Our parallel corpus is freely available in TMX format, with complementary information regarding document metadata

| Comments:          | Published in the PROPOR Conference. arXiv admin note: text overlap with [arXiv:1905.01712](https://arxiv.org/abs/1905.01712) |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**                         |
| Journal reference: | Computational Processing of the Portuguese Language 2018     |
| DOI:               | [10.1007/978-3-319-99722-3_35](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1007%2F978-3-319-99722-3_35&v=71962c3d) |
| Cite as:           | **arXiv:1905.01715 [cs.CL]**                                 |
|                    | (or **arXiv:1905.01715v1 [cs.CL]** for this version)         |



<h2 id="2019-05-07-3">3. A Large Parallel Corpus of Full-Text Scientific Articles</h2>

Title: [A Large Parallel Corpus of Full-Text Scientific Articles](https://arxiv.org/abs/1905.01852)

Authors: [Felipe Soares](https://arxiv.org/search/cs?searchtype=author&query=Soares%2C+F), [Viviane Pereira Moreira](https://arxiv.org/search/cs?searchtype=author&query=Moreira%2C+V+P), [Karin Becker](https://arxiv.org/search/cs?searchtype=author&query=Becker%2C+K)

*(Submitted on 6 May 2019)*

> The Scielo database is an important source of scientific information in Latin America, containing articles from several research domains. A striking characteristic of Scielo is that many of its full-text contents are presented in more than one language, thus being a potential source of parallel corpora. In this article, we present the development of a parallel corpus from Scielo in three languages: English, Portuguese, and Spanish. Sentences were automatically aligned using the Hunalign algorithm for all language pairs, and for a subset of trilingual articles also. We demonstrate the capabilities of our corpus by training a Statistical Machine Translation system (Moses) for each language pair, which outperformed related works on scientific articles. Sentence alignment was also manually evaluated, presenting an average of 98.8% correctly aligned sentences across all languages. Our parallel corpus is freely available in the TMX format, with complementary information regarding article metadata.

| Comments: | Published in Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1905.01852 [cs.CL]**                                 |
|           | (or **arXiv:1905.01852v1 [cs.CL]** for this version)         |



<h2 id="2019-05-07-4">4. BVS Corpus: A Multilingual Parallel Corpus of Biomedical Scientific Texts</h2>

Title: [UFRGS Participation on the WMT Biomedical Translation Shared Task](https://arxiv.org/abs/1905.01855)

Authors: [Felipe Soares](https://arxiv.org/search/cs?searchtype=author&query=Soares%2C+F), [Karin Becker](https://arxiv.org/search/cs?searchtype=author&query=Becker%2C+K)

*(Submitted on 6 May 2019)*

> This paper describes the machine translation systems developed by the Universidade Federal do Rio Grande do Sul (UFRGS) team for the biomedical translation shared task. Our systems are based on statistical machine translation and neural machine translation, using the Moses and OpenNMT toolkits, respectively. We participated in four translation directions for the English/Spanish and English/Portuguese language pairs. To create our training data, we concatenated several parallel corpora, both from in-domain and out-of-domain sources, as well as terminological resources from UMLS. Our systems achieved the best BLEU scores according to the official shared task evaluation.

| Comments: | Published on the Third Conference on Machine Translation (WMT18) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **arXiv:1905.01855 [cs.CL]**                                 |
|           | (or **arXiv:1905.01855v1 [cs.CL]** for this version)         |

