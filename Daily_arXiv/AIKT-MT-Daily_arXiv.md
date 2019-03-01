# Daily arXiv: Machine Translation - Mar., 2019

### Index

- [2019-03-01](#2019-03-01)
  - [1. Efficient Contextual Representation Learning Without Softmax Layer](#2019-03-01-1)

* [2019-02-28](https://github.com/SFFAI-AIKT/AIKT-Natural_Language_Processing/blob/master/Daily_arXiv/AIKT-MT-Daily_arXiv-2019-02.md)



# 2019-03-01

[Return to Index](#Index)

<h2 id="2019-03-01-1">1. Efficient Contextual Representation Learning Without Softmax Layer</h2> 

Title: [Efficient Contextual Representation Learning Without Softmax Layer](https://arxiv.org/pdf/1902.11269)

Authors: [Liunian Harold Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L+H), [Patrick H. Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+P+H), [Cho-Jui Hsieh](https://arxiv.org/search/cs?searchtype=author&query=Hsieh%2C+C), [Kai-Wei Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang%2C+K)

Note: Contextual Representaion

*(Submitted on 28 Feb 2019)*

> Contextual representation models have achieved great success in improving various downstream tasks. However, these language-model-based encoders are difficult to train due to the large parameter sizes and high computational complexity. By carefully examining the training procedure, we find that the softmax layer (the output layer) causes significant inefficiency due to the large vocabulary size. Therefore, we redesign the learning objective and propose an efficient framework for training contextual representation models. Specifically, the proposed approach bypasses the softmax layer by performing language modeling with dimension reduction, and allows the models to leverage pre-trained word embeddings. Our framework reduces the time spent on the output layer to a negligible level, eliminates almost all the trainable parameters of the softmax layer and performs language modeling without truncating the vocabulary. When applied to ELMo, our method achieves a 4 times speedup and eliminates 80% trainable parameters while achieving competitive performance on downstream tasks.

| Comments: | Work in progress                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | [arXiv:1902.11269](https://arxiv.org/abs/1902.11269) [cs.CL] |
|           | (or **arXiv:1902.11269v1 [cs.CL]** for this version)         |



