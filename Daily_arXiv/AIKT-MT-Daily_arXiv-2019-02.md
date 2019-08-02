# Daily arXiv: Machine Translation - Feb., 2019

### Index

* [2019-02-28](#2019-02-28)
  * [1. Non-Autoregressive Machine Translation with Auxiliary Regularization](#20190208-1)
  * [2. Multilingual Neural Machine Translation with Knowledge Distillation](#20190208-2)



# 2019-02-28

[Return to Index](#Index)

<h2 id="20190208-1">1. Non-Autoregressive Machine Translation with Auxiliary Regularization</h2> 
Title: [Non-Autoregressive Machine Translation with Auxiliary Regularization](https://arxiv.org/abs/1902.10245)

Authors: [Yiren Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Fei Tian](https://arxiv.org/search/cs?searchtype=author&query=Tian%2C+F), [Di He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+D), [Tao Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+T), [ChengXiang Zhai](https://arxiv.org/search/cs?searchtype=author&query=Zhai%2C+C), [Tie-Yan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+T)

*(Submitted on 22 Feb 2019)*

> As a new neural machine translation approach, Non-Autoregressive machine Translation (NAT) has attracted attention recently due to its high efficiency in inference. However, the high efficiency has come at the cost of not capturing the sequential dependency on the target side of translation, which causes NAT to suffer from two kinds of translation errors: 1) repeated translations (due to indistinguishable adjacent decoder hidden states), and 2) incomplete translations (due to incomplete transfer of source side information via the decoder hidden states). 
> In this paper, we propose to address these two problems by improving the quality of decoder hidden representations via two auxiliary regularization terms in the training process of an NAT model. First, to make the hidden states more distinguishable, we regularize the similarity between consecutive hidden states based on the corresponding target tokens. Second, to force the hidden states to contain all the information in the source sentence, we leverage the dual nature of translation tasks (e.g., English to German and German to English) and minimize a backward reconstruction error to ensure that the hidden states of the NAT decoder are able to recover the source side sentence. Extensive experiments conducted on several benchmark datasets show that both regularization strategies are effective and can alleviate the issues of repeated translations and incomplete translations in NAT models. The accuracy of NAT models is therefore improved significantly over the state-of-the-art NAT models with even better efficiency for inference.

| Comments: | AAAI 2019                                                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Machine Learning (stat.ML) |
| Cite as:  | [arXiv:1902.10245](https://arxiv.org/abs/1902.10245) [cs.CL] |





<h2 id="20190208-2">2. Multilingual Neural Machine Translation with Knowledge Distillation</h2> 
Title: [Multilingual Neural Machine Translation with Knowledge Distillation](https://arxiv.org/abs/1902.10461)

Authors: [Xu Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+X), [Yi Ren](https://arxiv.org/search/cs?searchtype=author&query=Ren%2C+Y), [Di He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+D), [Tao Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+T), [Zhou Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+Z), [Tieyan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+T)

*(Submitted on 27 Feb 2019)*

> Multilingual machine translation, which translates multiple languages with a single model, has attracted much attention due to its efficiency of offline training and online serving. However, traditional multilingual translation usually yields inferior accuracy compared with the counterpart using individual models for each language pair, due to language diversity and model capacity limitations. In this paper, we propose a distillation-based approach to boost the accuracy of multilingual machine translation. Specifically, individual models are first trained and regarded as teachers, and then the multilingual model is trained to fit the training data and match the outputs of individual models simultaneously through knowledge distillation. Experiments on IWSLT, WMT and Ted talk translation datasets demonstrate the effectiveness of our method. Particularly, we show that one model is enough to handle multiple languages (up to 44 languages in our experiment), with comparable or even better accuracy than individual models.

| Comments: | Accepted to ICLR 2019                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:1902.10461](https://arxiv.org/abs/1902.10461) [cs.CL] |
|           | (or **arXiv:1902.10461v1 [cs.CL]** for this version)         |

