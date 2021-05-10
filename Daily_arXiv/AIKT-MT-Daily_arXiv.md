# Daily arXiv: Machine Translation - May, 2021

# Index


- [2021-05-10](#2021-05-10)
  
  - [1. Adapting by Pruning: A Case Study on BERT](#2021-05-10-1)
  - [2. On-the-Fly Controlled Text Generation with Experts and Anti-Experts](#2021-05-10-2)
  - [3. Regression Bugs Are In Your Model! Measuring, Reducing and Analyzing Regressions In NLP Model Updates](#2021-05-10-3)
  - [4. A Survey of Data Augmentation Approaches for NLP](#2021-05-10-4)
  - [5. Learning Shared Semantic Space for Speech-to-Text Translation](#2021-05-10-5)
  - [6. Translation Quality Assessment: A Brief Survey on Manual and Automatic Methods](#2021-05-10-6)
  - [7. Are Pre-trained Convolutions Better than Pre-trained Transformers?](#2021-05-10-7)
  - [8. ∂-Explainer: Abductive Natural Language Inference via Differentiable Convex Optimization](#2021-05-10-8)
- [2021-05-07](#2021-05-07)
  - [1. XeroAlign: Zero-Shot Cross-lingual Transformer Alignment](#2021-05-07-1)
  - [2. Quantitative Evaluation of Alternative Translations in a Corpus of Highly Dissimilar Finnish Paraphrases](#2021-05-07-2)
  - [3. Content4All Open Research Sign Language Translation Datasets](#2021-05-07-3)
  - [4. Reliability Testing for Natural Language Processing Systems](#2021-05-07-4)
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



# 2021-05-10

[Return to Index](#Index)



<h2 id="2021-05-10-1">1. Adapting by Pruning: A Case Study on BERT
</h2>

Title: [Adapting by Pruning: A Case Study on BERT](https://arxiv.org/abs/2105.03343)

Authors: [Yang Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+Y), [Nicolo Colombo](https://arxiv.org/search/cs?searchtype=author&query=Colombo%2C+N), [Wei Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+W)

> Adapting pre-trained neural models to downstream tasks has become the standard practice for obtaining high-quality models. In this work, we propose a novel model adaptation paradigm, adapting by pruning, which prunes neural connections in the pre-trained model to optimise the performance on the target task; all remaining connections have their weights intact. We formulate adapting-by-pruning as an optimisation problem with a differentiable loss and propose an efficient algorithm to prune the model. We prove that the algorithm is near-optimal under standard assumptions and apply the algorithm to adapt BERT to some GLUE tasks. Results suggest that our method can prune up to 50% weights in BERT while yielding similar performance compared to the fine-tuned full model. We also compare our method with other state-of-the-art pruning methods and study the topological differences of their obtained sub-networks.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2105.03343](https://arxiv.org/abs/2105.03343) [cs.LG]** |
|           | (or **[arXiv:2105.03343v1](https://arxiv.org/abs/2105.03343v1) [cs.LG]** for this version) |





<h2 id="2021-05-10-2">2. On-the-Fly Controlled Text Generation with Experts and Anti-Experts
</h2>

Title: [On-the-Fly Controlled Text Generation with Experts and Anti-Experts](https://arxiv.org/abs/2105.03023)

Authors: [Alisa Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+A), [Maarten Sap](https://arxiv.org/search/cs?searchtype=author&query=Sap%2C+M), [Ximing Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+X), [Swabha Swayamdipta](https://arxiv.org/search/cs?searchtype=author&query=Swayamdipta%2C+S), [Chandra Bhagavatula](https://arxiv.org/search/cs?searchtype=author&query=Bhagavatula%2C+C), [Noah A. Smith](https://arxiv.org/search/cs?searchtype=author&query=Smith%2C+N+A), [Yejin Choi](https://arxiv.org/search/cs?searchtype=author&query=Choi%2C+Y)

> Despite recent advances in natural language generation, it remains challenging to control attributes of generated text. We propose DExperts: Decoding-time Experts, a decoding-time method for controlled text generation which combines a pretrained language model with experts and/or anti-experts in an ensemble of language models. Intuitively, under our ensemble, output tokens only get high probability if they are considered likely by the experts, and unlikely by the anti-experts. We apply DExperts to language detoxification and sentiment-controlled generation, where we outperform existing controllable generation methods on both automatic and human evaluations. Our work highlights the promise of using LMs trained on text with (un)desired attributes for efficient decoding-time controlled language generation.

| Comments: | Accepted to ACL 2021, camera-ready version coming soon       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.03023](https://arxiv.org/abs/2105.03023) [cs.CL]** |
|           | (or **[arXiv:2105.03023v1](https://arxiv.org/abs/2105.03023v1) [cs.CL]** for this version) |





<h2 id="2021-05-10-3">3. Regression Bugs Are In Your Model! Measuring, Reducing and Analyzing Regressions In NLP Model Updates
</h2>

Title: [Regression Bugs Are In Your Model! Measuring, Reducing and Analyzing Regressions In NLP Model Updates](https://arxiv.org/abs/2105.03048)

Authors: [Yuqing Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+Y), [Yi-an Lai](https://arxiv.org/search/cs?searchtype=author&query=Lai%2C+Y), [Yuanjun Xiong](https://arxiv.org/search/cs?searchtype=author&query=Xiong%2C+Y), [Yi Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Stefano Soatto](https://arxiv.org/search/cs?searchtype=author&query=Soatto%2C+S)

> Behavior of deep neural networks can be inconsistent between different versions. Regressions during model update are a common cause of concern that often over-weigh the benefits in accuracy or efficiency gain. This work focuses on quantifying, reducing and analyzing regression errors in the NLP model updates. Using negative flip rate as regression measure, we show that regression has a prevalent presence across tasks in the GLUE benchmark. We formulate the regression-free model updates into a constrained optimization problem, and further reduce it into a relaxed form which can be approximately optimized through knowledge distillation training method. We empirically analyze how model ensemble reduces regression. Finally, we conduct CheckList behavioral testing to understand the distribution of regressions across linguistic phenomena, and the efficacy of ensemble and distillation methods.

| Comments: | 13 pages, 3 figures, Accepted at ACL 2021 main conference    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.03048](https://arxiv.org/abs/2105.03048) [cs.CL]** |
|           | (or **[arXiv:2105.03048v1](https://arxiv.org/abs/2105.03048v1) [cs.CL]** for this version) |





<h2 id="2021-05-10-4">4. A Survey of Data Augmentation Approaches for NLP
</h2>

Title: [A Survey of Data Augmentation Approaches for NLP](https://arxiv.org/abs/2105.03075)

Authors: [Steven Y. Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+S+Y), [Varun Gangal](https://arxiv.org/search/cs?searchtype=author&query=Gangal%2C+V), [Jason Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+J), [Sarath Chandar](https://arxiv.org/search/cs?searchtype=author&query=Chandar%2C+S), [Soroush Vosoughi](https://arxiv.org/search/cs?searchtype=author&query=Vosoughi%2C+S), [Teruko Mitamura](https://arxiv.org/search/cs?searchtype=author&query=Mitamura%2C+T), [Eduard Hovy](https://arxiv.org/search/cs?searchtype=author&query=Hovy%2C+E)

> Data augmentation has recently seen increased interest in NLP due to more work in low-resource domains, new tasks, and the popularity of large-scale neural networks that require large amounts of training data. Despite this recent upsurge, this area is still relatively underexplored, perhaps due to the challenges posed by the discrete nature of language data. In this paper, we present a comprehensive and unifying survey of data augmentation for NLP by summarizing the literature in a structured manner. We first introduce and motivate data augmentation for NLP, and then discuss major methodologically representative approaches. Next, we highlight techniques that are used for popular NLP applications and tasks. We conclude by outlining current challenges and directions for future research. Overall, our paper aims to clarify the landscape of existing literature in data augmentation for NLP and motivate additional work in this area.

| Comments: | Accepted to ACL 2021 Findings                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2105.03075](https://arxiv.org/abs/2105.03075) [cs.CL]** |
|           | (or **[arXiv:2105.03075v1](https://arxiv.org/abs/2105.03075v1) [cs.CL]** for this version) |





<h2 id="2021-05-10-5">5. Learning Shared Semantic Space for Speech-to-Text Translation
</h2>

Title: [Learning Shared Semantic Space for Speech-to-Text Translation](https://arxiv.org/abs/2105.03095)

Authors: [Chi Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+C), [Mingxuan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+M), [Heng Ji](https://arxiv.org/search/cs?searchtype=author&query=Ji%2C+H), [Lei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L)

> Having numerous potential applications and great impact, end-to-end speech translation (ST) has long been treated as an independent task, failing to fully draw strength from the rapid advances of its sibling - text machine translation (MT). With text and audio inputs represented differently, the modality gap has rendered MT data and its end-to-end models incompatible with their ST counterparts. In observation of this obstacle, we propose to bridge this representation gap with Chimera. By projecting audio and text features to a common semantic representation, Chimera unifies MT and ST tasks and boosts the performance on ST benchmark, MuST-C, to a new state-of-the-art. Specifically, Chimera obtains 26.3 BLEU on EN-DE, improving the SOTA by a +2.7 BLEU margin. Further experimental analyses demonstrate that the shared semantic space indeed conveys common knowledge between these two tasks and thus paves a new way for augmenting training resources across modalities.

| Comments: | 8 pages, 5 figures, Accepted by Findings of ACL 2021         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.03095](https://arxiv.org/abs/2105.03095) [cs.CL]** |
|           | (or **[arXiv:2105.03095v1](https://arxiv.org/abs/2105.03095v1) [cs.CL]** for this version) |





<h2 id="2021-05-10-6">6. Translation Quality Assessment: A Brief Survey on Manual and Automatic Methods
</h2>

Title: [Translation Quality Assessment: A Brief Survey on Manual and Automatic Methods](https://arxiv.org/abs/2105.03311)

Authors: [Lifeng Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+L), [Gareth J. F. Jones](https://arxiv.org/search/cs?searchtype=author&query=Jones%2C+G+J+F), [Alan F. Smeaton](https://arxiv.org/search/cs?searchtype=author&query=Smeaton%2C+A+F)

> To facilitate effective translation modeling and translation studies, one of the crucial questions to address is how to assess translation quality. From the perspectives of accuracy, reliability, repeatability and cost, translation quality assessment (TQA) itself is a rich and challenging task. In this work, we present a high-level and concise survey of TQA methods, including both manual judgement criteria and automated evaluation metrics, which we classify into further detailed sub-categories. We hope that this work will be an asset for both translation model researchers and quality assessment researchers. In addition, we hope that it will enable practitioners to quickly develop a better understanding of the conventional TQA field, and to find corresponding closely relevant evaluation solutions for their own needs. This work may also serve inspire further development of quality assessment and evaluation methodologies for other natural language processing (NLP) tasks in addition to machine translation (MT), such as automatic text summarization (ATS), natural language understanding (NLU) and natural language generation (NLG).

| Comments: | Accepted to 23rd Nordic Conference on Computational Linguistics (NoDaLiDa 2021): Workshop on Modelling Translation: Translatology in the Digital Age (MoTra21). arXiv admin note: substantial text overlap with [arXiv:1605.04515](https://arxiv.org/abs/1605.04515) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.03311](https://arxiv.org/abs/2105.03311) [cs.CL]** |
|           | (or **[arXiv:2105.03311v1](https://arxiv.org/abs/2105.03311v1) [cs.CL]** for this version) |





<h2 id="2021-05-10-7">7. Are Pre-trained Convolutions Better than Pre-trained Transformers?
</h2>

Title: [Are Pre-trained Convolutions Better than Pre-trained Transformers?](https://arxiv.org/abs/2105.03322)

Authors: [Yi Tay](https://arxiv.org/search/cs?searchtype=author&query=Tay%2C+Y), [Mostafa Dehghani](https://arxiv.org/search/cs?searchtype=author&query=Dehghani%2C+M), [Jai Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+J), [Dara Bahri](https://arxiv.org/search/cs?searchtype=author&query=Bahri%2C+D), [Vamsi Aribandi](https://arxiv.org/search/cs?searchtype=author&query=Aribandi%2C+V), [Zhen Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+Z), [Donald Metzler](https://arxiv.org/search/cs?searchtype=author&query=Metzler%2C+D)

> In the era of pre-trained language models, Transformers are the de facto choice of model architectures. While recent research has shown promise in entirely convolutional, or CNN, architectures, they have not been explored using the pre-train-fine-tune paradigm. In the context of language models, are convolutional models competitive to Transformers when pre-trained? This paper investigates this research question and presents several interesting findings. Across an extensive set of experiments on 8 datasets/tasks, we find that CNN-based pre-trained models are competitive and outperform their Transformer counterpart in certain scenarios, albeit with caveats. Overall, the findings outlined in this paper suggest that conflating pre-training and architectural advances is misguided and that both advances should be considered independently. We believe our research paves the way for a healthy amount of optimism in alternative architectures.

| Comments: | Accepted to ACL 2021                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2105.03322](https://arxiv.org/abs/2105.03322) [cs.CL]** |
|           | (or **[arXiv:2105.03322v1](https://arxiv.org/abs/2105.03322v1) [cs.CL]** for this version) |





<h2 id="2021-05-10-8">8. ∂-Explainer: Abductive Natural Language Inference via Differentiable Convex Optimization
</h2>

Title: [∂-Explainer: Abductive Natural Language Inference via Differentiable Convex Optimization](https://arxiv.org/abs/2105.03417)

Authors: [Mokanarangan Thayaparan](https://arxiv.org/search/cs?searchtype=author&query=Thayaparan%2C+M), [Marco Valentino](https://arxiv.org/search/cs?searchtype=author&query=Valentino%2C+M), [Deborah Ferreira](https://arxiv.org/search/cs?searchtype=author&query=Ferreira%2C+D), [Julia Rozanova](https://arxiv.org/search/cs?searchtype=author&query=Rozanova%2C+J), [André Freitas](https://arxiv.org/search/cs?searchtype=author&query=Freitas%2C+A)

> Constrained optimization solvers with Integer Linear programming (ILP) have been the cornerstone for explainable natural language inference during its inception. ILP based approaches provide a way to encode explicit and controllable assumptions casting natural language inference as an abductive reasoning problem, where the solver constructs a plausible explanation for a given hypothesis. While constrained based solvers provide explanations, they are often limited by the use of explicit constraints and cannot be integrated as part of broader deep neural architectures. In contrast, state-of-the-art transformer-based models can learn from data and implicitly encode complex constraints. However, these models are intrinsically black boxes. This paper presents a novel framework named ∂-Explainer (Diff-Explainer) that combines the best of both worlds by casting the constrained optimization as part of a deep neural network via differentiable convex optimization and fine-tuning pre-trained transformers for downstream explainable NLP tasks. To demonstrate the efficacy of the framework, we transform the constraints presented by TupleILP and integrate them with sentence embedding transformers for the task of explainable science QA. Our experiments show up to ≈10% improvement over non-differentiable solver while still providing explanations for supporting its inference.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2105.03417](https://arxiv.org/abs/2105.03417) [cs.CL]** |
|           | (or **[arXiv:2105.03417v1](https://arxiv.org/abs/2105.03417v1) [cs.CL]** for this version) |






# 2021-05-07

[Return to Index](#Index)



<h2 id="2021-05-07-1">1. XeroAlign: Zero-Shot Cross-lingual Transformer Alignment
</h2>

Title: [XeroAlign: Zero-Shot Cross-lingual Transformer Alignment](https://arxiv.org/abs/2105.02472)

Authors: [Milan Gritta](https://arxiv.org/search/cs?searchtype=author&query=Gritta%2C+M), [Ignacio Iacobacci](https://arxiv.org/search/cs?searchtype=author&query=Iacobacci%2C+I)

> The introduction of pretrained cross-lingual language models brought decisive improvements to multilingual NLP tasks. However, the lack of labelled task data necessitates a variety of methods aiming to close the gap to high-resource languages. Zero-shot methods in particular, often use translated task data as a training signal to bridge the performance gap between the source and target language(s). We introduce XeroAlign, a simple method for task-specific alignment of cross-lingual pretrained transformers such as XLM-R. XeroAlign uses translated task data to encourage the model to generate similar sentence embeddings for different languages. The XeroAligned XLM-R, called XLM-RA, shows strong improvements over the baseline models to achieve state-of-the-art zero-shot results on three multilingual natural language understanding tasks. XLM-RA's text classification accuracy exceeds that of XLM-R trained with labelled data and performs on par with state-of-the-art models on a cross-lingual adversarial paraphrasing task.

| Comments: | Accepted as long paper at Findings of ACL 2021               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.02472](https://arxiv.org/abs/2105.02472) [cs.CL]** |
|           | (or **[arXiv:2105.02472v1](https://arxiv.org/abs/2105.02472v1) [cs.CL]** for this version) |





<h2 id="2021-05-07-2">2. Quantitative Evaluation of Alternative Translations in a Corpus of Highly Dissimilar Finnish Paraphrases
</h2>

Title: [Quantitative Evaluation of Alternative Translations in a Corpus of Highly Dissimilar Finnish Paraphrases](https://arxiv.org/abs/2105.02477)

Authors: [Li-Hsin Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang%2C+L), [Sampo Pyysalo](https://arxiv.org/search/cs?searchtype=author&query=Pyysalo%2C+S), [Jenna Kanerva](https://arxiv.org/search/cs?searchtype=author&query=Kanerva%2C+J), [Filip Ginter](https://arxiv.org/search/cs?searchtype=author&query=Ginter%2C+F)

> In this paper, we present a quantitative evaluation of differences between alternative translations in a large recently released Finnish paraphrase corpus focusing in particular on non-trivial variation in translation. We combine a series of automatic steps detecting systematic variation with manual analysis to reveal regularities and identify categories of translation differences. We find the paraphrase corpus to contain highly non-trivial translation variants difficult to recognize through automatic approaches.

| Comments: | Accepted to Workshop on MOdelling TRAnslation: Translatology in the Digital Age |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2105.02477](https://arxiv.org/abs/2105.02477) [cs.CL]** |
|           | (or **[arXiv:2105.02477v1](https://arxiv.org/abs/2105.02477v1) [cs.CL]** for this version) |





<h2 id="2021-05-07-3">3. Content4All Open Research Sign Language Translation Datasets
</h2>

Title: [Content4All Open Research Sign Language Translation Datasets](https://arxiv.org/abs/2105.02351)

Authors: [Necati Cihan Camgoz](https://arxiv.org/search/cs?searchtype=author&query=Camgoz%2C+N+C), [Ben Saunders](https://arxiv.org/search/cs?searchtype=author&query=Saunders%2C+B), [Guillaume Rochette](https://arxiv.org/search/cs?searchtype=author&query=Rochette%2C+G), [Marco Giovanelli](https://arxiv.org/search/cs?searchtype=author&query=Giovanelli%2C+M), [Giacomo Inches](https://arxiv.org/search/cs?searchtype=author&query=Inches%2C+G), [Robin Nachtrab-Ribback](https://arxiv.org/search/cs?searchtype=author&query=Nachtrab-Ribback%2C+R), [Richard Bowden](https://arxiv.org/search/cs?searchtype=author&query=Bowden%2C+R)

> Computational sign language research lacks the large-scale datasets that enables the creation of useful reallife applications. To date, most research has been limited to prototype systems on small domains of discourse, e.g. weather forecasts. To address this issue and to push the field forward, we release six datasets comprised of 190 hours of footage on the larger domain of news. From this, 20 hours of footage have been annotated by Deaf experts and interpreters and is made publicly available for research purposes. In this paper, we share the dataset collection process and tools developed to enable the alignment of sign language video and subtitles, as well as baseline translation results to underpin future research.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2105.02351](https://arxiv.org/abs/2105.02351) [cs.CV]** |
|           | (or **[arXiv:2105.02351v1](https://arxiv.org/abs/2105.02351v1) [cs.CV]** for this version) |





<h2 id="2021-05-07-4">4. Reliability Testing for Natural Language Processing Systems
</h2>

Title: [Reliability Testing for Natural Language Processing Systems](https://arxiv.org/abs/2105.02590)

Authors: [Samson Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+S), [Shafiq Joty](https://arxiv.org/search/cs?searchtype=author&query=Joty%2C+S), [Kathy Baxter](https://arxiv.org/search/cs?searchtype=author&query=Baxter%2C+K), [Araz Taeihagh](https://arxiv.org/search/cs?searchtype=author&query=Taeihagh%2C+A), [Gregory A. Bennett](https://arxiv.org/search/cs?searchtype=author&query=Bennett%2C+G+A), [Min-Yen Kan](https://arxiv.org/search/cs?searchtype=author&query=Kan%2C+M)

> Questions of fairness, robustness, and transparency are paramount to address before deploying NLP systems. Central to these concerns is the question of reliability: Can NLP systems reliably treat different demographics fairly and function correctly in diverse and noisy environments? To address this, we argue for the need for reliability testing and contextualize it among existing work on improving accountability. We show how adversarial attacks can be reframed for this goal, via a framework for developing reliability tests. We argue that reliability testing -- with an emphasis on interdisciplinary collaboration -- will enable rigorous and targeted testing, and aid in the enactment and enforcement of industry standards.

| Comments: | Accepted to ACL-IJCNLP 2021 (main conference). Final camera-ready version to follow shortly |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2105.02590](https://arxiv.org/abs/2105.02590) [cs.LG]** |
|           | (or **[arXiv:2105.02590v1](https://arxiv.org/abs/2105.02590v1) [cs.LG]** for this version) |









# 2021-05-06

[Return to Index](#Index)



<h2 id="2021-05-06-1">1. Data Augmentation by Concatenation for Low-Resource Translation: A Mystery and a Solution
</h2>

Title: [Data Augmentation by Concatenation for Low-Resource Translation: A Mystery and a Solution](https://arxiv.org/abs/2105.01691)

Authors: [Toan Q. Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+T+Q), [Kenton Murray](https://arxiv.org/search/cs?searchtype=author&query=Murray%2C+K), [David Chiang](https://arxiv.org/search/cs?searchtype=author&query=Chiang%2C+D)

> In this paper, we investigate the driving factors behind concatenation, a simple but effective data augmentation method for low-resource neural machine translation. Our experiments suggest that discourse context is unlikely the cause for the improvement of about +1 BLEU across four language pairs. Instead, we demonstrate that the improvement comes from three other factors unrelated to discourse: context diversity, length diversity, and (to a lesser extent) position shifting.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2105.01691](https://arxiv.org/abs/2105.01691) [cs.CL]** |
|           | (or **[arXiv:2105.01691v1](https://arxiv.org/abs/2105.01691v1) [cs.CL]** for this version) |



<h2 id="2021-05-06-2">2. Full-Sentence Models Perform Better in Simultaneous Translation Using the Information Enhanced Decoding Strategy
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

