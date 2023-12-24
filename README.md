# MRKD
This is an official implementation of “A Masked Reverse Knowledge Distillation Method Incorporating Global and Local Information for Image Anomaly Detection” (MRKD) with PyTorch, accepted by knowledge-based systems.<br />

[Paper link](https://www.sciencedirect.com/science/article/pii/S0950705123007323).<br />

**Abstract**: Knowledge distillation is an effective image anomaly detection and localization scheme. However, a major drawback of this scheme is its tendency to overly generalize, primarily due to the similarities between input and supervisory signals. In order to address this issue, this paper introduces a novel technique called masked reverse knowledge distillation (MRKD). By employing image-level masking (ILM) and feature-level masking (FLM), MRKD transforms the task of image reconstruction into image restoration. Specifically, ILM helps to capture global information by differentiating input signals from supervisory signals. On the other hand, FLM incorporates synthetic feature-level anomalies to ensure that the learned representations contain sufficient local information. With these two strategies, MRKD is endowed with stronger image context capture capacity and is less likely to be overgeneralized. Experiments on the widely-used MVTec anomaly detection dataset demonstrate that MRKD achieves impressive performance: image-level 98.9% AU-ROC, pixel-level 98.4% AU-ROC, and 95.3% AU-PRO. In addition, extensive ablation experiments have validated the superiority of MRKD in mitigating the overgeneralization problem..<br />

**Keywords**: Image anomaly detection, Knowledge distillation, Deep learning

# Implementation
1.Environment.<br />

2.Dataset.<br />
>Download the MVTec dataset [here](https://www.mvtec.com/company/research/datasets/mvtec-ad).<br />

3.Execute the following command to see the training and evaluation results..<br />
```
python main.py

```
