# MRKD - A Masked Reverse Knowledge Distillation Method Incorporating Global and Local Information for Image Anomaly Detection
This is an official implementation of “A Masked Reverse Knowledge Distillation Method Incorporating Global and Local Information for Image Anomaly Detection” (MRKD) with PyTorch, accepted by knowledge-based systems.<br />

[Paper link](https://www.sciencedirect.com/science/article/pii/S0950705123007323).<br />

![](https://github.com/yuxin-jiang/MRKD/blob/main/figures/figure1.png)

**Abstract**: Knowledge distillation is an effective image anomaly detection and localization scheme. However, a major drawback of this scheme is its tendency to overly generalize, primarily due to the similarities between input and supervisory signals. In order to address this issue, this paper introduces a novel technique called masked reverse knowledge distillation (MRKD). By employing image-level masking (ILM) and feature-level masking (FLM), MRKD transforms the task of image reconstruction into image restoration. Specifically, ILM helps to capture global information by differentiating input signals from supervisory signals. On the other hand, FLM incorporates synthetic feature-level anomalies to ensure that the learned representations contain sufficient local information. With these two strategies, MRKD is endowed with stronger image context capture capacity and is less likely to be overgeneralized. Experiments on the widely-used MVTec anomaly detection dataset demonstrate that MRKD achieves impressive performance: image-level 98.9% AU-ROC, pixel-level 98.4% AU-ROC, and 95.3% AU-PRO. In addition, extensive ablation experiments have validated the superiority of MRKD in mitigating the overgeneralization problem..<br />

**Keywords**: Image anomaly detection, Knowledge distillation, Deep learning

# Implementation
1. Environment.<br />
>pytorch == 1.12.0

>torchvision == 0.13.0

>numpy == 1.21.6

>scipy == 1.7.3

>matplotlib == 3.5.2

>tqdm

2. Dataset.<br />
>Download the MVTec dataset [here](https://www.mvtec.com/company/research/datasets/mvtec-ad).<br />

3. Execute the following command to see the training and evaluation results.<br />
```
python main.py

# Visualization

![](https://github.com/yuxin-jiang/MRKD/blob/main/figures/result.png)

```
# Reference
```
@article{jiang2023masked,
  title={A masked reverse knowledge distillation method incorporating global and local information for image anomaly detection},
  author={Jiang, Yuxin and Cao, Yunkang and Shen, Weiming},
  journal={Knowledge-Based Systems},
  volume={280},
  pages={110982},
  year={2023},
  publisher={Elsevier}
}
```
