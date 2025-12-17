<div align="left">
<br>
<br>
</div>
<div align="center">
<h1>MRKD: A Masked Reverse Knowledge Distillation Method Incorporating Global and Local Information for Image Anomaly Detection</h1>
<div>
&nbsp;&nbsp;&nbsp;&nbsp;Yuxin Jiang<sup>1</sup>&emsp;
&nbsp;&nbsp;&nbsp;&nbsp;Yunkang Cao<sup>1 </sup>&emsp;
&nbsp;&nbsp;&nbsp;&nbsp;Weiming Shen<sup>1, *</sup>&emsp;
</div>
<div>
&nbsp;&nbsp;&nbsp;&nbsp;<sup>1</sup>Huazhong University of Science and Technology
</div>
  
[[GitHub Repository]](https://github.com/yuxin-jiang/MRKD) [[Paper]](https://www.sciencedirect.com/science/article/pii/S0950705123007323)
  
![Framework](https://github.com/yuxin-jiang/MRKD/blob/main/figures/figure1.png)

---
</div>

>**Abstract:** Knowledge distillation is an effective image anomaly detection and localization scheme. However, a major drawback of this scheme is its tendency to overly generalize, primarily due to the similarities between input and supervisory signals. In order to address this issue, this paper introduces a novel technique called masked reverse knowledge distillation (MRKD). By employing image-level masking (ILM) and feature-level masking (FLM), MRKD transforms the task of image reconstruction into image restoration. Specifically, ILM helps to capture global information by differentiating input signals from supervisory signals. On the other hand, FLM incorporates synthetic feature-level anomalies to ensure that the learned representations contain sufficient local information. With these two strategies, MRKD is endowed with stronger image context capture capacity and is less likely to be overgeneralized. Experiments on the widely-used MVTec anomaly detection dataset demonstrate that MRKD achieves impressive performance: image-level 98.9% AU-ROC, pixel-level 98.4% AU-ROC, and 95.3% AU-PRO. In addition, extensive ablation experiments have validated the superiority of MRKD in mitigating the overgeneralization problem.

**Keywords:** Image anomaly detection; Knowledge distillation; Deep learning

## 💻 Requirements
- pytorch == 1.12.0
- torchvision == 0.13.0
- numpy == 1.21.6
- scipy == 1.7.3
- matplotlib == 3.5.2
- tqdm

## 📥 Dataset
Download the MVTec AD dataset [here](https://www.mvtec.com/company/research/datasets/mvtec-ad).

## 🚀 Usage
Execute the following command for training and evaluation:
```bash
python main.py
```

## 📊 Results

- **Anomaly detection results**:
![Results](https://github.com/yuxin-jiang/MRKD/blob/main/figures/table1.png)

- **Anomaly localization results**:
![Results](https://github.com/yuxin-jiang/MRKD/blob/main/figures/table2.png)

## 🖼️ Visualization
![Qualitative Results](https://github.com/yuxin-jiang/MRKD/blob/main/figures/result.png)

## 📝 Citation
If you find this work useful, please consider citing:
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
