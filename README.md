# MUIT-TTA: Annotation-Free Intracranial Hemorrhage Segmentation via Pseudo-Anomaly Synthesis and Test-Time Adaptation

[![Stars](https://img.shields.io/github/stars/JustlfC03/MUIT-TTA?style=social)](https://github.com/JustlfC03/MUIT-TTA/stargazers)
[![Forks](https://img.shields.io/github/forks/JustlfC03/MUIT-TTA?style=social)](https://github.com/JustlfC03/MUIT-TTA/network/members)
[![Project Page](https://img.shields.io/badge/project-page-d9534f.svg)](https://JustlfC03.github.io/MUIT-TTA/) 

This repository provides the PyTorch implementation of the paper **"MUIT-TTA: Annotation-Free Intracranial Hemorrhage Segmentation via Pseudo-Anomaly Synthesis and Test-Time Adaptation"** (Under Review in *Pattern Recognition*). 

Our method maintains robust and accurate annotation-free segmentation performance even under severe domain shifts between synthetic normality-based training and real clinical abnormalities. Furthermore, to eliminate the reliance on real pathological annotations, we incorporate morphological pseudo-lesion synthesis and a multi-view, uncertainty-aware, and integrity-driven Test-Time Adaptation (MUIT-TTA) strategy to systematically bridge the synthetic-to-real domain gap and mitigate target-domain noise.

---

## Proposed method

Our framework consists of two main stages: Source-Domain Pseudo-Anomaly Synthesis and Target-Domain Test-Time Adaptation (MUIT-TTA).

![Synthesis Pipeline](img/Fig1_01.png) 
*Figure 1: Multi-subtype pseudo-anomaly synthesis framework. Generating ICH, SAH, IVH, and SDH/EDH from normal brain CTs.*

![MUIT-TTA Framework](img/Fig2_01.png) 
*Figure 2: Overview of the MUIT-TTA framework. Integrating entropy minimization, uncertainty-weighted pseudo-labeling, and integrity constraints.*

---
## Repository Structure

MUIT-TTA/
│
├── synthesize_anomalies.py     # Generates 4 pseudo-anomaly subtypes (ICH, SAH, IVH, SDH) via morphological operations
├── dataset2D.py                # PyTorch Dataset class for 2D medical images with transformations
│
├── nnunet2d.py                 # 2D nnU-Net backbone modified with Instance Normalization
├── tta_model.py                # Core MUIT-TTA adapter: Multi-view ensemble, uncertainty filtering, and integrity loss
│
├── train_source2D.py           # Core training loop, losses (Dice + CE), and evaluation metrics
├── run_training_2d.py          # Entry script for source domain pretraining
│
├── test_nnunet.py              # Standard inference and metric computation (DSC, HD95, ASSD, PPV)
├── run_tta.py                  # Entry script for Test-Time Adaptation inference
└── requirements.txt            # Environment dependencies

---
## Datasets

This study strictly avoids using any real pathological annotations during the training phase.

### 📌 BHSD (Train & Test)
- **Source:** [PhysioNet / GitHub](https://github.com/vlbthambawita/BHSD)  
- **Training:** Only the 2,000 non-hemorrhage (normal) samples are used to synthesize pseudo-anomalies via `synthesize_anomalies.py`.  
- **Testing:** 192 annotated hemorrhage cases are reserved for in-distribution evaluation.  

---

### 📌 INSTANCE 2022 (Test Only)
- **Source:** [Grand Challenge](https://instance.grand-challenge.org/)  
- **Usage:** 100 scans are used as an external cross-domain test set.  

---

### 📌 CT-ICH (Test Only)
- **Source:** [PhysioNet](https://physionet.org/content/ct-ich/1.3.1/)  
- **Usage:** 36 positive cases are retained as an external cross-domain test set.  

---

### ⚙️ Preprocessing
All 3D volumes are converted into 2D slices and saved as `.png` files before running the scripts.
    data/
    ├── images/   # input CT slices
    ├── masks/    # corresponding masks (if available)

---
### Notes

- No real lesion annotations are used during training.  
- All anomaly patterns are generated via pseudo-anomaly synthesis.  
- External datasets are used **only for evaluation**, ensuring a strict domain generalization setting.

