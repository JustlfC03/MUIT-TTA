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
```text
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
```
---
## Datasets

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

All 3D CT volumes are converted into 2D slices, where brain regions are extracted to remove surrounding background (black borders), and each slice is uniformly resized to a fixed resolution of **256 × 256** for consistent model input.

The expected data structure is as follows:

```text
data/
├── images/   # input CT slices
└── masks/    # corresponding masks
```
---

## Environment

Please ensure your environment matches the following core specifications to guarantee reproducibility:

* **Python:** 3.10
* **CUDA:** 12.1
* **PyTorch:** 2.11.0

```bash
# Create and activate conda environment
conda create -n muit_tta python=3.10 -y
conda activate muit_tta

# Install PyTorch (CUDA 12.1)
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install core required packages
pip install numpy==2.2.6 opencv-python==4.12.0.88 pillow==12.0.0 scipy==1.15.3 scikit-image==0.25.2 scikit-learn==1.7.2 tqdm==4.66.1 thop==0.1.1.post2209072238 matplotlib==3.10.8
```
