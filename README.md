# MUIT-TTA: Annotation-Free Intracranial Hemorrhage Segmentation via Pseudo-Anomaly Synthesis and Test-Time Adaptation

[![Stars](https://img.shields.io/github/stars/JustlfC03/MUIT-TTA?style=social)](https://github.com/JustlfC03/MUIT-TTA/stargazers)
[![Forks](https://img.shields.io/github/forks/JustlfC03/MUIT-TTA?style=social)](https://github.com/JustlfC03/MUIT-TTA/network/members)
[![Project Page](https://img.shields.io/badge/project-page-d9534f.svg)](https://JustlfC03.github.io/MUIT-TTA/) 

This repository provides the PyTorch implementation of the paper **"MUIT-TTA: Annotation-Free Intracranial Hemorrhage Segmentation via Pseudo-Anomaly Synthesis and Test-Time Adaptation"** (Under Review in *Pattern Recognition*). 

Our method maintains robust and accurate annotation-free segmentation performance even under severe domain shifts between synthetic normality-based training and real clinical abnormalities. Furthermore, to eliminate the reliance on real pathological annotations, we incorporate morphological pseudo-lesion synthesis and a multi-view, uncertainty-aware, and integrity-driven Test-Time Adaptation (MUIT-TTA) strategy to systematically bridge the synthetic-to-real domain gap and mitigate target-domain noise.

---

## 1. Methodology Flowchart

Our framework consists of two main stages: Source-Domain Pseudo-Anomaly Synthesis and Target-Domain Test-Time Adaptation (MUIT-TTA).

> **[TODO: Replace with your actual image paths]**
![Synthesis Pipeline](docs/synthesis.png) 
*Figure 1: Multi-subtype pseudo-anomaly synthesis framework. Generating ICH, SAH, IVH, and SDH/EDH from normal brain CTs.*

![MUIT-TTA Framework](docs/framework.png) 
*Figure 2: Overview of the MUIT-TTA framework. Integrating entropy minimization, uncertainty-weighted pseudo-labeling, and integrity constraints.*

---
