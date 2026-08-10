# CTINet

Official implementation of:

**CTINet: An fNIRS-Informed Cross-Modal Token Interaction Network for EEG-fNIRS Fusion**

## Overview

CTINet is a hybrid EEG-fNIRS decoding framework for cognitive-task classification. The framework consists of Channel-wise Hemodynamic Modulation (CHM), Multi-window Statistical Tokenization (MWST), a bidirectional Cross-Modal Interactor Transformer (CMIT), Temporal Attention Pooling (TAP), and Modality Adaptive Fusion (MAF).

This repository provides the core PyTorch implementation of CTINet and the subject-dependent training code used in the main experiments.

## Dataset

- [Open access dataset for simultaneous EEG and NIRS brain-computer interface (BCI)](https://doi.org/10.82901/nemar.nm000267)
- [Simultaneous acquisition of EEG and NIRS during cognitive tasks for an open access dataset](https://doi.org/10.14279/depositonce-5830.2)

## Data Preprocessing

The preprocessing procedure follows that described in:

M. Liu et al.,  
**"STA-Net: Spatial-Temporal Alignment Network for Hybrid EEG-fNIRS Decoding,"**  
*Information Fusion*, vol. 119, 103023, 2025.

Please refer to the CTINet manuscript for the preprocessing and input-construction details used in this study.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```
## Environment

The code was developed and tested using the following environment:

- Python 3.12.7
- PyTorch 2.7.1+cu126
- CUDA 12.6
- NumPy 1.26.4
- pandas 2.2.2
- scikit-learn 1.5.1
- Matplotlib 3.9.2
- openpyxl 3.1.5
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU

## Usage

Run all three tasks:

```bash
python train_subject_dependent.py
```

Run an individual task:

```bash
python train_subject_dependent.py --task MI
python train_subject_dependent.py --task MA
python train_subject_dependent.py --task WG
```

## Code Availability

This repository provides the core CTINet model implementation and the subject-dependent training code.

The current release includes:

- `ctinet.py`: CTINet model implementation.
- `train_subject_dependent.py`: Subject-dependent training code.
- `requirements.txt`: Required Python packages.

## Citation

If you use this code in your research, please cite:

**CTINet: An fNIRS-Informed Cross-Modal Token Interaction Network for EEG-fNIRS Fusion**

Citation information will be updated after publication.

## Contact

If you have any questions, please contact us at [yanghaiqiang@qdu.edu.cn](mailto:yanghaiqiang@qdu.edu.cn).
