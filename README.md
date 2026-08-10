# CTINet

Official implementation of:

**CTINet: An fNIRS-Informed Cross-Modal Token Interaction Network for EEG-fNIRS Fusion**

## Overview

CTINet is a hybrid EEG-fNIRS decoding framework for cognitive-task classification. The framework consists of Channel-wise Hemodynamic Modulation (CHM), Multi-window Statistical Tokenization (MWST), a bidirectional Cross-Modal Interactor Transformer (CMIT), Temporal Attention Pooling (TAP), and Modality Adaptive Fusion (MAF).

This repository provides the core PyTorch implementation of CTINet and the subject-dependent training code used in the main experiments.

## Datasets

This study uses two publicly available simultaneous EEG-fNIRS datasets.

### Motor Imagery (MI) and Mental Arithmetic (MA)

J. Shin et al.,  
**"Open Access Dataset for EEG+NIRS Single-Trial Classification,"**  
*IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 2017.

Dataset:  
https://doi.org/10.82901/nemar.nm000267

### Word Generation (WG)

J. Shin et al.,  
**"Simultaneous Acquisition of EEG and NIRS During Cognitive Tasks for an Open Access Dataset,"**  
*Scientific Data*, 2018.

Dataset:  
https://doi.org/10.14279/depositonce-5830.2

Raw EEG and fNIRS data are not redistributed in this repository.

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

## License

This project is released under the MIT License.
