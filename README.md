# Editprint

An official implementation code for paper "Editprint: General Digital Image Forensics via Editing Fingerprint with Self-Augmentation Training".

## Table of Contents

- [Introduction](#introduction)
- [Dependency](#dependency)
- [Zero-shot Testing](#testing)

## Introduction

Digital image forensics, as an indispensable component of ensuring the trustworthiness of image data, finds extensive applications in fields such as image forgery localization (IFL), camera source identification (CSI), etc. In recent years, there has been a growing interest in self-supervised **general** forensics, as opposed to developing specific forensics for each individual task. Many existing forensic methods primarily focus on traces left by the camera, e.g., the well-known Photo Response Non-Uniformity (PRNU). While these methods have achieved a certain level of success, they suffer from two limitations: 1) their applicability is inherently confined to camera-related tasks, e.g., CSI, and 2) they demand a substantial amount of annotated training data.

To address these constraints, in this work, we introduce a novel general forensic feature, termed as Editprint, that captures a highly diverse range of in- and out-camera traces, with only minimal unlabeled training data. Ideally, we expect that any images with the same imaging, editing, and transmitting processes would have the same Editprints, and vice versa. To model the in- and out-camera operations, we devise an online editing pool based on data self-augmentation strategies. Requiring only minimal (e.g., 10) training data, the editing pool can simulate massive (e.g., 10^7) editing chains and traces arising from the in-camera processing and the subsequent out-camera operations. To ensure that Editprint exhibits high discriminative capabilities across various editing chains, we propose using textual descriptions of these chains as labels and supervise the Editprint through language-guided contrastive learning.

Extensive experiments demonstrate the superiority of our proposed Editprint over existing self-supervised forensics, particularly in non-camera related applications, e.g., IFL, social network provenance (SNP) and synthetic image detection (SID). Moreover, our well-trained Editprint can be readily incorporated with prior data through a fine-tuning strategy, so as to better fit various downstream tasks with significantly reduced training cost. We hope that Editprint would inspire the forensic community and serve as a novel benchmark for self-supervised general forensics.

<p align='center'>  
  <img src='https://github.com/HighwayWu/Editprint/blob/main/imgs/overview.jpg' width='850'/>
</p>
<p align='center'>  
  <em>Leveraging a self-augmentation strategy, Editprint can effectively learn over 10^7 in- and out-camera traces with a minimum of 10 training data, enabling its generalization across various zero-shot forensic tasks.</em>
</p>

## Dependency
- torch 1.9.0
- clip 1.0
- rawpy 0.18.1
- scikit-learn 1.2.1

## Zero-shot Testing
A demo of Social Network Provenance (SNP) in open verification and close classification:
```bash
python main.py --test
```
Then, Editprint will output:
```bash
FODB_6osn in Open Verification: AUC 0.9786
FODB_6osn in Close Classification: PRC 0.9667, RCL 0.9583, F1 0.9577
```

**Note: The pretrained weights can be downloaded from [Google Drive](https://www.google.com).**
