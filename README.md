# Editprint

An official implementation code for paper "Editprint: General Digital Image Forensics via Editing Fingerprint with Self-Augmentation Training"

<p align='center'>  
  <img src='https://github.com/HighwayWu/Editprint/blob/main/imgs/overview.jpg' width='850'/>
</p>
<p align='center'>  
  <em>Leveraging a self-augmentation strategy, Editprint effectively learns over 10^7 in- and out-camera traces with only 10 samples, supporting generalization across zero-shot forensic tasks.</em>
</p>

## Table of Contents

- [Background](#background)
- [Dependency](#dependency)
- [Usage](#usage)
- [Citation](#citation)


## Background
Digital image forensics can ensure information credibility in tasks like camera source identification (CSI), synthetic image detection (SID), and social network provenance (SNP). These tasks typically rely on image processing history clues left by in-camera operations, post-capture editing, or synthetic generation. However, most existing forensic methods have obvious limitations: 1) they often only focus on camera-specific traces (e.g., the well-known PRNU), and 2) they demand a substantial amount of annotated training data. To address these constraints, we propose Editprint, a novel general forensic feature that captures highly diverse in- and out-camera processing history clues with minimal unlabeled training data. Ideally, we expect that any images undergoing the same imaging, editing, and transmission processes would yield identical Editprints, and vice versa. To model the in- and out-camera operations, we devise an online editing pool based on self-augmentation strategies. Requiring only minimal (e.g., 10) training data, the editing pool can simulate massive (e.g., 10^7) editing chains and traces arising from the in-camera processing and the subsequent out-camera operations. To ensure that Editprint exhibits high discriminative capabilities across various editing chains, we propose using textual descriptions of these chains as labels and supervising their Editprints through language-guided contrastive learning. Extensive experiments show Editprint outperforms existing self-supervised forensics, particularly in non-camera applications such as SNP and SID. We hope that Editprint would inspire the forensic community and serve as a novel benchmark for self-supervised forensics.


## Dependency
- torch 1.9.0
- clip 1.0
- rawpy 0.18.1
- scikit-learn 1.2.1


## Usage

For training:
```bash
sh main.sh
```

For testing:
```bash
python main.py --test
```
Editprint will conduct Social Network Provenance (SNP) in open verification and close classification scenario over the images in the `data/FODB/`, and output:
```bash
FODB_6osn in Open Verification: AUC 0.9786
FODB_6osn in Close Classification: PRC 0.9667, RCL 0.9583, F1 0.9577
```

For prepare the train/test files:
```bash
python preprocess.py
```

**Note: The pretrained Editprint and demo data can be downloaded from [Google Drive](https://www.google.com) or [Baidu Pan](https://www.baidu.com).**


## Citation

If you use this code for your research, please cite the reference:
```
@inproceedings{editprint,
  title={Editprint: General Digital Image Forensics via Editing Fingerprint with Self-Augmentation Training},
  author={H. Wu and K. Li and Y. Li and J. Zhou},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},  
  year={2026},
}
```

