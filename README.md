# CIN-VBMLR: Cinema Variational Bayesian Multinomial Logistic Regression

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CINE--VBMLR%20Dataset-blue)](https://huggingface.co/datasets/Hurtubisedavid/CIN-VBMLR)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

**CIN-VBMLR** is the official Python implementation of the **Variational Bayesian Multinomial Logistic Regression (VBMLR)** method adapted for **Cinema Camera Tracking** and **Calibration**.

This project adapts the probabilistic framework originally developed for gaze estimation (VBMLR) to the problem of **Depth from Defocus (DfD)** in a cinematic context. By leveraging Bayesian inference, we jointly estimate:
* Camera Pose (Extrinsics)
* Intrinsic Parameters (Focal Length, Focus Distance)
* Blur Kernels (PSF/CoC)

This approach is designed to handle the high-resolution, shallow depth-of-field imagery typical of cinema lenses (e.g., ARRI Signature Primes), where traditional pinhole calibration fails.

## Dataset

This repository is paired with the **CIN-VBMLR Dataset**, a large-scale calibration dataset captured at **MELS Studios** with ARRI cameras and precise lens metadata.

**[Download the Dataset on Hugging Face](https://huggingface.co/datasets/Hurtubisedavid/CIN-VBMLR)**

To load the dataset in your Python script:

```python
from datasets import load_dataset

# Load the CINE-VBMLR dataset (Linear EXR images + Metadata)
dataset = load_dataset("Hurtubisedavid/CIN-VBMLR")

print(dataset['train'][0])
```

## Third-Party Components & Licensing
This pipeline integrates Video Depth Anything (VDA) (https://github.com/DepthAnything/Video-Depth-Anything) to provide dense temporal depth priors.

* Real-Time Adaptation: We have modified video_depth_stream to support low-latency real-time video streaming, enabling its use in live virtual production workflows.
* Default Configuration: This repository is configured to use the VDA-Small (vits) model by default.
* Licensing: The VDA-Small model and code are licensed under Apache-2.0, which is fully compatible with this project's license.
* Important: If you choose to manually download and use the VDA-Base or VDA-Large checkpoints, please note that those specific weights are governed by a CC-BY-NC-4.0 license (Non-Commercial).

## Theory & References

The core algorithm is based on **Variational Bayesian Multinomial Logistic Regression**. If you use this code or dataset, please cite the following foundational papers:

### 1. Core Method (CINE-VBMLR)
* **Hurtubise-Martin, D.**, et al. *Real-time eye gaze estimation on a computer screen*. (Manuscript in preparation).
  * *Foundation of the VBMLR probabilistic model adapted here for cinema.*

### 2. Depth from Defocus & Camera Parameters
* **Ziou, D., & Deschênes, F. (2001).** [Depth from defocus estimation in spatial domain](https://doi.org/10.1006/cviu.2000.0899). *Computer Vision and Image Understanding*, 81(2), 143-165.
* **Mannan, F., & Langer, M. S. (2015).** [Optimal camera parameters for depth from defocus](https://doi.org/10.1109/3DV.2015.44). In *2015 International Conference on 3D Vision* (pp. 326-334). IEEE.
* **LeBlanc, J. W., Thelen, B. J., & Hero, A. O. (2018).** [Joint camera blur and pose estimation from aliased data](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-35-4-639). *Journal of the Optical Society of America A*, 35(4), 639-651.
