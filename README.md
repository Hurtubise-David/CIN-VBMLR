# CIN-VBMLR: Cinema Variational Bayesian Multinomial Logistic Regression

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CINE--VBMLR%20Dataset-blue)](https://huggingface.co/datasets/Hurtubisedavid/CIN-VBMLR)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**CIN-VBMLR** is the official Python implementation of the **Variational Bayesian Multinomial Logistic Regression (VBMLR)** method adapted for **Cinema Camera Tracking** and **Calibration**.

This project adapts the probabilistic framework originally developed for gaze estimation (VBMLR) to the problem of **Depth from Defocus (DfD)** in a cinematic context. By leveraging Bayesian inference, we jointly estimate:
* Camera Pose (Extrinsics)
* Intrinsic Parameters (Focal Length, Focus Distance)
* Blur Kernels (PSF/CoC)

This approach is designed to handle the high-resolution, shallow depth-of-field imagery typical of cinema lenses (e.g., ARRI Signature Primes), where traditional pinhole calibration fails.

## Dataset

This repository is paired with the **CINE-VBMLR Dataset**, a large-scale calibration dataset captured at **MELS Studios** with ARRI cameras and precise lens metadata.

**[Download the Dataset on Hugging Face](https://huggingface.co/datasets/Hurtubisedavid/CIN-VBMLR)**

To load the dataset in your Python script:

```python
from datasets import load_dataset

# Load the CINE-VBMLR dataset (Linear EXR images + Metadata)
dataset = load_dataset("Hurtubisedavid/CIN-VBMLR")

print(dataset['train'][0])
