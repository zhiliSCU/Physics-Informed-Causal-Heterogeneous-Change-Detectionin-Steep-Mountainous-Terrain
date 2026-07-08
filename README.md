# PhyCDNet: Physics-Guided Causal Heterogeneous Change Detection Network

This repository provides the official implementation of:

**A Physics-Guided Causal Framework for Cross-Sensor Heterogeneous Change Detection in Remote Sensing Images**

---

## Overview

Cross-sensor heterogeneous change detection in remote sensing faces a **triple coupling problem**:

1. **Radiometric inconsistency**: Sensor-specific spectral response functions and atmospheric conditions produce systematic radiometric shifts between pre- and post-disaster images.
2. **Geometric misalignment**: Cross-resolution acquisition geometry introduces spatial misregistration that traditional two-stage pipelines cannot fully correct.
3. **Terrain-induced confounding**: Mountain shadows, slope-aspect variations, and cast-shadow regions produce sensor-dependent false alarms indistinguishable from genuine change in feature space.

Existing approaches treat these as independent preprocessing steps—radiometric normalization, geometric registration, change detection—which propagate errors irreversibly through the pipeline.

To address this, we propose **PhyCDNet**, a unified end-to-end framework that integrates physics-grounded radiometric modeling, causal terrain-change decoupling, and terrain-adaptive geometric alignment within a single differentiable architecture.

---

## Core Components

### 1️⃣ CRMA — Cross-sensor Radiometric Modulation Alignment

### 2️⃣ PCOD — Physics-Causal Object Decoupling

### 3️⃣ CFDA — Cross-sensor Feature Distribution Alignment

### 4️⃣ Three-Phase Dynamic Loss Scheduling


### Training

```bash
python main_cd.py \
    --data_name YOUR_DATASET \
    --net_G PhyCDNet \
    --max_epochs 200 \
    --batch_size 16 \
    --lr 0.0001 \
    --checkpoint_root /path/to/checkpoints \
    --vis_root /path/to/visualizations
```

### Evaluation

```bash
python eval_cd.py \
    --data_name YOUR_DATASET \
    --net_G PhyCDNet \
    --checkpoints_root /path/to/checkpoints \
    --project_name YOUR_PROJECT \
    --split test
```

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- torchvision ≥ 0.13
- numpy, opencv-python, matplotlib
- kornia (differentiable image warping)
- scikit-learn (t-SNE)
- tqdm (progress bars)

---


