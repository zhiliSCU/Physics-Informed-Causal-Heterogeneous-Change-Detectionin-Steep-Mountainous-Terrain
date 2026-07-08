# PhyCDNet: Physics-Informed Causal Heterogeneous Change Detection for Geological Disaster Assessment in Steep Mountainous Terrain

This repository provides the official implementation of:

**Physics-Informed Causal Heterogeneous Change Detection for Geological Disaster Assessment in Steep Mountainous Terrain**


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


