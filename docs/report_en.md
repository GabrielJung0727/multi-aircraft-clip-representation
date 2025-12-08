# Multi-Dataset Aircraft Representation Learning – Technical Report

## 1. Executive Summary
This report documents the implementation of the PlanesNet + HRPlanes + FGVC Aircraft training pipeline, the obstacles encountered during development, and the effect of the added epochs/visualization suite on performance tracking. The system integrates CLIP-style representation learning, rotation-oriented augmentation, UNet auxiliary segmentation, DiT refinement, and BERT-based text prompts into a single script (`src/train_multidataset.py`).

## 2. System Overview
- **Datasets**: PlanesNet (binary satellite patches), HRPlanes (high-resolution aerial imagery with bounding boxes transformed into segmentation masks), FGVC Aircraft (fine-grained aircraft classes).
- **Architecture**: CNN backbone inspired by CLIP → DiT refinement → dataset-specific classifier heads, plus a UNet decoder fed by multi-scale features for segmentation on HRPlanes.
- **Losses**: classification (all datasets), segmentation (HRPlanes only), contrastive CLIP loss between image embeddings and cached BERT text prompts.
- **Devices**: Vision stack runs on `--vision-device`; BERT prompts cached on `--text-device` to balance CPU/GPU usage.

## 3. Development Challenges & Fixes
| Issue | Root Cause | Resolution |
| --- | --- | --- |
| CUDA assertion when training | PyTorch build lacked CUDA | Added CPU fallback instructions (`--vision-device cpu`). |
| Contrastive loss dimension mismatch | Image embeddings 512-d vs. BERT 768-d | Increased backbone projection to 768-d and passed projected vectors to DiT/classifier heads. |
| HRPlanes masks created variable metadata | Metadata included raw box lists, breaking dataloader | Removed variable-length `boxes` from batch metadata. |
| Segmentation loss size mismatch | Decoder output 32×32, masks 256×256 | Upsampled logits via bilinear interpolation before BCE loss. |
| Git ignored logs/plots | `results/` rule masked whitelists | Switched to `results/*` and added explicit `!results/logs/**`, `!results/plots/**`. |

## 4. Training Behavior (Epochs)
Using the new pipeline with CPU devices:
- Early epochs (1–3) show rapid drop in loss and increased validation accuracy across datasets; PlanesNet peaked around epoch 3.
- Additional epochs (4–5+) on PlanesNet lead to mild overfitting (validation loss rises) so early stopping around epoch 3 is recommended.
- For multi-dataset runs, keeping `--epochs` modest (e.g., 4–6) strikes a balance between convergence and runtime.

## 5. Visualization Suite
`src/visualize.py` now produces the six required plots using the latest log in `results/logs/`:
1. **Bar Plot**: PlanesNet class distribution.
2. **Line Plot**: Train vs. validation accuracy per dataset across epochs.
3. **Box Plot**: Training loss distribution by dataset.
4. **Heatmap**: Validation accuracy correlation matrix.
5. **Scatter Plot**: Train vs. validation accuracy relationship.
6. **Histogram**: CLIP loss distribution per dataset.

## 6. Conclusions & Next Steps
- Integrated pipeline is ready for longer multi-dataset experiments once GPU-enabled PyTorch is available.
- Visualization workflow (train → `python src/visualize.py`) keeps progress transparent.
- Future work: incorporate actual CLIP weights, tune UNet decoder depth, and add automated early stopping hooks.
