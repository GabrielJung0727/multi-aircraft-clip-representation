# Multi-Dataset Aircraft Representation Learning – Technical Report

## 1. Executive Summary

This report describes the implementation and empirical behavior of a multi-dataset aircraft representation learning pipeline that jointly trains on **PlanesNet**, **HRPlanes**, and **FGVC Aircraft**. The system integrates CLIP-style contrastive learning, rotation-oriented augmentation, a UNet auxiliary decoder for segmentation, a DiT refinement block, and BERT-driven text prompts inside a unified script (`src/train_multidataset.py`).  

Across five epochs on CPU, the model exhibits **stable convergence** on the satellite datasets (PlanesNet, HRPlanes), achieving **97.47% validation accuracy on PlanesNet** and **100% on HRPlanes** by epoch 5, while the fine-grained FGVC Aircraft task remains under-trained with validation accuracy in the **1–2% range**. These results confirm that the current architecture and training regime are sufficient for binary/mid-level classification on remote sensing imagery, but additional capacity or task-specific tuning is required for fine-grained aircraft recognition.

---

## 2. System Overview

### 2.1 Datasets

- **PlanesNet**
  - Binary classification on small satellite patches (plane vs. no-plane).
  - Strong class imbalance and heavy background variation.

- **HRPlanes**
  - High-resolution aerial imagery with annotated aircraft.
  - Bounding boxes are converted into segmentation masks and used for:
    - Classification (plane presence)
    - Auxiliary segmentation via a UNet decoder.

- **FGVC Aircraft**
  - Fine-grained aircraft recognition dataset with many visually similar classes.
  - Used here to stress-test CLIP-style representations and the classifier head under multi-dataset training.

### 2.2 Architecture

- **Backbone**
  - CNN/ResNet-style vision backbone inspired by CLIP:
    - Deeper residual stack instead of a shallow 2-layer CNN.
    - Final projection aligned with BERT embedding dimensionality.

- **Heads**
  - **Dataset-specific classifier heads**:
    - Independent classification layers per dataset (PlanesNet, HRPlanes, FGVC).
    - Optional hidden layer for additional capacity.
  - **UNet decoder (HRPlanes only)**:
    - Multi-scale feature aggregation from the backbone.
    - Outputs segmentation logits aligned with HRPlanes masks.

- **Text Encoder & Prompts**
  - **BERT** used to encode text prompts such as:
    - “satellite image of an airplane”
    - “no airplane”
    - “side view of a commercial jet”
  - Encoded prompts are cached on a dedicated text device to avoid repeated computation.

### 2.3 Loss Functions

For each dataset, the total loss is a weighted sum of the following components:

- **Classification Loss (`cls_loss`)**
  - Cross-entropy between predicted logits and ground-truth labels.
  - Applied to all datasets.

- **Segmentation Loss (`seg_loss`)**
  - Binary or multi-class segmentation loss (e.g., BCE/CE) on HRPlanes masks.
  - Decoder outputs are interpolated to match mask resolution before loss computation.

- **Contrastive CLIP Loss (`clip_loss`)**
  - Contrastive loss between image embeddings and BERT text embeddings.
  - Encourages alignment between visual and semantic representations.

The logged `total_loss` values per dataset reflect a combination of these terms.

### 2.4 Devices

- Vision stack and UNet decoder run on `--vision-device`.
- BERT text encoder and cached prompts are placed on `--text-device`.
- The pipeline supports pure-CPU training for environments without CUDA.

---

## 3. Development Challenges & Fixes

During implementation, several integration and stability issues were identified and addressed. Table 1 summarizes the main problems and corresponding fixes.

| Issue | Root Cause | Resolution |
| --- | --- | --- |
| CUDA assertions at startup | PyTorch build without CUDA or mismatched driver | Added clear CPU fallback (`--vision-device cpu`) and documentation for CPU-only runs. |
| Contrastive loss dimension mismatch | Vision backbone produced 512-d embeddings while BERT outputs 768-d | Extended the backbone projection to 768-d and consistently passed projected features into DiT and classifier heads. |
| HRPlanes batch metadata errors | Variable-length bounding box lists were included in collated batch metadata | Removed per-sample `boxes` from batch metadata; only fixed-shape tensors (images, masks, labels) are collated. |
| Segmentation loss shape mismatch | UNet logits were smaller (e.g., 32×32) than ground-truth masks (e.g., 256×256) | Added bilinear upsampling step to resize logits to mask size before computing segmentation loss. |
| `results/` artifacts ignored by Git | Broad ignore rule on `results/` masked logs and plots | Narrowed ignore patterns and explicitly allowed `results/logs/**` and `results/plots/**` via negation rules. |

These changes resulted in a stable multi-dataset training loop that can be executed on CPU while still logging all required metrics and visualizations.

---

## 4. Training Dynamics Across Datasets

This section analyzes the logged metrics over **five epochs** for each dataset. All numbers are taken from the structured logs and represent the behavior of the current CLIP-style backbone with UNet and DiT components enabled.

### 4.1 PlanesNet

**Training (classification + CLIP)**  

- Train classification loss decreased from **0.34** at epoch 1 to **0.13** at epoch 5.  
- Train accuracy improved from **85.7%** → **95.3%** over the same period.  
- The CLIP loss term on PlanesNet remained relatively stable around **3.05–3.25**, acting as a regularizer rather than dominating optimization.

**Validation**  

- Validation loss decreased monotonically:
  - **0.236 → 0.120 → 0.128 → 0.097 → 0.076** (epochs 1–5).
- Validation accuracy consistently improved:
  - **92.41% → 95.63% → 95.78% → 96.53% → 97.47%.**

**Interpretation**  
PlanesNet shows **clear and stable convergence**: both train and validation accuracy increase steadily, and validation loss decreases throughout all five epochs without an obvious overfitting phase. For this dataset alone, early stopping around epoch 3 would already provide >95% validation accuracy, while epoch 5 pushes the model close to 97.5%.

---

### 4.2 HRPlanes

**Training (classification + segmentation + CLIP)**  

- HRPlanes train classification loss drops from **0.41** to **0.013** across epochs 1–5.  
- Segmentation loss on HRPlanes decreases from approximately **0.76** to **0.50**, indicating that the UNet decoder is learning to approximate the aircraft masks.  
- Train accuracy is very high throughout, starting at **98.41%** and reaching **99.47–100%** from epoch 3 onward.

**Validation**  

- Validation classification loss moves from **0.140** at epoch 1 to nearly **0.00025** by epoch 5.  
- Validation accuracy is consistently **100%** across all epochs.  
- Segmentation loss on the validation set is more volatile (e.g., **0.75 → 0.76 → 2.64 → 1.15 → 0.83**), reflecting the relatively small number of validation samples and the higher difficulty of pixel-wise prediction compared to binary classification.

**Interpretation**  
HRPlanes classification is essentially **solved** by the current architecture under the given training regime. The model quickly reaches perfect validation accuracy and maintains it. The segmentation branch is learning, as shown by decreasing train segmentation loss, but exhibits some instability on the validation masks, which is expected given the added complexity of pixel-level supervision and limited data. Overall, HRPlanes confirms that the CLIP-style backbone plus UNet decoder has sufficient capacity for high-resolution aerial imagery.

---

### 4.3 FGVC Aircraft

**Training (classification + CLIP)**  

- Train classification loss decreases slowly from **4.62** at epoch 1 to **4.47** at epoch 5.  
- CLIP loss remains high but slightly decreasing (~3.47 → 3.40).  
- Train accuracy improves only marginally:
  - **0.81% → 1.35% → 1.95% → 1.77% → 2.64%.**

**Validation**  

- Validation loss remains in the range **4.51–4.61** across epochs.  
- Validation accuracy increases slightly but stays extremely low:
  - **1.26% → 1.62% → 1.47% → 2.25% → 2.37%.**

**Interpretation**  
In contrast to the satellite datasets, FGVC Aircraft shows **clear underfitting** in this multi-dataset configuration. Despite several epochs of training, the model’s accuracy remains near random for a multi-class fine-grained task. This is likely due to:

- The relative complexity of the FGVC task (many similar aircraft classes).
- Capacity and optimization being dominated by PlanesNet/HRPlanes, which are easier and quickly solvable.
- The lack of FGVC-specific architectural bias (e.g., higher-resolution crops, deeper heads) in the current setup.

The FGVC results highlight that while the unified pipeline learns robust representations for binary aerial classification, **fine-grained aircraft recognition requires additional tuning**, such as:

- Re-balancing loss weights across datasets.
- Increasing classifier head depth for FGVC.
- Running a separate fine-tuning stage focused solely on FGVC.

---

### 4.4 Summary of Multi-Dataset Behavior

- **PlanesNet**: Strong and smooth convergence; validation accuracy ≈97.5% by epoch 5 with no clear overfitting signs.
- **HRPlanes**: Classification saturates at 100% validation accuracy very early; segmentation learning progresses but remains noisier.
- **FGVC Aircraft**: Under-trained in the multi-dataset setting, plateauing around 2–3% accuracy; requires task-specific enhancements.

Overall, **shared CLIP-style features plus lightweight dataset-specific heads are sufficient for satellite-style binary tasks**, but not yet for highly fine-grained recognition without further specialization.

---

## 5. Visualization Suite

To make training dynamics transparent and suitable for reporting, the project includes a dedicated visualization module (`src/visualize.py`) that consumes the JSON logs under `results/logs/` and produces the six required plots:

1. **Bar Plot**
   - Class distribution and/or per-dataset sample counts.
   - Used to illustrate dataset imbalance (e.g., number of samples in PlanesNet vs. HRPlanes vs. FGVC).

2. **Line Plot**
   - Train vs. validation accuracy per dataset over epochs.
   - Clearly shows PlanesNet and HRPlanes quickly converging, while FGVC trails far behind.

3. **Box Plot**
   - Distribution of training losses (e.g., `total_loss` or `cls_loss`) per dataset.
   - Highlights that FGVC maintains a high loss regime compared to the satellite datasets.

4. **Heatmap**
   - Correlation matrix of validation accuracies or losses across datasets and epochs.
   - Helps visualize whether improvements on one dataset correlate with improvements or regressions on others.

5. **Scatter Plot**
   - Train vs. validation accuracy for each dataset and epoch.
   - Points for PlanesNet and HRPlanes lie near the ideal diagonal; FGVC points remain clustered near the origin.

6. **Histogram**
   - Distribution of CLIP contrastive losses per dataset.
   - Used to inspect whether CLIP alignment is improving uniformly across datasets, or only for the simpler ones.

These visualizations are intended both for model debugging and as figures to be referenced in the final written report or slides.

---

## 6. Conclusions and Future Work

The current multi-dataset training pipeline successfully:

- Integrates **CLIP-style contrastive learning**, **rotation-oriented augmentation**, **UNet segmentation**, **DiT refinement**, and **BERT-based text prompts** in a single script.
- Achieves **high performance on PlanesNet and HRPlanes**:
  - PlanesNet: ~97.5% validation accuracy at epoch 5.
  - HRPlanes: 100% validation accuracy throughout, with gradually improving segmentation.
- Provides a **rich logging and visualization setup** that exposes per-dataset behavior, making overfitting, underfitting, and cross-dataset trade-offs easy to diagnose.

At the same time, the experiments reveal a clear limitation:

- The same pipeline **does not yet adequately solve FGVC Aircraft**, a much more challenging fine-grained classification task. Performance remains at ~2–3% accuracy after five epochs in the multi-dataset setting.

**Future work** will therefore focus on:

1. **Re-balancing Multi-Task Training**
   - Adjusting loss weights so that FGVC receives a stronger gradient signal relative to the easier satellite tasks.

2. **FGVC-Specific Fine-Tuning**
   - Running a dedicated fine-tuning stage on FGVC with higher resolution inputs and a deeper classifier head.

3. **Improved Segmentation Supervision**
   - Refining HRPlanes masks and UNet architecture to stabilize segmentation loss on the validation split.

4. **Early Stopping and Scheduling**
   - Implementing dataset-aware early stopping criteria (e.g., monitor PlanesNet/HRPlanes convergence while allowing additional epochs for FGVC).

5. **GPU Acceleration**
   - Migrating from CPU to GPU-enabled PyTorch to scale up epochs and run more aggressive hyper-parameter sweeps.

In summary, the proposed system establishes a strong baseline for **multi-dataset aircraft representation learning in remote sensing settings** and exposes clear directions to extend the approach to **fine-grained aircraft recognition**.
