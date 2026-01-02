# EO19: Family-Level, Life-Stage-Aware Insect Detection Dataset for Agricultural Pest Monitoring

> **Notice (Confidential / Under Review)**  
> This repository is prepared for the EO19 paper and will be **partially released**.  
> Since the paper/dataset is **not yet published**, some sensitive information (e.g., download links, full author list, checkpoints, and exact paths) is intentionally withheld.  
> Placeholders are used to mark unpublished items:
> - **TODO**: to be filled after publication (final info not ready yet)  
> - **TBA**: to be announced (will be disclosed later)  
> Please do not redistribute any private materials until the official release.

> Paper: TBA (PDF / arXiv / project page)  
> Dataset download: TODO
> Authors: TODO
> Code: TODO

---

## Contents
- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Taxonomy & Label Space](#taxonomy--label-space)
- [Image Collection](#image-collection)
- [Rename Rule](#rename-rule)
- [Annotations & Formats](#annotations--formats)
- [Requirements (Software)](#requirements-software)
- [Pretrained Models](#pretrained-models)
- [Preparation for Testing](#preparation-for-testing)
- [Model Zoo & Results (3-run Average)](#model-zoo--results-3-run-average)
  - [Co-DINO (ViT-Large, 5-scale)](#co-dino-vit-large-5-scale)
  - [RT-DETR (PResNet-18)](#rt-detr-presnet-18)
  - [Co-DETR (ResNet-50)](#co-detr-resnet-50)
  - [DEIM v1 (HGNetv2)](#deim-v1-hgnetv2)
  - [D-FINE Large (HGNetv2)](#d-fine-large-hgnetv2)
  - [D-FINE Medium (HGNetv2)](#d-fine-medium-hgnetv2)
- [Download](#download)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

---

## Introduction
EO19 is an insect detection dataset designed for agricultural pest monitoring.  
It organizes categories at the **family** level (Latin names), and introduces a **life-stage-aware labeling rule**: for holometabolous families, **Adult** and **Larva** are annotated as **two separate categories** to reduce intra-class appearance conflict in training.

---

## Dataset Overview
- **Scope:** Insecta → **8 orders**, **19 families**, **30 categories**
- **Images:** **24,626** total
- **Life-stage split:** **11 holometabolous families** split into **Adult / Larva** (→ 22 categories); remaining families keep one category each.
- **Sources:** habitat photos + specimen photos + a very small number of AI-generated images.

---

## Taxonomy & Label Space
### Why “Family” and Latin names?
EO19 uses family-level categories to reduce ambiguity and align with practical pest monitoring decisions.

### Life-stage-aware labeling
- **Holometabolous:** egg–larva–pupa–adult → split `Family_Adult` and `Family_Larva`
- **Hemimetabolous:** nymph and adult are similar → keep one category

---

## Image Collection
EO19 images come from:
1) Existing datasets (screened and cleaned)  
2) Web collection via Latin/English/Chinese/common names + species names, followed by the same screening strategy.

---

## Rename Rule
Unified filename rule:
- Holometabolous families:
  - Adult: `A_<FamilyLatin>_<Index>.jpg`
  - Larva: `L_<FamilyLatin>_<Index>.jpg`
- Hemimetabolous families:
  - `<FamilyLatin>_<Index>.jpg`

Examples:
- `A_Cerambycidae_0001.jpg`
- `L_Cerambycidae_0001.jpg`
- `Acrididae_0008.jpg`

---

## Annotations & Formats
- Annotation tool: LabelImg
- Bounding boxes: tightly cover entire insect body
- Primary format: Pascal VOC XML
- Export formats: YOLO (`.txt`), COCO (`.json`)

---

## Requirements (Software)

> Notes:
> - Different baselines may use different original frameworks; pin dependencies per model.

| Model | Python | PyTorch / TorchVision | CUDA / cuDNN | Framework / Key deps |
|---|---|---|---|---|
| Co-DINO (ViT-L, 5-scale) | 3.7.12 | 1.11.0 / 0.12.0 | 11.3 / 8.2 | OpenCV 4.11.0, MMCV 1.5.0, MMDetection 2.25.3 |
| Co-DETR (R50) | 3.7.12 | 1.11.0 / 0.12.0 | 11.3 / 8.2 | OpenCV 4.11.0, MMCV 1.5.0, MMDetection 2.25.3 |
| RT-DETR (PResNet-18) | TODO | 2.4.1 | 12.1 / 9.1.2 | Project native deps |
| DEIM v1 (HGNetv2) | TODO | 2.0.1 / 0.15.2 | TODO | faster-coco-eval 1.6.5, PyYAML, TensorBoard, SciPy, calflops, Transformers |
| D-FINE (M/L) (HGNetv2) | TODO | TODO | TODO | Pin versions in your repo |

---

## Pretrained Models

Recommended layout:
```text
pretrained/
  codino_vitl_5scale.pth
  codetr_r50.pth
  rtdetr_r18.pth
  deim_v1_hgnetv2.pth
  dfine_m_hgnetv2.pth
  dfine_l_hgnetv2.pth
```

| Model | Checkpoint | Download | SHA256 |
|---|---|---|---|
| Co-DINO (ViT-L, 5-scale) | `pretrained/codino_vitl_5scale.pth` | TODO | TODO |
| RT-DETR (PResNet-18) | `pretrained/rtdetr_r18.pth` | TODO | TODO |
| Co-DETR (R50) | `pretrained/codetr_r50.pth` | TODO | TODO |
| DEIM v1 (HGNetv2) | `pretrained/deim_v1_hgnetv2.pth` | TODO | TODO |
| D-FINE Medium (HGNetv2) | `pretrained/dfine_m_hgnetv2.pth` | TODO | TODO |
| D-FINE Large (HGNetv2) | `pretrained/dfine_l_hgnetv2.pth` | TODO | TODO |

---

## Preparation for Testing

### 1) Dataset structure (COCO format)
```text
EO19/
  images/
    train/
    val/
    test/
  annotations/
    eo19_train.json
    eo19_val.json
    eo19_test.json
```

### 2) Ensure class count matches
EO19 uses **30 categories**. Make sure `num_classes = 30` in your model config.

### 3) Point configs to EO19 paths
Examples (edit to your framework):
- `data_root = /path/to/EO19`
- `ann_file = annotations/eo19_val.json`
- `img_prefix = images/val/`

### 4) Run evaluation (fill in exact commands)
Co-DINO
```bash
python tools/train.py projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py --work-dir /home/apulis-dev/code/answerCoDINO --launcher none --cfg-options load_from=/home/apulis-dev/code/pytorch_model.pth data.samples_per_gpu=1 model.backbone.img_size=1024
```

Co-DETR :
```bash
python tools/train.py projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py --launcher none
```

RT-DETR:
```bash
python -u tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_sp3_120e_coco.yml --output-dir /home/apulis-dev/code/answer -u print_freq=10 2>&1 | tee -a /home/apulis-dev/code/rtdetr-answer-twice/train.log
```

DEIM
```bash
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/deim_dfine/deim_hgnetv2_m_coco.yml --use-amp --seed=0 > output2.log 2>&1
```

D-FINE
```bash
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/dfine/dfine_hgnetv2_l_coco.yml --use-amp --seed=0
```

> Replace the above with the exact scripts/arguments used in your repos.

---

## Model Zoo & Results (3-run Average)

All results are COCO-style metrics (AP, AP50, AP75, AP_S/M/L) on the same EO19 validation split.

### Leaderboard (avg of 3 runs)
| Model | Backbone | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---|---|---:|---:|---:|---:|---:|---:|
| D-FINE Large | HGNetv2 | 0.701 | 0.900 | 0.762 | 0.321 | 0.534 | 0.799 |
| D-FINE Medium | HGNetv2 | 0.693 | 0.885 | 0.756 | 0.282 | 0.524 | 0.792 |
| DEIM v1 | HGNetv2 | 0.692 | 0.887 | 0.754 | 0.309 | 0.545 | 0.794 |
| RT-DETR | PResNet-18 | 0.666 | 0.866 | 0.729 | 0.277 | 0.492 | 0.770 |
| Co-DETR | ResNet-50 | 0.611 | 0.814 | 0.668 | 0.218 | 0.408 | 0.720 |
| Co-DINO | ViT-L (5-scale) | TBD | TBD | TBD | TBD | TBD | TBD |

---

## Co-DINO (ViT-Large, 5-scale)

| Run | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.733 | 0.930 | 0.801 | 0.416 | 0.590 | 0.823 |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD | TBD |
| **avg** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

---

## RT-DETR (PResNet-18)

| Run | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.662 | 0.858 | 0.723 | 0.275 | 0.480 | 0.770 |
| 2 | 0.668 | 0.870 | 0.727 | 0.290 | 0.490 | 0.768 |
| 3 | 0.668 | 0.869 | 0.736 | 0.266 | 0.505 | 0.772 |
| **avg** | **0.666** | **0.866** | **0.729** | **0.277** | **0.492** | **0.770** |

---

## Co-DETR (ResNet-50)

| Run | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.612 | 0.815 | 0.667 | 0.226 | 0.411 | 0.718 |
| 2 | 0.617 | 0.819 | 0.678 | 0.235 | 0.380 | 0.727 |
| 3 | 0.605 | 0.807 | 0.660 | 0.194 | 0.433 | 0.714 |
| **avg** | **0.611** | **0.814** | **0.668** | **0.218** | **0.408** | **0.720** |

---

## DEIM v1 (HGNetv2)

| Run | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.692 | 0.889 | 0.755 | 0.297 | 0.542 | 0.794 |
| 2 | 0.691 | 0.884 | 0.752 | 0.305 | 0.549 | 0.793 |
| 3 | 0.692 | 0.888 | 0.755 | 0.325 | 0.545 | 0.795 |
| **avg** | **0.692** | **0.887** | **0.754** | **0.309** | **0.545** | **0.794** |

---

## D-FINE Large (HGNetv2)

| Run | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.701 | 0.899 | 0.763 | 0.331 | 0.528 | 0.798 |
| 2 | 0.697 | 0.894 | 0.758 | 0.309 | 0.544 | 0.797 |
| 3 | 0.704 | 0.906 | 0.765 | 0.323 | 0.529 | 0.801 |
| **avg** | **0.701** | **0.900** | **0.762** | **0.321** | **0.534** | **0.799** |

---

## D-FINE Medium (HGNetv2)

| Run | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.691 | 0.887 | 0.755 | 0.292 | 0.538 | 0.791 |
| 2 | 0.698 | 0.883 | 0.756 | 0.279 | 0.513 | 0.794 |
| 3 | 0.690 | 0.885 | 0.756 | 0.276 | 0.521 | 0.792 |
| **avg** | **0.693** | **0.885** | **0.756** | **0.282** | **0.524** | **0.792** |

---
## Gradcam

## Gradcam
<p>
  <img src="https://github.com/user-attachments/assets/720eb01a-7370-4afa-9ae0-c3a35d7ef3ee" width="45%" />
  <img src="https://github.com/user-attachments/assets/b9fae2d4-4bfa-493c-b650-876a3388758c" width="45%" />
</p>

<p>
  <img src="https://github.com/user-attachments/assets/2b812fb6-4a1c-42bf-a0d2-773c49f43738" width="45%" />
  <img src="https://github.com/user-attachments/assets/36e18aea-68d2-4766-ae48-913e472e667e" width="45%" />
</p>

<p>
  <img src="https://github.com/user-attachments/assets/5f952303-8036-4663-99d1-9216aacb0b6e" width="45%" />
  <img src="https://github.com/user-attachments/assets/c667cab8-13a1-42d5-959a-7d30dea0911e" width="45%" />
</p>



---

## Download
- Dataset: TODO
- Checksums: TODO

---

## Citation
```bibtex
@article{EO19_2026,
  title   = {EO19: A Family-Level, Life-Stage-Aware Insect Detection Dataset for Agricultural Pest Monitoring},
  author  = {TODO},
  journal = {Technical Report / Course Final Project Report},
  year    = {2026},
  note    = {Macau University of Science and Technology},
  url     = {TODO}
}
```

---

## License
- Paper/text: TODO
- Dataset: TODO (e.g., CC BY-NC 4.0 / research-only)
- Code: TODO (e.g., MIT)

---

## Contact
- GitHub: https://github.com/chfff123
- Email: TODO

---

## Acknowledgements
- IP102 dataset (screened and cleaned as a major image source)
- Thanks to agriculture experts for verification and label auditing
