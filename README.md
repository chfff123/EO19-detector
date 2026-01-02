# EO19: Family-Level, Life-Stage-Aware Insect Detection Dataset for Agricultural Pest Monitoring

**EO19** is an insect detection dataset designed for agricultural pest monitoring.  
It organizes categories at the **family** level (Latin names), and introduces a **life-stage-aware labeling rule**: for holometabolous families, **Adult** and **Larva** are annotated as **two separate categories** to reduce intra-class appearance conflict in training.

> TODO: Add paper PDF link / project page / dataset download link  
> Paper: **EO19: A Family-Level, Life-Stage-Aware Insect Detection Dataset for Agricultural Pest Monitoring** (Jan 2026)  
> Authors: TODO (Your author list)  
> Affiliation: Macau University of Science and Technology (Course Final Project Report)

---

## Contents
- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Taxonomy & Label Space](#taxonomy--label-space)
- [Image Collection](#image-collection)
- [Rename Rule](#rename-rule)
- [Annotations & Formats](#annotations--formats)
- [Evaluation Protocol](#evaluation-protocol)
- [Baseline Results](#baseline-results)
- [Reproducibility](#reproducibility)
- [Download](#download)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

---

## Introduction
Insect pests can significantly reduce crop yields; building reliable vision models depends heavily on **high-quality datasets**. Existing pest datasets often suffer from limited species coverage, insufficient sample sizes, or taxonomy inaccuracies. EO19 targets these issues by combining **taxonomy-aware category design** and **expert-verified data curation**.  
(See the paper for full motivation and background.) :contentReference[oaicite:4]{index=4}

---

## Dataset Overview
- **Scope:** Insecta → **8 orders**, **19 families** (taxonomy-based), **30 categories** (life-stage-aware for part of families) :contentReference[oaicite:5]{index=5}  
- **Images:** **24,626** total :contentReference[oaicite:6]{index=6}  
- **Holometabolous split:** **11 families** are holometabolous and are split into **Adult / Larva** (→ 22 categories). The remaining families keep one category each. :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}  
- **Image sources:** real habitat scenes, wild high-res shots, specimen photos, plus a very small number of AI-generated images (≈5). :contentReference[oaicite:9]{index=9}

---

## Taxonomy & Label Space
### Why “Family” (科) and Latin names?
EO19 uses **family** as the practical recognition unit because pesticide selection often correlates well with family-level identification, while Latin names provide strict one-to-one mapping and reduce ambiguity. :contentReference[oaicite:10]{index=10}

### Life-stage-aware labeling
- **Holometabolous (完全变态):** egg–larva–pupa–adult. Larva and adult differ greatly in morphology and ecological niche, so EO19 splits them into separate categories (e.g., `Crambidae_Adult` and `Crambidae_Larva`). :contentReference[oaicite:11]{index=11}  
- **Hemimetabolous (不完全变态):** nymph and adult are morphologically similar, so EO19 keeps them in one category (e.g., `Fulgoridae`). :contentReference[oaicite:12]{index=12}

---

## Image Collection
EO19 images come from:
1) **Existing datasets** (IP102 as a major source) with multi-round filtering (low-res removal, watermark/occlusion removal, mislabel removal, expert screening). :contentReference[oaicite:13]{index=13}  
2) **New web collection** via Latin/English/Chinese/common names + species names, crawler-based pre-collection, then the same screening strategy; about **14,000** images collected by this route. :contentReference[oaicite:14]{index=14}

---

## Rename Rule
To make category membership explicit in filenames, EO19 applies a unified renaming rule: :contentReference[oaicite:15]{index=15}
- **Holometabolous families:** `A_<FamilyLatin>_<Index>.jpg` for Adult, `L_<FamilyLatin>_<Index>.jpg` for Larva  
  - Example: `A_Cerambycidae_0001.jpg`, `L_Cerambycidae_0001.jpg`
- **Hemimetabolous families:** `<FamilyLatin>_<Index>.jpg`  
  - Example: `Acrididae_0008.jpg`

---

## Annotations & Formats
- Annotation tool: **LabelImg**, labeled by four annotators with category-specific biology prep; all boxes tightly cover the entire object; difficult/incorrect samples are replaced; labels are **expert-verified**. :contentReference[oaicite:16]{index=16}  
- Primary label format: **Pascal VOC XML** (per-image). :contentReference[oaicite:17]{index=17}  
- Export formats: converted via Python scripts into:
  - **YOLO** (`.txt`)
  - **COCO** (`.json`) :contentReference[oaicite:18]{index=18}

---

## Evaluation Protocol
EO19 reports standard COCO-style metrics (AP, AP50, AP75, AP_S/M/L).  
Baseline experiments run under unified protocols with repeated trials to ensure reproducibility. :contentReference[oaicite:19]{index=19}

---

## Baseline Results
Below are representative baselines evaluated on EO19 (mean over 3 runs when available).  
> **Note:** “AP” here refers to COCO-style AP (AP@[.50:.95]). :contentReference[oaicite:20]{index=20}

| Model | Backbone / Setting | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---|---|---:|---:|---:|---:|---:|---:|
| Co-DINO | ViT-Large, 5-scale | 0.733* | 0.930* | 0.801* | 0.416* | 0.590* | 0.823* |
| RT-DETR | PResNet-18 | 0.666 | 0.866 | 0.729 | 0.277 | 0.492 | 0.770 |
| Co-DETR | ResNet-50 | 0.611 | 0.814 | 0.668 | 0.218 | 0.408 | 0.720 |
| DEIM v1 | HGNetv2 | 0.692 | 0.887 | 0.754 | 0.309 | 0.545 | 0.794 |
| D-FINE (L) | HGNetv2 | 0.701 | 0.900 | 0.762 | 0.321 | 0.534 | 0.799 |
| D-FINE (M) | HGNetv2 | 0.693 | 0.885 | 0.756 | 0.282 | 0.524 | 0.792 |

\* Co-DINO表中给出了第1次实验的完整指标，其余轮次在论文表格中汇报。 :contentReference[oaicite:21]{index=21} :contentReference[oaicite:22]{index=22} :contentReference[oaicite:23]{index=23} :contentReference[oaicite:24]{index=24} :contentReference[oaicite:25]{index=25} :contentReference[oaicite:26]{index=26}

---

## Reproducibility
This project evaluated multiple frameworks under controlled environments. Example (MMDetection stack used in Co-DETR experiments):  
- Python 3.7.12, PyTorch 1.11.0, TorchVision 0.12.0  
- CUDA 11.3, cuDNN 8.2  
- OpenCV 4.11.0, MMCV 1.5.0, MMDetection 2.25.3 :contentReference[oaicite:27]{index=27}

> TODO: Add your exact training commands / configs / checkpoints download links.

---

## Download
> TODO: Provide dataset download link(s) and checksum(s).

Recommended structure (COCO format example):
