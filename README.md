# TECRFN: Text-enhanced Cross-modal Reinforced Fusion Network for Multimodal Sentiment Analysis

This repository contains the implementation and experimental materials for **TECRFN**, a text-guided framework for **multimodal sentiment analysis (MSA)** under the **unaligned** setting.

TECRFN is designed to address two coupled challenges in multimodal sentiment analysis:

1. **Textual shortcut bias**: spurious lexical correlations in the text stream may corrupt the textual anchor and mislead cross-modal learning.
2. **Unstable multimodal fusion**: noisy or asynchronous acoustic/visual evidence may degrade the final fused representation.

To address these issues, TECRFN follows a staged design:

- **Text Debiasing Module (TDM)** purifies the textual anchor by suppressing shortcut-prone lexical signals.
- **Text-enhanced Transformer (TET)** injects sentiment-relevant textual semantics into audio and vision.
- **Text-guided Cross-modal Reinforced Transformer (TGCRT)** performs query-adaptive dual-path fusion with similarity-biased gating, channel recalibration, and residual stabilization.
- **Memory Encoder** and **Unimodal Label Generation Module (ULGM)** preserve long-range context and unimodal discriminability.

- Code Access
The source code for TECRFN is available via Baidu Netdisk:

Link: https://pan.baidu.com/s/1qfSkGpB_BtY7M6nDpMxxxA
Password: z6xb

## Overview

Multimodal sentiment analysis aims to infer sentiment from **text**, **audio**, and **visual** signals. Although recent fusion models achieve strong performance, many still suffer from spurious lexical correlations and unstable fusion under noisy non-textual evidence.

**TECRFN** treats text as the semantic anchor while allowing audio and visual cues to contribute **adaptively** and **reliability-aware**. Instead of simply stacking standard modules, the framework is organized as a failure-mode-oriented pipeline:

**text purification → semantic injection → reliability-aware fusion**

This design improves alignment quality, stability, and robustness under bias shift.

## Main Components

### 1. Text Debiasing Module (TDM)
TDM uses a dual-branch design to reduce sentiment-irrelevant lexical shortcuts in the text stream:

- **In-sample knowledge selection** via self-attention
- **Cross-sample knowledge selection** via a global dictionary
- **Adversarial debiasing objective** with a gradient reversal layer

The goal is to improve the quality of the textual anchor **before** cross-modal interaction begins.

### 2. Text-enhanced Transformer (TET)
TET injects text-derived sentiment semantics into the non-textual streams:

- text-oriented multi-head attention
- cross-modal mapping between modality pairs
- self-attention encoding for modality-aware refinement

This reduces the semantic gap between text and non-text modalities before deep fusion.

### 3. Text-guided Cross-modal Reinforced Transformer (TGCRT)
TGCRT performs reliability-aware cross-modal fusion with four key mechanisms:

- **Dual-path attention**
  - direct target-to-text alignment branch
  - text-reinforced semantic branch
- **Similarity-biased gating** to adaptively select the more trustworthy branch
- **Channel attention** to suppress noisy channels and recalibrate sentiment-relevant dimensions
- **Residual steady-state connection** to preserve target-modality identity and stabilize optimization

### 4. Memory Encoder
The memory encoder captures long-range emotional dependencies and global contextual patterns that may not be preserved by local attention alone.

### 5. Unimodal Label Generation Module (ULGM)
ULGM provides unimodal auxiliary supervision to preserve modality-specific discriminability during multimodal fusion.

## Framework Pipeline

The overall TECRFN pipeline contains the following stages:

1. **Unimodal feature extraction**
2. **Contextual encoding**
3. **Text debiasing (TDM)**
4. **Text-enhanced interaction (TET)**
5. **Text-guided reinforced fusion (TGCRT)**
6. **Memory encoding and unimodal supervision (ULGM)**
7. **Sentiment prediction**

> Note: If you want to show the architecture figure in GitHub, convert the paper figure from `.eps` to `.png` or `.jpg` first, then place it in the repository and insert it below.

<!-- Example:
<p align="center">
  <img src="figures/tecrfn_framework.png" width="90%" alt="TECRFN framework" />
</p>
-->

## Datasets

We evaluate TECRFN on two standard multimodal sentiment benchmarks under the **unaligned** setting.

| Dataset | Train | Valid | Test | Total |
|--------|------:|------:|-----:|------:|
| CMU-MOSI  | 1,284  | 229  | 686  | 2,199  |
| CMU-MOSEI | 16,326 | 1,871 | 4,659 | 22,856 |

### Task Setting
- **Input modalities**: text, audio, vision
- **Learning setting**: unaligned multimodal sentiment regression/classification
- **Evaluation metrics**:
  - **Regression**: MAE, Corr
  - **Classification**: Acc-2, F1
    - **NN**: negative vs. non-negative
    - **PN**: negative vs. positive (zero labels excluded)

## Feature Extraction

The framework uses the following feature extraction strategy described in the paper:

- **Text**: BERT embeddings
- **Vision**: ViT features from preprocessed speaker facial regions
- **Audio**: COVAREP acoustic features

A contextual encoder composed of **sLSTM** and **TCN / projection layers** aligns the three modalities into a shared sequence space.

## Experimental Settings

The main experimental settings reported in the paper are:

- **Optimizer**: Adam
- **Learning rate**: `6e-6`
- **Dropout**: `0.3`
- **Weight decay**: `0.1`
- **Unified sequence length**: `T = 50`
- **Hidden dimension**: `d = 50`
- **Text-oriented attention heads**: `5`
- **Batch size (CMU-MOSI)**: `35`
- **Batch size (CMU-MOSEI)**: `32`

For the bias-shift robustness study:

- shortcut-token injection probability: `p_inj = 0.30`
- candidate shortcut tokens are selected from the training corpus only
- robustness results are averaged over **5 random seeds**

## Main Results

### Comparative Results

| Dataset | MAE ↓ | Corr ↑ | Acc-2 (NN / PN) ↑ | F1 (NN / PN) ↑ |
|--------|------:|-------:|------------------:|---------------:|
| CMU-MOSI  | **0.699** | **0.812** | **85.12 / 86.68** | **84.89 / 86.64** |
| CMU-MOSEI | **0.535** | **0.775** | **85.13 / 87.65** | **84.93 / 87.43** |

### Comparison with Recent Text-guided Fusion Methods

| Dataset | Model | MAE ↓ | Corr ↑ | Acc-2 (NN / PN) ↑ | F1 (NN / PN) ↑ |
|--------|------|------:|-------:|------------------:|---------------:|
| CMU-MOSI  | TETFN  | 0.717 | 0.800 | 84.05 / 86.10 | 83.83 / 86.07 |
| CMU-MOSI  | TCHFN  | 0.748 | 0.780 | 85.57 / 86.13 | 85.41 / 86.31 |
| CMU-MOSI  | **TECRFN** | **0.699** | **0.812** | **85.12 / 86.68** | **84.89 / 86.64** |
| CMU-MOSEI | TETFN  | 0.551 | 0.748 | 84.25 / 85.18 | 84.18 / 85.27 |
| CMU-MOSEI | TCHFN  | 0.538 | 0.770 | 84.01 / 86.27 | 84.14 / 86.48 |
| CMU-MOSEI | **TECRFN** | **0.535** | **0.775** | **85.13 / 87.65** | **84.93 / 87.43** |

## Why TECRFN?

Compared with previous multimodal fusion methods, TECRFN provides a unified solution to two persistent failure modes in unaligned multimodal sentiment analysis:

- it **reduces shortcut-prone lexical bias** before fusion,
- it **injects text semantics** into non-text modalities,
- it **adapts branch selection** according to query-branch agreement,
- it **suppresses noisy channels** after fusion,
- it **preserves long-range context** and **unimodal discriminability**.

## Reproducibility Notes

To reproduce the setting described in the paper, please make sure that:

1. CMU-MOSI and CMU-MOSEI are prepared with the **official unaligned splits**.
2. Text, visual, and audio features are extracted using **BERT**, **ViT**, and **COVAREP**, respectively.
3. All modalities are projected into a unified sequence representation with `T = 50` and `d = 50`.
4. Validation data are used to select the adversarial coefficient in TDM.
5. Robustness experiments are run with multiple random seeds.

## Citation

If you find this repository useful, please cite the following manuscript:

```bibtex
@misc{zhao_tecrfn,
  title={Text-enhanced Cross-modal Reinforced Fusion Network for Multimodal Sentiment Analysis},
  author={Wantong Zhao and Yongqing Wu},
  note={Manuscript}
}
