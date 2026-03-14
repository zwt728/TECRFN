TECRFN: Text-enhanced Cross-modal Reinforced Fusion Network for Multimodal Sentiment Analysis
This repository contains the implementation and experimental resources for TECRFN, a text-guided framework designed for multimodal sentiment analysis (MSA) under the unaligned setting.

TECRFN addresses two coupled challenges in multimodal sentiment analysis:

Textual shortcut bias: Spurious lexical correlations in the text stream can corrupt the textual anchor and mislead cross-modal learning.

Unstable multimodal fusion: Noisy or asynchronous acoustic/visual evidence may degrade the final fused representation.

To tackle these issues, TECRFN adopts a staged architecture:

Text Debiasing Module (TDM) – Purifies the textual anchor by suppressing shortcut-prone lexical signals.

Text-enhanced Transformer (TET) – Injects sentiment-relevant textual semantics into audio and visual streams.

Text-guided Cross-modal Reinforced Transformer (TGCRT) – Performs query-adaptive dual-path fusion with similarity-biased gating, channel recalibration, and residual stabilization.

Memory Encoder and Unimodal Label Generation Module (ULGM) – Preserve long-range context and unimodal discriminability.

Overview
Multimodal sentiment analysis aims to infer sentiment from text, audio, and visual signals. Although recent fusion models have achieved strong performance, many still suffer from spurious lexical correlations and unstable fusion under noisy non-textual evidence.

TECRFN treats text as the semantic anchor while allowing audio and visual cues to contribute adaptively and reliability‑awarely. Instead of simply stacking standard modules, the framework is organized as a failure‑mode‑oriented pipeline:

text purification → semantic injection → reliability‑aware fusion

This design improves alignment quality, stability, and robustness under bias shift.

Main Components
1. Text Debiasing Module (TDM)
TDM uses a dual‑branch design to reduce sentiment‑irrelevant lexical shortcuts in the text stream:

In‑sample knowledge selection via self‑attention.

Cross‑sample knowledge selection via a global dictionary.

Adversarial debiasing objective with a gradient reversal layer.

The goal is to improve the quality of the textual anchor before cross‑modal interaction begins.

2. Text‑enhanced Transformer (TET)
TET injects text‑derived sentiment semantics into the non‑textual streams:

Text‑oriented multi‑head attention.

Cross‑modal mapping between modality pairs.

Self‑attention encoding for modality‑aware refinement.

This reduces the semantic gap between text and non‑text modalities before deep fusion.

3. Text‑guided Cross‑modal Reinforced Transformer (TGCRT)
TGCRT performs reliability‑aware cross‑modal fusion with four key mechanisms:

Dual‑path attention:

Direct target‑to‑text alignment branch.

Text‑reinforced semantic branch.

Similarity‑biased gating – adaptively selects the more trustworthy branch.

Channel attention – suppresses noisy channels and recalibrates sentiment‑relevant dimensions.

Residual steady‑state connection – preserves target‑modality identity and stabilizes optimization.

4. Memory Encoder
The memory encoder captures long‑range emotional dependencies and global contextual patterns that may not be preserved by local attention alone.

5. Unimodal Label Generation Module (ULGM)
ULGM provides unimodal auxiliary supervision to preserve modality‑specific discriminability during multimodal fusion.

Framework Pipeline
The overall TECRFN pipeline consists of the following stages:

Unimodal feature extraction

Contextual encoding

Text debiasing (TDM)

Text‑enhanced interaction (TET)

Text‑guided reinforced fusion (TGCRT)

Memory encoding and unimodal supervision (ULGM)

Sentiment prediction

Note: If you wish to include the architecture figure, convert the paper figure from .eps to .png or .jpg, place it in the repository, and insert it below.

<!-- Example: <p align="center"> <img src="figures/tecrfn_framework.png" width="90%" alt="TECRFN framework" /> </p> -->
Datasets
We evaluate TECRFN on two standard multimodal sentiment benchmarks under the unaligned setting.

Dataset	Train	Valid	Test	Total
CMU-MOSI	1,284	229	686	2,199
CMU-MOSEI	16,326	1,871	4,659	22,856
Task Setting
Input modalities: text, audio, vision

Learning setting: unaligned multimodal sentiment regression/classification

Evaluation metrics:

Regression: MAE, Corr

Classification: Acc‑2, F1

NN: negative vs. non‑negative

PN: negative vs. positive (zero labels excluded)

Feature Extraction
The framework uses the following features, as described in the paper:

Text: BERT embeddings

Vision: ViT features from preprocessed speaker facial regions

Audio: COVAREP acoustic features

A contextual encoder composed of sLSTM and TCN / projection layers aligns the three modalities into a shared sequence space.

Experimental Settings
Main experimental settings reported in the paper:

Optimizer: Adam

Learning rate: 6e-6

Dropout: 0.3

Weight decay: 0.1

Unified sequence length: T = 50

Hidden dimension: d = 50

Text‑oriented attention heads: 5

Batch size (CMU‑MOSI): 35

Batch size (CMU‑MOSEI): 32

For the bias‑shift robustness study:

Shortcut‑token injection probability: p_inj = 0.30

Candidate shortcut tokens are selected from the training corpus only

Robustness results are averaged over 5 random seeds

Main Results
Comparative Results
Dataset	MAE ↓	Corr ↑	Acc‑2 (NN / PN) ↑	F1 (NN / PN) ↑
CMU‑MOSI	0.699	0.812	85.12 / 86.68	84.89 / 86.64
CMU‑MOSEI	0.535	0.775	85.13 / 87.65	84.93 / 87.43
Comparison with Recent Text‑guided Fusion Methods
Dataset	Model	MAE ↓	Corr ↑	Acc‑2 (NN / PN) ↑	F1 (NN / PN) ↑
CMU‑MOSI	TETFN	0.717	0.800	84.05 / 86.10	83.83 / 86.07
CMU‑MOSI	TCHFN	0.748	0.780	85.57 / 86.13	85.41 / 86.31
CMU‑MOSI	TECRFN	0.699	0.812	85.12 / 86.68	84.89 / 86.64
CMU‑MOSEI	TETFN	0.551	0.748	84.25 / 85.18	84.18 / 85.27
CMU‑MOSEI	TCHFN	0.538	0.770	84.01 / 86.27	84.14 / 86.48
CMU‑MOSEI	TECRFN	0.535	0.775	85.13 / 87.65	84.93 / 87.43
Why TECRFN?
Compared with previous multimodal fusion methods, TECRFN provides a unified solution to two persistent failure modes in unaligned multimodal sentiment analysis:

It reduces shortcut‑prone lexical bias before fusion.

It injects text semantics into non‑text modalities.

It adapts branch selection according to query‑branch agreement.

It suppresses noisy channels after fusion.

It preserves long‑range context and unimodal discriminability.

Code Access
The source code for TECRFN is available via Baidu Netdisk:

Link: https://pan.baidu.com/s/1He_O291gbAG1ubawWAfcaQ?pwd=wt25
Password: wt25

Note: For easier access and international availability, we recommend uploading the code to GitHub or another open‑access platform. The provided link is for convenience and may be subject to change.

Reproducibility Notes
To reproduce the setting described in the paper, please ensure:

CMU‑MOSI and CMU‑MOSEI are prepared with the official unaligned splits.

Text, visual, and audio features are extracted using BERT, ViT, and COVAREP, respectively.

All modalities are projected into a unified sequence representation with T = 50 and d = 50.

Validation data are used to select the adversarial coefficient in TDM.

Robustness experiments are run with multiple random seeds.

Citation
If you find this repository useful, please cite the following manuscript:

bibtex
@misc{zhao_tecrfn,
  title={Text-enhanced Cross-modal Reinforced Fusion Network for Multimodal Sentiment Analysis},
  author={Wantong Zhao and Yongqing Wu},
  note={Manuscript}
}
