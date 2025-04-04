# HGF-MiLaG: Hierarchical Graph Fusion for Emotion Recognition in Conversation with Mid-Late Gender-Aware Strategy

This repository contains the implementation of **HGF-MiLaG**, a novel model for Emotion Recognition in Conversation (ERC) that leverages hierarchical graph fusion and a mid-late gender-aware strategy to improve performance. The model is designed to capture intra-modal and inter-modal speaker dependencies, temporal dependencies, and gender-related emotional dynamics in multimodal conversational data.

## 📌 Overview

HGF-MiLaG addresses two key challenges in ERC:

1. **Hierarchical Graph Fusion**: Constructs unimodal and multimodal graphs to model dependencies within and across modalities (text, audio, video).
2. **Mid-Late Gender-Aware Strategy**: Injects gender information at both mid-stage (unimodal graphs) and late-stage (multimodal fusion) to enhance emotion recognition by accounting for gender differences in emotional expression.

The model outperforms existing baselines on the IEMOCAP and MELD datasets, achieving state-of-the-art results in weighted F1 scores and accuracy.

## 📜 Paper

The original paper is published in Sensors (MDPI):
🔗 [HGF-MiLaG: Hierarchical Graph Fusion for Emotion Recognition in Conversation with Mid-Late Gender-Aware Strategy](https://www.mdpi.com/1424-8220/25/4/1182)

## 🚀 Key Features

- **Unimodal Graph Construction**: Captures speaker and temporal dependencies within each modality (text, audio, video) using directed graphs and Graph Transformers.
- **Multimodal Graph Fusion**: Integrates cross-modal features via a cross-modal attention mechanism and refines them using RGCN and the Weisfeiler-Lehman algorithm.
- **Gender-Aware Multi-Task Learning**: Jointly optimizes emotion and gender classification tasks with a mid-late fusion strategy to balance task-specific and shared features.
- **Dynamic Context Modeling**: Uses GRUs for sequential context encoding and temporal attention to capture emotional dynamics in conversations.

## 🛠 Installation

- python==3.8.16
- torch==1.12.0+cu116
- torch-geometric==2.3.0

## 🏃‍♂️ Usage

### Preparing datasets for training

    python preprocess.py --data './data/iemocap/newdata.pkl' --dataset="iemocap"

### Training networks 

    python train.py --data './data/iemocap/newdata.pkl' --from_begin --device=cuda --epochs=80 --batch_size=32 --n_speakers 2 

### Training networks_Automatic parameter tuning

    python train_optuna.py --data './data/iemocap/newdata.pkl' --from_begin --device=cuda --epochs=80 --batch_size=32 --n_speakers 2 

### Predictioning networks 

    python prediction.py --data=./data/iemocap/newdata.pkl --device=cuda --epochs=1 --batch_size=20 --n_speakers 2

## 📊 Results

HGF-MiLaG achieves the following performance on benchmark datasets:

| Dataset | Accuracy (%) | Weighted F1 (%) |
| :------ | :----------- | :-------------- |
| IEMOCAP | 70.98        | 71.02           |
| MELD    | 66.22        | 65.26           |

## 📧 Contact

For questions, contact Ziheng Li at lzh912743486@gmail.com

