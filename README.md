# GLAMNet: High-Fidelity Streaming Neural Vocoder for Indic Languages

Welcome to the official repository for **GLAMNet**. 
This repository contains the configuration files, evaluation sample website, and setup instructions.

**Note**: The full training codebase, model weights, and inference scripts will be released here upon paper acceptance.

## Audio Samples Website
We provide an interactive demo to listen to our GLAMNet reconstructed audio compared against baselines.
Please check our GitHub pages: [Demo Website](https://speechforge.github.io/GlamNet/)

## Setup and Indic MFA Alignment

Our training pipeline relies on aligned phonetic data using IndicMFA. 
Before the full code release, you can prepare your dataset using the following instructions:

### 1. Clone IndicMFA
Please clone the IndicMFA repository to handle forced alignment for Indic languages:
```bash
git clone https://github.com/AI4Bharat/IndicMFA.git
cd IndicMFA
# Follow the official setup instructions inside the IndicMFA repo
```

### 2. Dataset Format
Ensure your dataset is formatted in standard Kaldi/MFA format. For each language, you need:
- `wavs/`: Directory containing all `.wav` audio files (24kHz recommended).
- `transcripts.txt`: A single text file mapping `file_id|transcript`.

### 3. Running Alignment
Run the forced aligner to generate TextGrids for your dataset. The output TextGrids will be directly consumed by GLAMNet once the code is available.

## Repository Structure
- `docs/`: Contains the interactive HTML demo and baseline audio samples.
- `configs/`: YAML configuration files containing the hyperparameter structures used in our models.
- `train.py`: Skeletal training entry point (full logic pending).

