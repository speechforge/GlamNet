# GLAMNet: High-Fidelity Streaming Neural Vocoder for Indic Languages

GLAMNet is a streaming neural vocoder designed specifically for the phonetic complexities and spectral characteristics of Indic languages.

## Acoustic & Phonetic Alignment via IndicMFA

GLAMNet's training pipeline requires precise letter-wise alignment between the speech corpus and text transcripts. To achieve this, we utilize a Grapheme-to-Grapheme (G2G) methodology rather than traditional Grapheme-to-Phoneme (G2P) processes.

### 1. Setup and Dependencies
We employ [IndicMFA](https://github.com/AI4Bharat/IndicMFA) to perform forced alignment for Indian languages.
```bash
git clone https://github.com/AI4Bharat/IndicMFA.git
cd IndicMFA
```

You must download the pre-trained Acoustic Models (AM) and G2G Pronunciation Dictionaries (mapping each grapheme to itself) for your target languages from the [IndicMFA Releases](https://github.com/AI4Bharat/IndicMFA/releases).

### 2. Dataset Preparation
Your speech corpus must contain `.wav` audio files paired with corresponding text transcripts. The audio should be resampled to 24kHz for compatibility with the GLAMNet data loaders.

### 3. Generating Alignments
Run the Montreal Forced Aligner to generate the necessary TextGrid representations:
```bash
mfa align <corpus_directory> <g2g_dictionary> <acoustic_model> <output_textgrids>
```
The output TextGrids contain temporal boundaries for each grapheme. These phonetic alignments are strictly required by the GLAMNet architecture for accurate pitch and duration conditioning.

## Training Configuration

Initialize the distributed training pipeline by providing the target YAML configuration file.

```bash
python train.py --config configs/vocos.yaml --num_gpus 1
```

*Note: The dataset initialization will fail if the pre-computed MFA TextGrid alignments are not located in the target directory.*

## Interactive Audio Samples

We provide an interactive demo to listen to our GLAMNet reconstructed audio compared against state-of-the-art baselines.

[Listen to the Audio Samples](https://speechforge.github.io/GlamNet/)
