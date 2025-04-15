# STAR for NeMo RNN-T Models

Implementation of STAR (Self-adaptation Transcripts via Attentive and Reliable Scores) for RNN-T models in NVIDIA NeMo, based on the paper [STAR: Self-adaptation Transcripts via Attentive and Reliable Scores for ASR](https://arxiv.org/pdf/2405.14161).

## Overview

STAR generates pseudo-labels by combining attention scores and confidence scores. For RNN-T models:

1. **Confidence Score**: Token-level confidence from RNN-T beam search
2. **Attentive Score**: Estimated from timestamps (as RNN-T doesn't provide direct attention weights)
3. **STAR Score**: Combined score identifying reliable tokens

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NVIDIA NeMo toolkit
- torchaudio, jiwer, fire

## Usage

### Data Format

Kaldi-style data directories with:
- `wav.scp`: Audio files
- `text`: Reference transcriptions

### Running

1. Edit `run_star_nemo.sh` with your paths:

```bash
DATASET="librispeech"
MODEL="stt_en_conformer_transducer_large"
TRAIN_DATA="/path/to/train/data/"
DEV_DATA="/path/to/dev/data/"
```

2. Run:

```bash
chmod +x run_star_nemo.sh
./run_star_nemo.sh
```

### Key Parameters

- `THRESHOLD`: For conflict detection (default: 2.0)
- `TAU`: Temperature parameter (default: 10.0)
- `TOP_PERCENT`: Data filtering ratio (default: 0.8)

## STAR Formula

```
c_over_a = c² / a
a_over_c = a² / c
conflict = (sigmoid((c_over_a - threshold) * tau) + sigmoid((a_over_c - threshold) * tau)) * a
no_conflict = sigmoid((threshold - c_over_a) * tau) * sigmoid((threshold - a_over_c) * tau) * a * exp((c - a) / tau)
star_score = conflict + no_conflict
```

## Output

- Processed datasets in `data/`
- Model checkpoints in `runs/`
- Final and best models based on WER

## Citation

If you use this implementation, please cite the original STAR paper:

```
@article{li2024star,
  title={STAR: Self-adaptation Transcripts via Attentive and Reliable Scores for ASR},
  author={Li, Ximing and Li, Jinchuan and Bu, Yiyao and Zhang, Chao and Cheng, Guangzhi},
  journal={arXiv preprint arXiv:2405.14161},
  year={2024}
}
``` 