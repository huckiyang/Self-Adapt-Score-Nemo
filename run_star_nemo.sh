#!/usr/bin/env bash

# Activate NeMo environment - modify as needed
source activate nemo_env

# Dataset and model settings
DATASET="librispeech"
MODEL="stt_en_conformer_transducer_large"  # Pre-trained NeMo model
TRAIN_DATA="/path/to/train/data/"  # Path to Kaldi format training data 
DEV_DATA="/path/to/dev/data/"      # Path to Kaldi format dev data
OUTPUT_DIR="runs"

# STAR parameters
THRESHOLD=2.0
TAU=10.0
TOP_PERCENT=0.8

# Training parameters
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=2  # Effectively a batch size of 32
LEARNING_RATE=1e-5
EPOCHS=10
SAVE_EVERY=50

# Run STAR finetuning
python finetune_star_nemo.py \
    --MODEL ${MODEL} \
    --DATASET ${DATASET} \
    --TRAIN_DATA ${TRAIN_DATA} \
    --DEV_DATA ${DEV_DATA} \
    --SAVE_EVERY ${SAVE_EVERY} \
    --BATCH_SIZE ${BATCH_SIZE} \
    --GRADIENT_ACCUMULATION_STEPS ${GRADIENT_ACCUMULATION_STEPS} \
    --LEARNING_RATE ${LEARNING_RATE} \
    --EPOCHS ${EPOCHS} \
    --THRESHOLD ${THRESHOLD} \
    --TOP_PERCENT ${TOP_PERCENT} \
    --TAU ${TAU} \
    --OUTPUT_DIR ${OUTPUT_DIR} 