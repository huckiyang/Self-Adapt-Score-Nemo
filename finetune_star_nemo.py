import torch
import torchaudio
import numpy as np
import os
import random
import copy
import heapq
import fire
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
from jiwer import wer as calculate_wer
from tqdm import tqdm
import types

# NeMo imports
import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecRNNTModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.utils import logging

# English text normalizer
from nemo.collections.asr.modules.rnnt import RNNTDecoder
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class StarRNNT:
    """Implementation of STAR for RNN-T model"""
    def __init__(
        self, 
        asr_model: EncDecRNNTModel,
        tokenizer: TokenizerSpec,
        threshold: float = 2.0,
        tau: float = 10.0,
    ):
        self.asr_model = asr_model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.tau = tau
        self.state_dict = copy.deepcopy(asr_model.state_dict())
        
    def get_star_scores(self, hyp: Hypothesis, encoder_outputs=None) -> List[float]:
        """Calculate STAR scores from a hypothesis
        
        Args:
            hyp: Hypothesis object with token confidences
            encoder_outputs: Optional outputs from encoder with attention weights
            
        Returns:
            List of STAR scores for each token
        """
        # Get confidence scores (probs)
        probs = hyp.token_confidence if hyp.token_confidence is not None else []
        
        # If token_confidence is empty, try to use frame_confidence
        if not probs and hyp.non_blank_frame_confidence:
            probs = hyp.non_blank_frame_confidence
        
        if not probs:
            # No confidence scores available, return uniform weights
            return [1.0] * (len(hyp.y_sequence) - 1)  # -1 to exclude blank token
        
        # Normalize confidence scores
        mean_probs = sum(probs) / len(probs)
        probs = [round(p / mean_probs, 3) for p in probs]
        
        # Get attention weights
        weights = []
        
        # Try to extract attention weights directly from FastConformer
        if encoder_outputs is not None and hasattr(encoder_outputs, 'attention_weights'):
            # Get attention weights from encoder
            attn_weights = encoder_outputs.attention_weights
            print("=== using fastconformer attention weights ===")
            
            # Select a representative layer and head (typically last layer works well)
            if isinstance(attn_weights, list) and len(attn_weights) > 0:
                # Take the last layer's attention
                layer_attn = attn_weights[-1]
                
                # Average across heads or take a specific head
                # For simplicity, let's average across all heads
                avg_attn = layer_attn.mean(dim=1)  # Shape: [batch, seq_len, seq_len]
                
                # Extract relevant attention scores for each token
                # This mapping can be complex and depends on how the model aligns
                # encoder frame positions to decoded tokens
                if hasattr(hyp, 'alignments') and hyp.alignments:
                    for align_idx in hyp.alignments:
                        if isinstance(align_idx, list):
                            # For RNNT, alignments can be 2D
                            # We need to find corresponding encoder frames
                            continue
                        elif isinstance(align_idx, int):
                            # For simpler cases, direct mapping
                            if 0 <= align_idx < avg_attn.size(-1):
                                # Sum attention for this position
                                weight = avg_attn[0, align_idx, :].sum().item()
                                weights.append(weight)
        
        # If direct attention extraction failed or returned empty, fall back to timestamp-based approach
        if not weights:
            # If timestamps available, use them to estimate attention weights
            if hasattr(hyp, 'timestamp') and hyp.timestamp:
                timestamp = hyp.timestamp
                if isinstance(timestamp, dict):
                    timestamp = timestamp.get('timestep', [])
                
                if timestamp:
                    # Calculate normalized time progression as attention estimate
                    max_time = max(timestamp) if timestamp else 1
                    weights = [round(t / max_time, 3) for t in timestamp]
        
        # If still no weights, use uniform weights
        if not weights:
            weights = [1.0] * len(probs)
        
        # Normalize weights
        if weights:
            mean_weights = sum(weights) / len(weights)
            weights = [round(w / mean_weights, 3) for w in weights]
        
        # Calculate STAR scores (final_weights)
        final_weights = []
        for ci, ai in zip(probs, weights):
            c_over_a, a_over_c = ci * ci / ai if ai > 0 else 0, ai * ai / ci if ci > 0 else 0
            conflict = (sigmoid((c_over_a - self.threshold) * self.tau) + sigmoid((a_over_c - self.threshold) * self.tau)) * ai
            no_conflict = (sigmoid((self.threshold - c_over_a) * self.tau) * sigmoid((self.threshold - a_over_c) * self.tau)) * ai * np.exp((ci - ai) / self.tau)
            final_weights.append(conflict + no_conflict)
        
        return final_weights
    
    def generate_pseudo_labels(self, audio: torch.Tensor, sample_rate: int = 16000) -> Tuple[str, List[float], float, int]:
        """Generate pseudo-labels with STAR scores for an audio sample
        
        Args:
            audio: Audio tensor
            sample_rate: Audio sample rate
            
        Returns:
            tuple: (transcription, star_scores, avg_wer, diversity)
        """
        # Resample audio if needed
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
            
        # Put model in evaluation mode and process audio
        self.asr_model.eval()
        with torch.no_grad():
            # Set the model to output attention weights
            self.asr_model.encoder._capture_attention = True
            
            # Process audio
            features = self.asr_model.preprocessor(audio.unsqueeze(0).to(device))
            encoded, encoded_len = self.asr_model.encoder(features)
            
            # Store encoder outputs with attention
            encoder_outputs = encoded  # This should now contain attention_weights attribute
            
            # Reset attention capturing to avoid memory issues
            self.asr_model.encoder._capture_attention = False
            
            # Get beam search results with confidence scores
            beam_results = self.asr_model.decoding.rnnt_decoder_predictions_tensor(
                encoded,
                encoded_len,
                return_hypotheses=True,
                calculate_token_confidence=True
            )
            
            # Get the best hypothesis
            hyp = beam_results[0][0]  # First sample, best hypothesis
            
            # Calculate STAR scores
            star_scores = self.get_star_scores(hyp, encoder_outputs)
            
            # Get transcription
            transcription = hyp.text if hyp.text else ''
            
            # Generate multiple outputs with small noise for diversity measurement
            avg_wer, generated_texts = 0, []
            for _ in range(5):
                # Add small noise to model weights
                new_state_dict = copy.deepcopy(self.state_dict)
                for k in new_state_dict.keys():
                    if torch.is_tensor(new_state_dict[k]) and new_state_dict[k].numel() > 0:
                        std = torch.std(new_state_dict[k])
                        noise = torch.randn_like(new_state_dict[k])
                        new_state_dict[k] = new_state_dict[k] + noise * std * 0.1
                
                # Load noisy weights
                self.asr_model.load_state_dict(new_state_dict)
                
                # Generate transcription with noisy model
                with torch.no_grad():
                    noisy_results = self.asr_model.transcribe(audio.unsqueeze(0).cpu().numpy())
                    noisy_text = noisy_results[0]
                    generated_texts.append(noisy_text)
                    avg_wer += calculate_wer([transcription], [noisy_text]) / 5
            
            # Restore original weights
            self.asr_model.load_state_dict(self.state_dict)
            
            # Calculate diversity as number of unique transcripts
            diversity = len(set(generated_texts))
            
        return transcription, star_scores, avg_wer, diversity

def data_preparation(
    data_path: str,
    star_rnnt: StarRNNT,
) -> Tuple[List[Dict[str, Any]], float]:
    """Prepare dataset with STAR pseudo-labels
    
    Args:
        data_path: Path to dataset (Kaldi format)
        star_rnnt: StarRNNT instance
        
    Returns:
        tuple: (dataset, WER)
    """
    # Read wav.scp and text files
    with open(data_path + "wav.scp", 'r') as f1:
        wave_data = f1.readlines()
    with open(data_path + "text", 'r') as f2:
        trans_data = f2.readlines()
    
    audio_data, txt_data = [], []
    for i in range(len(wave_data)):
        audio_data.append(wave_data[i])
        txt_data.append(trans_data[i])
    
    dataset = []
    all_pred, all_gt = [], []
    
    # Process each audio file
    for audio_line, text_line in tqdm(zip(audio_data, txt_data), total=len(audio_data)):
        audio_path = audio_line.strip().split()[1]
        text = ' '.join(text_line.split()[1:]).lower().strip()
        
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        audio = audio.squeeze(0)  # Remove channel dimension
        
        # Create dataset item
        item = {'audio': audio, 'text': text}
        
        # Generate pseudo-label and STAR scores
        pseudo_text, star_scores, avg_wer, diversity = star_rnnt.generate_pseudo_labels(audio, sr)
        
        # Store results
        item['pseudo_text'] = pseudo_text
        item['star_scores'] = torch.tensor(star_scores).unsqueeze(0)
        item['avg_wer'] = avg_wer
        item['diversity'] = diversity
        
        # Clean text
        pseudo_text = pseudo_text if len(pseudo_text) > 0 else '<UNK>'
        gt = text if len(text) > 0 else '<UNK>'
        
        dataset.append(item)
        all_pred.append(pseudo_text)
        all_gt.append(gt)
    
    # Calculate overall WER
    wer = calculate_wer(all_gt, all_pred)
    
    return dataset, wer

def evaluate(model: EncDecRNNTModel, dataset: List[Dict]) -> float:
    """Evaluate model on dataset
    
    Args:
        model: NeMo ASR model
        dataset: Dataset to evaluate on
        
    Returns:
        float: WER
    """
    model.eval()
    with torch.no_grad():
        all_pred, all_gt = [], []
        for item in tqdm(dataset):
            audio = item['audio'].unsqueeze(0)  # Add batch dimension
            pred = model.transcribe(audio.cpu().numpy())[0]
            
            # Clean text
            pred = pred if len(pred) > 0 else '<UNK>'
            gt = item['text'] if len(item['text']) > 0 else '<UNK>'
            
            all_pred.append(pred)
            all_gt.append(gt)
    
    return calculate_wer(all_gt, all_pred)

def train(
    MODEL = "stt_en_conformer_transducer_large",
    DATASET = "librispeech",
    TRAIN_DATA = "",
    DEV_DATA = "",
    SAVE_EVERY = 10,
    BATCH_SIZE = 16,
    GRADIENT_ACCUMULATION_STEPS = 4,
    LEARNING_RATE = 1e-5,
    EPOCHS = 10,
    THRESHOLD = 2.0,
    TOP_PERCENT = 0.8,
    TAU = 10,
    OUTPUT_DIR = "runs",
):
    """Train ASR model with STAR pseudo-labels
    
    Args:
        MODEL: NeMo model name or path to .nemo file
        DATASET: Dataset name
        TRAIN_DATA: Path to training data (Kaldi format)
        DEV_DATA: Path to dev data (Kaldi format)
        SAVE_EVERY: Save checkpoint every N steps
        BATCH_SIZE: Batch size
        GRADIENT_ACCUMULATION_STEPS: Gradient accumulation steps
        LEARNING_RATE: Learning rate
        EPOCHS: Number of epochs
        THRESHOLD: STAR threshold for conflict detection
        TOP_PERCENT: Percentage of top samples to keep
        TAU: Temperature for STAR score calculation
        OUTPUT_DIR: Directory to save outputs
    """
    # Load pre-trained model
    if MODEL.endswith('.nemo'):
        model = nemo_asr.models.EncDecRNNTModel.restore_from(MODEL)
    else:
        model = nemo_asr.models.EncDecRNNTModel.from_pretrained(MODEL)
    
    model = model.to(device)
    
    # Get tokenizer
    tokenizer = model.tokenizer
    
    # Create StarRNNT instance
    star_rnnt = StarRNNT(model, tokenizer, threshold=THRESHOLD, tau=TAU)
    
    # Prepare datasets
    logging.info(f"Preparing training data from {TRAIN_DATA}")
    train_dataset, train_wer = data_preparation(TRAIN_DATA, star_rnnt)
    logging.info(f"Preparing dev data from {DEV_DATA}")
    dev_dataset, dev_wer = data_preparation(DEV_DATA, star_rnnt)
    
    # Create output directory
    os.makedirs("data", exist_ok=True)
    torch.save(train_dataset, f'data/train_{DATASET}.pt')
    torch.save(dev_dataset, f'data/dev_{DATASET}.pt')
    
    # Filter training data based on uncertainty
    logging.info("Filtering training data")
    def product(item):
        return item['avg_wer'] * item['diversity']
    filtered_train_dataset = heapq.nsmallest(int(len(train_dataset) * TOP_PERCENT), train_dataset, key=product)
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CTCLoss(reduction='none')  # We will apply per-token weights
    
    # Create experiment directory
    exp_dir = os.path.join(OUTPUT_DIR, f'{DATASET}_{Path(MODEL).stem}')
    os.makedirs(exp_dir, exist_ok=True)
    
    # Training loop
    steps, best_loss, best_wer = 0, float('inf'), float('inf')
    
    for epoch in range(EPOCHS):
        logging.info(f"Epoch: {epoch + 1}/{EPOCHS}")
        
        # Shuffle training data
        random.shuffle(filtered_train_dataset)
        
        # Training
        model.train()
        logging.info("Training...")
        
        for i in range(0, len(filtered_train_dataset), BATCH_SIZE):
            batch_data = filtered_train_dataset[i:i+BATCH_SIZE]
            
            # Process batch
            batch_loss = 0
            for item in batch_data:
                # Forward pass
                audio = item['audio'].unsqueeze(0).to(device)
                
                # Get encoder features
                features = model.preprocessor(audio)
                encoded, encoded_len = model.encoder(features)
                
                # Get decoder outputs for pseudo-label
                pseudo_text = item['pseudo_text']
                pseudo_tokens = tokenizer.text_to_ids(pseudo_text)
                
                # Forward decoder to get log probs
                log_probs = model(
                    audio_signal=audio,
                    length=torch.tensor([audio.shape[1]]).to(device),
                    transcripts=[pseudo_text],
                    transcript_lengths=torch.tensor([len(pseudo_tokens)]).to(device)
                )
                
                # Get loss with token-level weighting from STAR scores
                loss = log_probs.loss
                
                # Apply STAR weights to loss
                star_weights = item['star_scores'].to(device)
                star_weights = star_weights / torch.mean(star_weights)  # Normalize
                
                # Weight loss by STAR scores (simple approximation since we can't directly
                # weight each token's contribution to RNNT loss)
                weighted_loss = loss * star_weights.mean()
                
                batch_loss += weighted_loss
            
            # Average loss over batch
            loss = batch_loss / len(batch_data)
            
            # Backward pass
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            steps += 1
            
            # Update weights
            if steps % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Evaluate and save checkpoint
            if steps % SAVE_EVERY == 0:
                # Save model
                model.save_to(f"{exp_dir}/step_{steps}.nemo")
                
                # Evaluate
                model.eval()
                dev_wer = evaluate(model, dev_dataset)
                model.train()
                
                logging.info(f"Step {steps}, Dev WER: {dev_wer:.4f}")
                
                # Save best model
                if dev_wer < best_wer or (dev_wer == best_wer and loss < best_loss):
                    model.save_to(f"{exp_dir}/best_model.nemo")
                    best_loss, best_wer = loss, dev_wer
    
    # Save final model
    model.save_to(f"{exp_dir}/final_model.nemo")
    logging.info(f"Training complete. Best WER: {best_wer:.4f}")

def setup_attention_capture(model):
    """Add attention capturing capability to the model's encoder"""
    encoder = model.encoder
    
    # Store the original forward method
    original_forward = encoder.forward
    
    # Create a new forward method that can capture attention
    def forward_with_attention(self, audio_signal, length=None):
        # Call the original forward method
        outputs = original_forward(audio_signal, length)
        
        # If attention capturing is enabled, extract and attach attention weights
        if hasattr(self, '_capture_attention') and self._capture_attention:
            # This assumes the conformer/fastconformer has self-attention layers
            # Extract attention from the relevant layers
            attention_weights = []
            for layer in self.conformer_layers:
                if hasattr(layer, 'self_attention'):
                    # Assume self_attention module has stored attention weights
                    # You might need to modify the self-attention module to store these weights
                    if hasattr(layer.self_attention, 'attention_weights'):
                        attention_weights.append(layer.self_attention.attention_weights)
            
            # Attach attention weights to outputs
            outputs.attention_weights = attention_weights
        
        return outputs
    
    # Add the capture flag
    encoder._capture_attention = False
    
    # Replace the forward method
    encoder.forward = types.MethodType(forward_with_attention, encoder)
    
    return model

def add_attention_hooks(model):
    """Add hooks to capture attention weights from Conformer layers"""
    
    # Function to capture attention weights
    def capture_attention(self, module, input, output):
        # For multi-head attention, the attention weights are typically in the output
        # The exact structure depends on the implementation
        if isinstance(output, tuple) and len(output) > 1:
            # Often attention weights are the 2nd element in the output tuple
            self.attention_weights = output[1]
        
    # Add hooks to all self-attention modules in the encoder
    for i, layer in enumerate(model.encoder.conformer_layers):
        if hasattr(layer, 'self_attention'):
            # Register forward hook to capture attention
            layer.self_attention.attention_weights = None
            layer.self_attention.register_forward_hook(
                types.MethodType(capture_attention, layer.self_attention)
            )
    
    return model

def add_nemo_attention_hooks(model):
    """Add hooks to capture attention weights from Conformer layers"""
    
    # Function to capture attention weights
    def capture_attention(self, module, input, output):
        # For nemo multi-head attention, the attention weights are not stored, so extract from the attention dropout's input
        # The exact structure depends on the implementation
        if isinstance(input, tuple) and len(input) > 0:
            # Often attention weights are the 2nd element in the output tuple
            self.attention_weights = input[0]
        
    # Add hooks to all self-attention modules in the encoder
    for i, layer in enumerate(model.encoder.conformer_layers):
        if hasattr(layer, 'self_attn'):
            # Register forward hook to capture attention
            layer.self_attn.attention_weights = None
            layer.self_attn.dropout.register_forward_hook(
                types.MethodType(capture_attention, layer.self_attention)
            )
    
    return model

if __name__ == "__main__":
    fire.Fire(train) 