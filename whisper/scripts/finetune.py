"""
Whisper Fine-Tuning Script for Romansh ASR.

This script orchestrates the fine-tuning of an OpenAI Whisper model using Hugging Face's 
Seq2SeqTrainer. It loads custom Romansh idiom data, sets up an on-the-fly feature extraction 
and tokenization dataset wrapper, hooks up evaluation metrics via partial functions, and 
handles model checkpointing and serialization.
"""

# Imports & Setup
import os
import sys
from pathlib import Path
import torch
from transformers import Seq2SeqTrainer
from functools import partial

# Resolve the repository paths dynamically so the execution context can safely 
# locate and import the absolute local 'whisper_asr' module namespace.
script_dir = Path(__file__).resolve().parent
whisper_dir = script_dir.parent
sys.path.append(str(whisper_dir))

from whisper_asr import (
    load_idiom_data,
    OnTheFlyDataset,
    collate_fn,
    load_model_and_processor,
    compute_metrics,
    get_training_args
)
from whisper_asr.constants import MODELS_ROOT
from whisper_asr import apply_causal_attention_mask

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Pin the target compute engine explicitly to an available NVIDIA GPU wrapper
DEVICE = torch.device("cuda")
# set to true if you want to apply attention masking to future speech
STREAMING = False
print(f"Using device: {DEVICE}")

# load train and validation set
print("Loading all Romansh idiom datasets...")
train_samples, val_samples = load_idiom_data()
print(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

# Load Model & Processor
MODEL_NAME = "openai/whisper-medium"
OUTPUT_DIR = MODELS_ROOT / "whisper-medium-rm-stream"

# Load the base model structure, log mel filterbanks extractor, and target tokenizers.
# Note: "it" (Italian) is specified as a baseline proxy
model, feature_extractor, tokenizer, processor = load_model_and_processor(
    MODEL_NAME, DEVICE, language="it"
)

print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# attention masking
if STREAMING:
    apply_causal_attention_mask(model)

# Wrap lists in an on-the-fly generator class. Audio sampling arrays and target 
# transcript tokens will be encoded lazily during training steps to conserve RAM.
train_dataset = OnTheFlyDataset(
    train_samples,          # list of dicts
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    language="it",
    task="transcribe",
)
eval_dataset = OnTheFlyDataset(
    val_samples,
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    language="it",
    task="transcribe",
)
print(f"Train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}")

# Extract standard batch dimensions, learning rates, and save strategies
training_args = get_training_args(OUTPUT_DIR)
print("Training arguments:")
print(training_args)

# Bind the current tokenizer state into our metric scoring function ahead of time.
# This aligns the callback footprint with what the Seq2SeqTrainer expect loop expects.
compute_metrics_fn = partial(compute_metrics, tokenizer=tokenizer)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,       # Dynamic batch padding logic for lengths
    compute_metrics=compute_metrics_fn,
)
print("Trainer initialized.")

# Start Training
print("Starting training...")
trainer.train()

# Save Final Model
print(f"Saving model to {OUTPUT_DIR}")
# Write the fine-tuned model checkpoint tensor array to disk
trainer.save_model()
# Save config configurations alongside vocabulary matrices to ensure full standalone portability
tokenizer.save_pretrained(OUTPUT_DIR)
feature_extractor.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Done.")