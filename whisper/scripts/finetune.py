# Cell 1: Imports & Setup
import os
import sys
from pathlib import Path
import torch
from transformers import Seq2SeqTrainer

# Ensure the package is importable
notebook_dir = Path.cwd()
whisper_dir = notebook_dir.parent
sys.path.append(str(whisper_dir))

from whisper_asr import (
    load_idiom_data,
    OnTheFlyDataset,
    collate_fn,
    load_model_and_processor,
    compute_metrics,
    get_training_args,
)
from whisper_asr.utils import get_best_gpu

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = torch.device("cuda")
#DEVICE = torch.device(f"cuda:{get_best_gpu()}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Cell 2: Load Data (now returns plain lists)

print("Loading all Romansh idiom datasets...")
train_samples, val_samples = load_idiom_data()
print(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

# Cell 3: Load Model & Processor (with Italian language fixed)
MODEL_NAME = "openai/whisper-medium"
OUTPUT_DIR = "../models/whisper-medium-rm-all-it"

model, feature_extractor, tokenizer, processor = load_model_and_processor(
    MODEL_NAME, DEVICE, language="it"
)

print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# Cell 4: Create On‑the‑Fly Datasets (pass the lists directly)
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

# Cell 5: Training Arguments
training_args = get_training_args(OUTPUT_DIR)
print("Training arguments:")
print(training_args)

# Cell 6: Define a wrapper for compute_metrics that already knows the tokenizer.
from functools import partial
compute_metrics_fn = partial(compute_metrics, tokenizer=tokenizer)

# Cell 7: Initialize Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics_fn,
)
print("Trainer initialized.")

# Cell 9: Start Training
print("Starting training...")
trainer.train()

# Cell 10: Save Final Model
print(f"Saving model to {OUTPUT_DIR}")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)
feature_extractor.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Done.")