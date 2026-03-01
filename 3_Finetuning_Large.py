#!/usr/bin/env python
# coding: utf-8

# ## 3 Whisper Finetuning - On-the-Fly Version
# 
# This notebook finetunes Whisper on all Romansh idioms using on-the-fly processing to handle 300+ hours of data without memory errors.

# In[1]:


import os
import torch
from datasets import load_dataset, DatasetDict, Audio, Dataset
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import numpy as np
import pandas as pd
import librosa
from torch.utils.data import DataLoader
from constants import FOLDER_NAMES, DATA_ROOT
from helpers import get_idiom_name_by_folder


# In[2]:


# Set your desired GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(5)  # Change to your GPU


# In[3]:

output_dir = "./whisper-medium-rm-all"
model_name = "openai/whisper-medium"
task = "transcribe"


# In[4]:


# Cell 4: Load and Combine All Idiom Datasets
print("="*60)
print("Loading all Romansh idiom datasets")
print("="*60)
print(f"Found {len(FOLDER_NAMES)} idiom folders:")
for folder in FOLDER_NAMES:
    print(f"  📁 {folder}")

# Initialize lists to hold all data as dictionaries (not DataFrames)
all_train_samples = []
all_validation_samples = []

# Load data from each idiom folder
for idiom_folder in FOLDER_NAMES:
    idiom_path = os.path.join(DATA_ROOT, idiom_folder)
    clips_path = os.path.join(idiom_path, "clips")
    idiom_name = get_idiom_name_by_folder(idiom_folder)
    
    print(f"\n📂 Processing {idiom_folder}...")
    
    try:
        # Load TSV files
        train_df = pd.read_csv(os.path.join(idiom_path, "train.tsv"), sep="\t")
        validation_df = pd.read_csv(os.path.join(idiom_path, "validation.tsv"), sep="\t")
        
        # Convert to dictionaries (simpler for Dataset.from_list)
        for _, row in train_df.iterrows():
            audio_path = os.path.join(clips_path, row["path"])
            if os.path.exists(audio_path):
                all_train_samples.append({
                    "audio": audio_path,
                    "sentence": str(row["sentence"]),
                    "idiom": idiom_name
                })
        
        for _, row in validation_df.iterrows():
            audio_path = os.path.join(clips_path, row["path"])
            if os.path.exists(audio_path):
                all_validation_samples.append({
                    "audio": audio_path,
                    "sentence": str(row["sentence"]),
                    "idiom": idiom_name
                })
        
        print(f"  Train: {len(train_df)} total, {len([s for s in all_train_samples if s['idiom']==idiom_name])} with audio")
        print(f"  Validation: {len(validation_df)} total, {len([s for s in all_validation_samples if s['idiom']==idiom_name])} with audio")
        
    except Exception as e:
        print(f"  ⚠️ Error loading {idiom_folder}: {e}")

# Create datasets directly from dictionaries
print("\n" + "="*60)
print("📊 Combined Dataset Statistics")
print("="*60)

if all_train_samples and all_validation_samples:
    # Create datasets from list of dictionaries
    train_dataset = Dataset.from_list(all_train_samples)
    validation_dataset = Dataset.from_list(all_validation_samples)
    
    # Create DatasetDict
    common_voice = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset
    })
    
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(validation_dataset)}")
    
    # Show distribution by idiom
    print("\n📈 Training distribution by idiom:")
    train_idioms = {}
    for item in all_train_samples:
        train_idioms[item["idiom"]] = train_idioms.get(item["idiom"], 0) + 1
    for idiom, count in train_idioms.items():
        print(f"  {idiom}: {count} samples ({count/len(train_dataset)*100:.1f}%)")
    
    print("\n✅ Raw DatasetDict created successfully!")
    print(f"  Train: {len(common_voice['train'])} samples")
    print(f"  Validation:  {len(common_voice['validation'])} samples")
    
else:
    print("\n❌ No data loaded! Check your folder structure.")
    raise ValueError("No data loaded")

# Optional: Show first few samples
print("\n📝 First 2 training samples:")
for i in range(min(2, len(train_dataset))):
    print(f"  {i+1}. {train_dataset[i]['sentence'][:100]}...")


# In[5]:


print("Loading Whisper components...")

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name, task=task)
processor = WhisperProcessor.from_pretrained(model_name, task=task)

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("✅ Components loaded")

# In[7]:


# Cell 7: Create On-the-Fly Dataset Class
class WhisperOnTheFlyDataset(torch.utils.data.Dataset):
    """Dataset that processes audio on-the-fly during training"""
    
    def __init__(self, hf_dataset, feature_extractor, tokenizer):
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        audio_path = item["audio"]
        audio_array, sr = librosa.load(audio_path, sr=16000)
        
        input_features = self.feature_extractor(
            audio_array,
            sampling_rate=16000
        ).input_features[0]
        
        # Add truncation here!
        labels = self.tokenizer(
            item["sentence"],
            truncation=True,
            max_length=448  # Explicitly set max length
        ).input_ids
        
        return {
            "input_features": input_features,
            "labels": labels
        }

# Create on-the-fly datasets
print("Creating on-the-fly datasets...")
train_dataset = WhisperOnTheFlyDataset(
    common_voice["train"], 
    feature_extractor, 
    tokenizer
)
eval_dataset = WhisperOnTheFlyDataset(
    common_voice["validation"], 
    feature_extractor, 
    tokenizer
)
print(f"✅ Created train dataset with {len(train_dataset)} samples")
print(f"✅ Created eval dataset with {len(eval_dataset)} samples")


# In[8]:


# Cell 8: Custom Collate Function for Variable-Length Sequences
def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    
    # Pad input features (shape: [80, variable_len])
    input_features = [item["input_features"] for item in batch]
    max_feat_len = max(f.shape[-1] for f in input_features)
    
    padded_features = []
    for f in input_features:
        pad_len = max_feat_len - f.shape[-1]
        if pad_len > 0:
            # Pad along time dimension
            padding = np.zeros((f.shape[0], pad_len))
            padded = np.concatenate([f, padding], axis=-1)
        else:
            padded = f
        padded_features.append(padded)
    
    # Pad labels
    labels = [item["labels"] for item in batch]
    max_label_len = max(len(l) for l in labels)
    
    padded_labels = []
    for l in labels:
        pad_len = max_label_len - len(l)
        padded = l + [-100] * pad_len  # -100 is ignored in loss
        padded_labels.append(padded)
    
    return {
        "input_features": torch.tensor(np.array(padded_features), dtype=torch.float32),
        "labels": torch.tensor(np.array(padded_labels), dtype=torch.long)
    }

print("✅ Collate function defined")


# In[9]:


# Cell 9: Evaluation Metric
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# In[10]:


# Cell 10: Load Model
print(f"Loading Whisper model: {model_name}...")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

# Disable cache during training
model.config.use_cache = False

print(f"✅ Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters)")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# In[11]:


# Cell 11: Training Arguments (Memory-Optimized)
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,  # Reduced for memory
    gradient_accumulation_steps=2,   # Effective batch size = 16
    learning_rate=1e-5,
    warmup_steps=1000,
    max_steps=10000,  # More steps for larger dataset
    gradient_checkpointing=True,      # Critical for memory
    fp16=True,                        # Mixed precision
    eval_strategy="steps",
    per_device_eval_batch_size=4,     # Smaller eval batches
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    dataloader_num_workers=0,          # No multiprocessing to avoid issues
    remove_unused_columns=False,        # Keep all columns
    ddp_find_unused_parameters=None,
)

print("Training arguments:")
print(f"  Batch size (per device): {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Max steps: {training_args.max_steps}")
print(f"  FP16: {training_args.fp16}")
print(f"  Gradient checkpointing: {training_args.gradient_checkpointing}")


# In[12]:


# Cell 12: Initialize Trainer with Custom Datasets and Collate
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,           # Use custom collate, not processor
    compute_metrics=compute_metrics,
)

print("✅ Trainer initialized")


# In[13]:


# Cell 13: Quick Test Before Full Training
print("\n" + "="*60)
print("🔍 Quick test on one batch")
print("="*60)

# Test one batch
test_batch = next(iter(torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=2, 
    collate_fn=collate_fn
)))

print(f"Input features shape: {test_batch['input_features'].shape}")
print(f"Labels shape: {test_batch['labels'].shape}")
print("✅ Batch test passed - ready for training!")


# In[14]:


# Cell 14: Train!
print("\n" + "="*60)
print("🚀 Starting training on all Romansh idioms")
print("="*60)
print(f"Total training samples: {len(train_dataset)}")
print(f"Total validation samples: {len(eval_dataset)}")
print(f"Training will run for {training_args.max_steps} steps")
print("="*60 + "\n")

try:
    trainer.train()
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60)
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")
except Exception as e:
    print(f"\n❌ Training error: {e}")
    raise

# In[15]:


# Cell 15: Save Model
print(f"\n💾 Saving model to {output_dir}...")

trainer.save_model()
tokenizer.save_pretrained(output_dir)
feature_extractor.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

print("✅ Model saved!")
