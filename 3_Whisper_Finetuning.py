#!/usr/bin/env python
# coding: utf-8

# ## 3 Whisper Finetuning
# 
# Now the fun part finally starts. This notebook will finetune the whisper model on romansh to improve transcription performance.

# In[1]:


import os
import torch
from datasets import load_dataset, DatasetDict, Audio, Features, Sequence, Value, Dataset
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
from constants import FOLDER_NAMES, DATA_ROOT


# In[2]:


os.environ['CUDA_VISIBLE_DEVICES'] = str(5)


# In[3]:

output_dir = "./whisper-medium-rm-finetuned"

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

# Initialize lists to hold all data
all_train_data = []
all_test_data = []

# Load data from each idiom folder
for idiom_folder in FOLDER_NAMES:
    idiom_path = os.path.join(DATA_ROOT, idiom_folder)
    clips_path = os.path.join(idiom_path, "clips")
    
    print(f"\n📂 Processing {idiom_folder}...")
    
    try:
        # Load TSV files
        train_df = pd.read_csv(os.path.join(idiom_path, "train.tsv"), sep="\t")
        validated_df = pd.read_csv(os.path.join(idiom_path, "validated.tsv"), sep="\t")
        test_df = pd.read_csv(os.path.join(idiom_path, "test.tsv"), sep="\t")
        
        # Combine train and validated
        idiom_train = pd.concat([train_df, validated_df], ignore_index=True)
        
        # Add full audio paths
        idiom_train["audio"] = clips_path + "/" + idiom_train["path"]
        test_df["audio"] = clips_path + "/" + test_df["path"]
        
        # Add idiom label (optional, but useful for analysis)
        idiom_train["idiom"] = idiom_folder
        test_df["idiom"] = idiom_folder
        
        # Filter existing audio files (optional but recommended)
        train_before = len(idiom_train)
        idiom_train = idiom_train[idiom_train["audio"].apply(os.path.exists)]
        test_before = len(test_df)
        test_df = test_df[test_df["audio"].apply(os.path.exists)]
        
        print(f"  Train+Validated: {train_before} → {len(idiom_train)} (filtered)")
        print(f"  Test: {test_before} → {len(test_df)} (filtered)")
        
        # Add to combined lists
        all_train_data.append(idiom_train)
        all_test_data.append(test_df)
        
    except Exception as e:
        print(f"  ⚠️ Error loading {idiom_folder}: {e}")

# Combine all idioms
if all_train_data and all_test_data:
    train_combined_df = pd.concat(all_train_data, ignore_index=True)
    test_combined_df = pd.concat(all_test_data, ignore_index=True)
    
    print("\n" + "="*60)
    print("📊 Combined Dataset Statistics")
    print("="*60)
    print(f"Total training samples: {len(train_combined_df)}")
    print(f"Total test samples: {len(test_combined_df)}")
    
    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_combined_df[["audio", "sentence", "idiom"]])
    test_dataset = Dataset.from_pandas(test_combined_df[["audio", "sentence", "idiom"]])
    
    # Create DatasetDict
    common_voice = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    print("\n✅ DatasetDict created successfully!")
    print(f"  Train: {len(common_voice['train'])} samples")
    print(f"  Test:  {len(common_voice['test'])} samples")
    
else:
    print("\n❌ No data loaded! Check your folder structure.")

# Optional: Show first few samples
print("\n📝 First 2 training samples:")
for i in range(min(2, len(train_combined_df))):
    print(f"  {i+1}. {train_combined_df['sentence'].iloc[i][:100]}...")


# In[5]:


print("Loading Whisper components...")

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name, task=task)
processor = WhisperProcessor.from_pretrained(model_name, task=task)

print("✅ Components loaded")


# In[6]:


# Cell 6: Memory-Efficient Dataset Preparation

print("Preparing dataset in streaming mode...")

# Define the features of our dataset explicitly
features = Features({
    "audio": Audio(sampling_rate=16000),
    "sentence": Value("string"),
    "idiom": Value("string")  # If you kept idiom column
})

def prepare_dataset_streaming(batch):
    """Process a batch of data efficiently"""
    processed_features = []
    processed_labels = []
    
    for i in range(len(batch["audio"])):
        audio_path = batch["audio"][i]
        
        try:
            # Load audio (librosa loads on-the-fly, memory efficient)
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            
            # Compute features
            input_features = feature_extractor(
                audio_array, 
                sampling_rate=16000
            ).input_features[0]
            processed_features.append(input_features)
            
            # Encode text
            labels = tokenizer(batch["sentence"][i]).input_ids
            processed_labels.append(labels)
            
        except Exception as e:
            print(f"⚠️ Error processing {audio_path}: {e}")
            # Add empty placeholders
            processed_features.append(np.zeros((80, 3000)))  # Dummy features
            processed_labels.append([0])  # Dummy label
    
    batch["input_features"] = processed_features
    batch["labels"] = processed_labels
    return batch

# Process with smaller batch size and remove_columns=False
common_voice = common_voice.map(
    prepare_dataset_streaming,
    batched=True,
    batch_size=32,  # Smaller batches to control memory
    remove_columns=None,  # Don't remove columns yet
    num_proc=1,  # Reduce parallel processes
    desc="Preparing dataset"
)

# Now remove unnecessary columns
columns_to_remove = [col for col in common_voice["train"].column_names 
                     if col not in ["input_features", "labels"]]
common_voice = common_voice.remove_columns(columns_to_remove)

print("✅ Dataset prepared successfully")


# In[7]:


# Cell 7: Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# In[8]:


# Cell 8: Evaluation Metric
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


# In[9]:


# Cell 9: Load Model
print(f"Loading Whisper model: {model_name}...")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

# Disable cache during training
model.config.use_cache = False

# For Romansh, set forced decoder ids for the task
# IMPORTANT: Do NOT modify model.config.suppress_tokens directly!
if hasattr(processor, "get_decoder_prompt_ids"):
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(task=task)

# Remove this line completely:
# model.config.suppress_tokens = []   ← DELETE THIS LINE

print(f"✅ Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters)")

# In[10]:


# Cell 10: Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

print("Training arguments:")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Max steps: {training_args.max_steps}")
print(f"  FP16: {training_args.fp16}")


# In[11]:


# Cell 11: Initialize Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("✅ Trainer initialized")


# In[ ]:


# Cell 12: Train!
print("="*60)
print("Starting training...")
print("="*60)

trainer.train()

print("="*60)
print("✅ Training complete!")
print("="*60)


# In[ ]:


# Cell 13: Save Model
print(f"Saving model to {output_dir}...")

trainer.save_model()
tokenizer.save_pretrained(output_dir)
feature_extractor.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

print("✅ Model saved!")