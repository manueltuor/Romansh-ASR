import os
import torch
import pandas as pd
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments
import librosa
from sklearn.metrics import accuracy_score, f1_score
from helpers import load_all_data, get_best_gpu

# ---------- GPU Selection ----------
best_gpu = get_best_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # ensure consistent ordering

print(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
device = torch.device(f"cuda:{best_gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} (GPU {best_gpu})")

# ---------- Load Data ----------
train_df = load_all_data("train")
val_df = load_all_data("validation")
# test_df = load_all_data("test")  # optional

# Create label mapping
idioms = train_df['idiom'].unique()
label2id = {idiom: i for i, idiom in enumerate(idioms)}
id2label = {i: idiom for idiom, i in label2id.items()}
train_df['label'] = train_df['idiom'].map(label2id)
val_df['label'] = val_df['idiom'].map(label2id)

# ---------- Model & Feature Extractor ----------
model_name = "facebook/wav2vec2-base"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(idioms),
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
)
# Do NOT manually move model to device; Trainer will handle it.

# ---------- Custom Dataset ----------
class IdiomDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_extractor):
        self.df = df
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, sr = librosa.load(row['audio_path'], sr=16000)
        inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="np")  # use numpy
        return {
            'input_values': inputs['input_values'][0],          # (time,) as numpy array
            'attention_mask': np.ones(len(inputs['input_values'][0]), dtype=np.int64),
            'labels': row['label']                               # integer label
        }

train_dataset = IdiomDataset(train_df, feature_extractor)
val_dataset = IdiomDataset(val_df, feature_extractor)

# ---------- Custom Collate Function ----------
def collate_fn(batch):
    input_values = [item['input_values'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    max_len = max(len(v) for v in input_values)
    padded_inputs = []
    padded_masks = []
    for v, m in zip(input_values, attention_masks):
        pad_len = max_len - len(v)
        if pad_len > 0:
            padded_inputs.append(np.pad(v, (0, pad_len), 'constant'))
            padded_masks.append(np.pad(m, (0, pad_len), 'constant'))
        else:
            padded_inputs.append(v)
            padded_masks.append(m)

    return {
        'input_values': torch.tensor(padded_inputs, dtype=torch.float32),
        'attention_mask': torch.tensor(padded_masks, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long)
    }

# ---------- Metrics ----------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': acc, 'f1': f1}

# ---------- Training Arguments ----------
training_args = TrainingArguments(
    output_dir='./idiom_classifier',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=2000,
    eval_strategy="steps",
    eval_steps=250,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="tensorboard",
    dataloader_pin_memory=False,          # can help with memory
    local_rank=-1,                         # ensure no distributed training
    ddp_find_unused_parameters=False       # avoid DDP
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

# ---------- Train! ----------
print("Starting training...")
trainer.train()

# Optionally save the final model
trainer.save_model("./idiom_classifier/final")