import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, DatasetDict
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
from typing import List, Dict
import librosa
import torch
import numpy as np

from .constants import DATA_ROOT, FOLDER_NAMES, SPLITS
from .utils import get_audio_duration, get_idiom_name_by_folder


class RomanshDataset(HFDataset):
    DATA_ROOT = DATA_ROOT

    def __init__(self, manifest: List[Dict], processor: WhisperProcessor, max_input_length: int = 30, language: str = "it", task: str = "transcribe"):
        self.manifest = manifest
        self.processor = processor
        self.max_input_length = max_input_length
        self.language = language
        self.task = task

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        audio, sr = librosa.load(item['audio_path'], sr=16000)
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
        if input_features.shape[0] > self.max_input_length * 100:
            input_features = input_features[:self.max_input_length * 100]
        labels = self.processor(
            text=item['transcript'],
            language=self.language,
            task=self.task,
        ).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
            "audio_path": item['audio_path']
        }

    @classmethod
    def aggregate_corpus_stats(cls) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Aggregate statistics for the entire Romansh corpus using cls.DATA_ROOT.
        Returns (duration_df, count_df, word_df).
        """
        duration_stats = defaultdict(lambda: defaultdict(float))
        utterance_counts = defaultdict(lambda: defaultdict(int))
        word_counts = defaultdict(lambda: defaultdict(int))

        for idiom_file in FOLDER_NAMES:
            idiom_path = os.path.join(cls.DATA_ROOT, idiom_file)
            if not os.path.isdir(idiom_path):
                print(f"Missing idiom folder: {idiom_file}")
                continue

            idiom = get_idiom_name_by_folder(idiom_file)

            for split in SPLITS:
                tsv_path = os.path.join(idiom_path, f"{split}.tsv")
                clips_path = os.path.join(idiom_path, "clips")
                if not os.path.isfile(tsv_path):
                    continue

                df = pd.read_csv(tsv_path, sep="\t")
                num_words = df["sentence"].astype(str).apply(lambda x: len(x.split())).sum()
                word_counts[idiom][split] = num_words

                total_seconds = 0.0
                for rel_path in tqdm(df["path"], desc=f" {split}", leave=False):
                    audio_path = os.path.join(clips_path, rel_path)
                    total_seconds += get_audio_duration(audio_path)

                duration_stats[idiom][split] = total_seconds / 3600.0
                utterance_counts[idiom][split] = len(df)

        duration_df = pd.DataFrame(duration_stats).T.fillna(0)
        count_df = pd.DataFrame(utterance_counts).T.fillna(0)
        word_df = pd.DataFrame(word_counts).T.fillna(0)

        return duration_df, count_df, word_df

class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that dynamically pads the input features and labels
    for a sequence‑to‑sequence speech model like Whisper.
    """
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def load_all_data(split):

  if split not in SPLITS:
    raise Exception(f"Invalid split, must be one of: {SPLITS}")

  df = pd.DataFrame()

  for folder_name in FOLDER_NAMES:
    idiom_path = os.path.join(DATA_ROOT, folder_name)
    split_path = os.path.join(idiom_path, split + ".tsv")
    clips_path = os.path.join(idiom_path, "clips")
    idiom_name = get_idiom_name_by_folder(folder_name)

    if not os.path.exists(split_path):
      raise Exception(f"File {split_path} not found")
    
    df_idiom = pd.read_csv(split_path, sep='\t')
    df_idiom['audio_path'] = df_idiom['path'].apply(lambda p: os.path.join(clips_path, p))
    df_idiom['idiom'] = idiom_name
    df = pd.concat([df, df_idiom[['audio_path', 'sentence', 'idiom']]], ignore_index=True)
  
  return df

def build_dataset_dict(
    folder_names: List[str] = FOLDER_NAMES,
    data_root: str = DATA_ROOT
) -> DatasetDict:
    """
    Load all train and validation TSV files from all idioms,
    combine into a HuggingFace DatasetDict.
    """
    all_train = []
    all_val = []

    for idiom_folder in folder_names:
        idiom_path = os.path.join(data_root, idiom_folder)
        clips_path = os.path.join(idiom_path, "clips")
        idiom_name = get_idiom_name_by_folder(idiom_folder)

        train_path = os.path.join(idiom_path, "train.tsv")
        val_path = os.path.join(idiom_path, "validation.tsv")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            print(f"Skipping {idiom_folder}: missing train/validation.tsv")
            continue

        train_df = pd.read_csv(train_path, sep="\t")
        val_df = pd.read_csv(val_path, sep="\t")

        for _, row in train_df.iterrows():
            audio = os.path.join(clips_path, row["path"])
            if os.path.exists(audio):
                all_train.append({
                    "audio": audio,
                    "sentence": str(row["sentence"]),
                    "idiom": idiom_name
                })

        for _, row in val_df.iterrows():
            audio = os.path.join(clips_path, row["path"])
            if os.path.exists(audio):
                all_val.append({
                    "audio": audio,
                    "sentence": str(row["sentence"]),
                    "idiom": idiom_name
                })

    dataset_dict = DatasetDict({
        "train": Dataset.from_list(all_train),
        "validation": Dataset.from_list(all_val)
    })
    return dataset_dict


class OnTheFlyDataset(Dataset):          # ← inherits from torch.utils.data.Dataset
    def __init__(self, samples, feature_extractor, tokenizer,
                 language="it", task="transcribe", max_label_length=448):
        self.samples = samples            # list of dicts
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.language = language
        self.task = task
        self.max_label_length = max_label_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]          # idx is always an integer now
        audio_array, sr = librosa.load(item["audio"], sr=16000)
        input_features = self.feature_extractor(
            audio_array, sampling_rate=16000
        ).input_features[0]

        labels = self.tokenizer(
            item["sentence"],
            truncation=True,
            max_length=self.max_label_length,
            language=self.language,
            task=self.task,
        ).input_ids

        return {
            "input_features": input_features,
            "labels": labels
        }

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function that pads input features and labels exactly as original."""
    # Pad input features (shape: 80, time)
    input_features = [item["input_features"] for item in batch]
    max_feat_len = max(f.shape[-1] for f in input_features)
    padded_features = []
    for f in input_features:
        pad_len = max_feat_len - f.shape[-1]
        if pad_len > 0:
            padding = np.zeros((f.shape[0], pad_len))
            padded = np.concatenate([f, padding], axis=-1)
        else:
            padded = f
        padded_features.append(padded)

    # Pad labels with -100
    labels = [item["labels"] for item in batch]
    max_label_len = max(len(l) for l in labels)
    padded_labels = []
    for l in labels:
        pad_len = max_label_len - len(l)
        padded_labels.append(l + [-100] * pad_len)

    return {
        "input_features": torch.tensor(np.array(padded_features), dtype=torch.float32),
        "labels": torch.tensor(np.array(padded_labels), dtype=torch.long)
    }

def load_idiom_data(
    folder_names: List[str] = FOLDER_NAMES,
    data_root: str = DATA_ROOT
) -> tuple[List[Dict], List[Dict]]:
    """
    Load all train & validation TSV files from all idioms.
    Returns (train_samples, val_samples) where each is a list of dicts
    with keys: 'audio' (path), 'sentence', 'idiom'.
    """
    train_samples = []
    val_samples = []

    for idiom_folder in folder_names:
        idiom_path = os.path.join(data_root, idiom_folder)
        clips_path = os.path.join(idiom_path, "clips")
        idiom_name = get_idiom_name_by_folder(idiom_folder)

        train_path = os.path.join(idiom_path, "train.tsv")
        val_path = os.path.join(idiom_path, "validation.tsv")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            print(f"Skipping {idiom_folder}: missing train/validation.tsv")
            continue

        train_df = pd.read_csv(train_path, sep="\t")
        val_df = pd.read_csv(val_path, sep="\t")

        for _, row in train_df.iterrows():
            audio = os.path.join(clips_path, row["path"])
            if os.path.exists(audio):
                train_samples.append({
                    "audio": audio,
                    "sentence": str(row["sentence"]),
                    "idiom": idiom_name
                })

        for _, row in val_df.iterrows():
            audio = os.path.join(clips_path, row["path"])
            if os.path.exists(audio):
                val_samples.append({
                    "audio": audio,
                    "sentence": str(row["sentence"]),
                    "idiom": idiom_name
                })

    return train_samples, val_samples
 