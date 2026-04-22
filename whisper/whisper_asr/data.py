import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import WhisperProcessor
from typing import List, Dict
import librosa

from .constants import DATA_ROOT, FOLDER_NAMES, SPLITS
from .utils import get_audio_duration, get_idiom_name_by_folder


class RomanshDataset(Dataset):
    DATA_ROOT = DATA_ROOT

    def __init__(self, manifest: List[Dict], processor: WhisperProcessor, max_input_length: int = 30):
        self.manifest = manifest
        self.processor = processor
        self.max_input_length = max_input_length

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        audio, sr = librosa.load(item['audio_path'], sr=16000)
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
        if input_features.shape[0] > self.max_input_length * 100:
            input_features = input_features[:self.max_input_length * 100]
        labels = self.processor(text=item['transcript']).input_ids
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