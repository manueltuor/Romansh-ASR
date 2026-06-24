import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
from jiwer import wer, cer
from typing import List
from collections import defaultdict
from evaluate import load as load_metric

class AudioDataset(Dataset):
    """Simple dataset that loads audio files and returns Whisper input features."""
    
    def __init__(self, audio_paths: List[str], processor: WhisperProcessor):
        self.audio_paths = audio_paths
        self.processor = processor

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        """
        Lazily reads an audio file from disk, resamples to 16kHz, and uses 
        the Whisper processor to extract its Log-Mel Spectrogram features.
        """
        audio_path = self.audio_paths[idx]
        # Load waveform arrays dynamically (resampled to Whisper standard 16,000Hz)
        audio, sr = librosa.load(audio_path, sr=16000)
        # Extract 80-channel filterbank log-mel features and strip the batch dimension
        input_features = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features[0]
        return input_features

def collate_audio_batch(batch: List[torch.Tensor]) -> torch.Tensor:
    """
    Pad features to the same length within a batch.
    
    Ensures that asymmetric temporal dimensions along the trailing axis (dim=-1) 
    are right-padded with zeros to establish an even tensor shape for parallel matrix operations.
    """
    max_len = max(f.shape[-1] for f in batch)
    padded_batch = []
    for features in batch:
        pad_len = max_len - features.shape[-1]
        if pad_len > 0:
            # Generate a zero matrix to pad the remaining temporal spectrum slice
            padding = torch.zeros((features.shape[0], pad_len))
            padded = torch.cat([features, padding], dim=-1)
        else:
            padded = features
        padded_batch.append(padded)
    # Stack individual 2D audio representations into a unified 3D tensor batch block
    return torch.stack(padded_batch)

def transcribe_whisper(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_paths: List[str],
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 225,
) -> List[str]:
    """
    Transcribe a list of audio files using a Whisper model.
    
    Args:
        model: Loaded WhisperForConditionalGeneration model.
        processor: Corresponding WhisperProcessor.
        audio_paths: List of paths to audio files.
        batch_size: Batch size for inference.
        device: Device to run inference on.
        max_length: Maximum generation length.
    
    Returns:
        List of transcription strings.
    """
    # Instantiate the data reading pipelines
    dataset = AudioDataset(audio_paths, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_audio_batch,
        num_workers=0,
    )
    
    # Target execution device and freeze layer normalizations/dropouts
    model.to(device)
    model.eval()
    
    transcriptions = []
    # Deactivate the autograd graph tracking to save memory overhead and compute time
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Transcribing (Whisper)"):
            batch = batch.to(device)
            # Generate token prediction IDs via sequential auto-regressive decoding
            predicted_ids = model.generate(
                batch,
                max_length=max_length,
                num_beams=1,    # Greedy decoding baseline setup
                task="transcribe"
            )
            # Convert vocabulary index IDs back into natural text strings
            texts = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            transcriptions.extend(texts)
    
    return transcriptions

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER) as a float.
    Raises ValueError if inputs are empty.
    """
    if not reference or not hypothesis:
        raise ValueError("Reference and hypothesis must be non-empty strings.")
    return wer(reference, hypothesis)

def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate (CER) as a float.
    Raises ValueError if inputs are empty.
    """
    if not reference or not hypothesis:
        raise ValueError("Reference and hypothesis must be non-empty strings.")
    return cer(reference, hypothesis)

def compute_metrics(pred, tokenizer):
    """
    Compute WER from Seq2SeqTrainer prediction output.
    Requires the tokenizer to decode tokens.
    """
    # Load dynamic evaluation file asset metrics via Hugging Face backend
    wer_metric = load_metric("wer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 masking values back into pad_token_ids.
    # This ensures the tokenizer skips decoding tracking locations that were padded 
    # out inside the sequence collator.
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Parse numerical logit slices down to natural human read layouts
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Calculate error metric and scale it up to a % score
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

def compute_idiom_results(references, transcriptions, idioms):
    """
    Compute overall and per-idiom WER/CER.
    Returns: summary_df, overall_wer, overall_cer, valid_pairs
    """
    idiom_data = defaultdict(lambda: {"refs": [], "hyps": []})
    valid_pairs = []
    # Filter out records where an alignment pair component is blank
    for ref, hyp, idiom in zip(references, transcriptions, idioms):
        if ref and hyp:
            valid_pairs.append((ref, hyp, idiom))
            idiom_data[idiom]["refs"].append(ref)
            idiom_data[idiom]["hyps"].append(hyp)

    # Compute macro global corpus validation performance figures
    all_refs = [p[0] for p in valid_pairs]
    all_hyps = [p[1] for p in valid_pairs]
    overall_wer = wer(all_refs, all_hyps)
    overall_cer = cer(all_refs, all_hyps)

    # Segment validation performances independently for every sub-dialect category
    per_idiom = []
    for idiom, d in idiom_data.items():
        if d["refs"]:
            i_wer = wer(d["refs"], d["hyps"])
            i_cer = cer(d["refs"], d["hyps"])
            per_idiom.append({
                "idiom": idiom, "samples": len(d["refs"]),
                "wer": i_wer, "cer": i_cer
            })

    # Wrap the categorized dictionary elements inside a clean pandas DataFrame format
    summary_df = pd.DataFrame(per_idiom)
    return summary_df, overall_wer, overall_cer, valid_pairs

def print_evaluation_results(summary_df, overall_wer, overall_cer, total_samples, valid_count):
    """
    Formats and prints a clean evaluation analysis matrix directly to stdout logs.
    """
    print("\n" + "="*50)
    print("OVERALL RESULTS")
    print("="*50)
    print(f"Total test samples: {total_samples}")
    print(f"Valid pairs: {valid_count}/{total_samples}")
    print(f"\nWord Error Rate (WER): {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    print(f"Character Error Rate (CER): {overall_cer:.4f} ({overall_cer*100:.2f}%)")

    print("\n" + "="*50)
    print("PER IDIOM RESULTS")
    print("="*50)
    for _, row in summary_df.iterrows():
        print(f"\n{row['idiom'].upper()}")
        print(f"  Samples: {row['samples']}")
        print(f"  WER: {row['wer']:.4f} ({row['wer']*100:.2f}%)")
        print(f"  CER: {row['cer']:.4f} ({row['cer']*100:.2f}%)")

    print("\n" + "="*50)
    print("SUMMARY TABLE")
    print("="*50)
    # Output the structured tabular data string layout
    print(summary_df.to_string(index=False))