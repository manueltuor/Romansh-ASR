"""
Romansh ASR Evaluation Script for Omnilingual Model.

This script loads the Romansh test dataset, batches audio samples through 
the ASRInferencePipeline using a specified language code configuration, 
post-processes the predicted transcriptions, and computes dialect-specific 
error metrics (WER/CER) to generate a comprehensive evaluation summary.

This script is for existing Omnilingual models, if you want to evaluate your own
model from a checkpoint refer to evaluate-checkpoint.py. You can change the
`MODEL_CARD` constant to evaluate a different model.
"""

import torch
from tqdm import tqdm
from omnilingual_asr.data import load_all_data
from omnilingual_asr.evaluate import add_metrics_columns, idiom_summary, print_evaluation_summary, show_examples
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs
from omnilingual_asr.utils import get_best_gpu, normalize_romansh_text

# set constants
MODEL_CARD = "omniASR_CTC_1B_v2"
LANGUAGE_CODE = "roh_Latn_surs1244"
BATCH_SIZE = 8
best_gpu = get_best_gpu()
DEVICE = f"cuda:{best_gpu}" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}, Lang supported: {LANGUAGE_CODE in supported_langs}")

# load test data
df_test = load_all_data("test")
print(f"Loaded {len(df_test)} samples")

# spin up pipeline
pipeline = ASRInferencePipeline(model_card=MODEL_CARD, device=DEVICE)
audio_paths = df_test["audio_path"].tolist()
transcriptions = []

# transcribe test set
for i in tqdm(range(0, len(audio_paths), BATCH_SIZE)):
    batch = audio_paths[i:i+BATCH_SIZE]
    try:
        results = pipeline.transcribe(batch, lang=[LANGUAGE_CODE]*len(batch), batch_size=len(batch))
        transcriptions.extend(results)
    except Exception as e:
        print(f"Batch error at {i}: {e}")
        transcriptions.extend([""] * len(batch))

df_test["omnilingual_transcription"] = transcriptions

# normalize transcriptions
df_test['omnilingual_transcription'] = df_test['omnilingual_transcription'].apply(normalize_romansh_text)
# compute wer and cer
df_test = add_metrics_columns(df_test, "sentence", "omnilingual_transcription")
# print summary
summary = idiom_summary(df_test)
print_evaluation_summary(summary)

# show some transcription examples
show_examples(df_test, "omnilingual_transcription")