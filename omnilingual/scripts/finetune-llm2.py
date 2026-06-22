#!/usr/bin/env python3
"""
Omnilingual LLM LoRA Fine‑Tuning – universal LoRA wrapper.
Trains on all Parquet data with LoRA, no length skipping.
"""

import os, sys, io, yaml, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from fairseq2.models import load_model
from fairseq2.data.tokenizers import load_tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import soundfile as sf, torchaudio

# ----------------------------------------------------------------------
# 1. Environment & paths
# ----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SUBMODULE_ROOT = ROOT_DIR / "omnilingual_asr"

sys.path.insert(0, str(ROOT_DIR))
from omnilingual_asr.utils import get_best_gpu
best_gpu = get_best_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
print(f"Using GPU {best_gpu}")

for var in ["OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","TQDM_DISABLE"]:
    os.environ[var] = "1"

# ----------------------------------------------------------------------
# 2. Universal LoRA wrapper (works with ANY module that has weight)
# ----------------------------------------------------------------------
class LoRAWrapper(nn.Module):
    def __init__(self, original_module, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.original = original_module
        for p in self.original.parameters():
            p.requires_grad = False

        weight = original_module.weight
        self.out_features, self.in_features = weight.shape
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, self.in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        self.lora_dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        base_out = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

def apply_lora_to_attention(model, r=8, alpha=16, dropout=0.1):
    """Wraps every module whose name ends with 'q_proj' or 'v_proj' with LoRAWrapper."""
    replace_list = []
    for name, module in model.named_modules():
        if name.endswith('q_proj') or name.endswith('v_proj'):
            replace_list.append((name, module))

    if not replace_list:
        raise RuntimeError("No q_proj/v_proj layers found. Cannot apply LoRA.")

    for name, module in replace_list:
        parent = model
        parts = name.split('.')
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRAWrapper(module, r=r, alpha=alpha, dropout=dropout))

    for param in model.parameters():
        param.requires_grad = False
    for n, param in model.named_parameters():
        if 'lora_' in n:
            param.requires_grad = True

    print(f"LoRA applied to {len(replace_list)} layers (q_proj/v_proj).")
    return model

# ----------------------------------------------------------------------
# 3. Load config & model
# ----------------------------------------------------------------------
CONFIG_FILE = SUBMODULE_ROOT / "workflows/recipes/wav2vec2/asr/configs/romansh-llm-finetune.yaml"
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]
OUTPUT_DIR = Path("../models/omnilingual-llm-rm-1b-lora")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading base model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = load_model(MODEL_NAME, device=device)
tokenizer = load_tokenizer(config["tokenizer"]["name"])
pad_idx = tokenizer.vocab_info.pad_idx

print("Applying LoRA...")
base_model = apply_lora_to_attention(base_model, r=8, alpha=16, dropout=0.1)
base_model.to(device)
base_model.train()

# ----------------------------------------------------------------------
# 4. Minimal layout class + batch class
# ----------------------------------------------------------------------
class TargetLayout:
    """Wraps target sequence lengths, providing both seq_lens and seq_lens_pt."""
    def __init__(self, seq_lens_tensor):
        self.seq_lens = seq_lens_tensor          # used for indexing
        self.seq_lens_pt = seq_lens_tensor       # used for arithmetic (same values)

class SimpleBatch:
    """Mimics the batch object expected by Wav2Vec2LlamaModel."""
    def __init__(self, src_tokens, prev_output_tokens, lang="roh_Latn_surs1244"):
        self.source_seqs = src_tokens
        self.source_seq_lens = torch.full((src_tokens.size(0),), src_tokens.size(1), dtype=torch.long)
        self.target_seqs = prev_output_tokens
        self.target_seq_lens = torch.full((prev_output_tokens.size(0),), prev_output_tokens.size(1), dtype=torch.long)
        self.example = {"lang": [lang] * src_tokens.size(0)}

    def as_target_input(self):
        """Return target tokens and a layout with seq_lens."""
        return self.target_seqs, TargetLayout(self.target_seq_lens)

# ----------------------------------------------------------------------
# 5. Dataset – raw audio, tokenized text (memory‑safe)
# ----------------------------------------------------------------------
import pyarrow.dataset as pa_ds, pyarrow.compute as pc

ds = pa_ds.dataset(
    "/local/scratch/matuor/parquet-dataset/rm-dataset/version=0",
    partitioning="hive"
)
train_df = ds.to_table(filter=pc.field("split") == "train").to_pandas()
valid_df = ds.to_table(filter=pc.field("split") == "dev").to_pandas()

class RomanshParquetDataset(Dataset):
    def __init__(self, df, tokenizer, max_audio_len=320_000):   # 20 seconds
        self.df = df
        self.tokenizer = tokenizer
        self.max_audio_len = max_audio_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ogg_bytes = row["audio_bytes"]
        audio, sr = sf.read(io.BytesIO(ogg_bytes), dtype="float32")
        if sr != 16000:
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio), orig_freq=sr, new_freq=16000
            ).numpy()
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if len(audio) > self.max_audio_len:
            audio = audio[:self.max_audio_len]

        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        encoder = self.tokenizer.create_encoder()
        token_ids = [t.item() for t in encoder(row["text"])]

        return audio_tensor, torch.tensor(token_ids, dtype=torch.long), len(audio_tensor), len(token_ids)

def collate_batch(batch):
    audios, tokens, a_len, t_len = zip(*batch)
    max_a = max(a_len)
    max_t = max(t_len)
    padded_audios = torch.zeros((len(batch), max_a))
    padded_tokens = torch.full((len(batch), max_t), pad_idx, dtype=torch.long)
    for i, (a, t) in enumerate(zip(audios, tokens)):
        padded_audios[i, :len(a)] = a
        padded_tokens[i, :len(t)] = t
    return padded_audios.to(device), padded_tokens.to(device), torch.tensor(a_len), torch.tensor(t_len)

train_dataset = RomanshParquetDataset(train_df, tokenizer, max_audio_len=320_000)
valid_dataset = RomanshParquetDataset(valid_df, tokenizer, max_audio_len=320_000)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_batch, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=collate_batch, num_workers=0)

# ----------------------------------------------------------------------
# 6. Training loop (batch size 2, grad accum 16)
# ----------------------------------------------------------------------
optimizer = AdamW(base_model.parameters(), lr=float(config["optimizer"]["config"]["lr"]))
scaler = GradScaler(device=device.type, enabled=torch.cuda.is_available())
num_epochs = 10
grad_accum_steps = 16
save_every = 500
valid_every = 500
total_steps = 0

from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader) // grad_accum_steps)

print("Starting training...")
for epoch in range(num_epochs):
    base_model.train()
    train_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        audio, labels, a_len, t_len = batch

        with autocast(device_type=device.type, dtype=torch.float16):
            batch_obj = SimpleBatch(audio, labels[:, :-1], lang="roh_Latn_surs1244")
            outputs = base_model(batch=batch_obj)
            loss = outputs

        scaler.scale(loss / grad_accum_steps).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        train_loss += loss.item()
        total_steps += 1

        if total_steps % valid_every == 0:
            base_model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for v_batch in tqdm(valid_loader, desc="Validating"):
                    v_audio, v_labels, _, _ = v_batch
                    with autocast(device_type=device.type, dtype=torch.float16):
                        v_batch_obj = SimpleBatch(v_audio, v_labels[:, :-1], lang="roh_Latn_surs1244")
                        v_outputs = base_model(batch=v_batch_obj)
                        valid_loss += v_outputs.item()
            valid_loss /= len(valid_loader)
            print(f"Step {total_steps}: train loss {train_loss/valid_every:.4f}, valid loss {valid_loss:.4f}")
            train_loss = 0.0
            base_model.train()

        if total_steps % save_every == 0:
            lora_state = {k: v for k, v in base_model.state_dict().items() if 'lora_' in k}
            save_path = OUTPUT_DIR / f"lora_step_{total_steps}.pt"
            torch.save(lora_state, save_path)
            print(f"Saved LoRA adapter to {save_path}")

print("Training complete. Final LoRA adapter saved.")