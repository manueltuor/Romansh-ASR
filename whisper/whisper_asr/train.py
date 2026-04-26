import os
from dataclasses import dataclass
from typing import Callable, Optional
import torch
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments
from tqdm import tqdm

@dataclass
class TrainingConfig:
    output_dir: str = "./models/whisper-small-rm"
    num_epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    fp16: bool = False            # set True if using mixed precision (requires GPU)

class Trainer:
    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        processor: WhisperProcessor,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: TrainingConfig,
        compute_wer_fn: Callable[[str, str], float],   # & returns a float in [0,1]
    ):
        self.model = model
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.compute_wer_fn = compute_wer_fn

        self.device = next(model.parameters()).device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        # Optional: learning rate scheduler (linear warmup + linear decay)
        total_steps = len(train_dataloader) // config.gradient_accumulation_steps * config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_steps / total_steps if total_steps > 0 else 0.1,
        ) if config.warmup_steps > 0 else None

    def train(self):
        global_step = 0
        for epoch in range(self.config.num_epochs):
            self.model.train()
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            total_loss = 0.0

            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()
                total_loss += loss.item()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / (self.config.logging_steps * self.config.gradient_accumulation_steps)
                        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})
                        total_loss = 0.0

                    # Evaluation
                    if global_step % self.config.eval_steps == 0:
                        self.evaluate(global_step)

                    # Save checkpoint
                    if global_step % self.config.save_steps == 0:
                        self.save_checkpoint(global_step)

            # Evaluate at end of epoch (optional)
            self.evaluate(f"epoch_{epoch+1}")

    def evaluate(self, step):
        self.model.eval()
        total_wer = 0.0
        n_samples = 0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                generated_ids = self.model.generate(batch["input_features"])
                preds = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                # Decode references while ignoring -100
                labels = batch["labels"]
                labels[labels == -100] = self.processor.tokenizer.pad_token_id
                refs = self.processor.batch_decode(labels, skip_special_tokens=True)

                for pred, ref in zip(preds, refs):
                    wer = self.compute_wer_fn(ref, pred)
                    total_wer += wer
                    n_samples += 1

        avg_wer = (total_wer / n_samples) * 100 if n_samples > 0 else 0.0
        print(f"Step {step} – Validation WER: {avg_wer:.2f}%")
        self.model.train()

    def save_checkpoint(self, step):
        os.makedirs(self.config.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        self.model.save_pretrained(checkpoint_path)
        self.processor.save_pretrained(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

def get_training_args(output_dir: str, **overrides) -> Seq2SeqTrainingArguments:
    """Return the exact training arguments from the original notebook."""
    defaults = dict(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=1000,
        max_steps=10000,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        ddp_find_unused_parameters=None,
        gradient_checkpointing=False
    )
    defaults.update(overrides)
    return Seq2SeqTrainingArguments(**defaults)