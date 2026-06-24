import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from .utils import get_best_gpu

def load_model_and_processor(
    model_name: str = "openai/whisper-medium",
    device: str = None,
    language: str = "it",
    task: str = "transcribe",
    use_forced_decoder_ids: bool = True,
):
    """
    Load model, feature extractor, tokenizer, processor.

    Args:
        model_name (str): Hugging Face repository ID or local directory path.
        device (str | None): Target compute device. If None, resolves dynamically.
        language (str): Target language identifier token proxy (e.g., "it" for Italian).
        task (str): Downstream task generation mode, typically "transcribe" or "translate".
        use_forced_decoder_ids (bool): If True, forces generation to start with language/task tokens.

    Returns:
        tuple: (model, feature_extractor, tokenizer, processor) fully initialized.
    """
    if device is None:
        device = f"cuda:{get_best_gpu()}" if torch.cuda.is_available() else "cpu"

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, task=task)
    processor = WhisperProcessor.from_pretrained(model_name, task=task)

    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    # Must be False to allow gradient checkpointing during fine-tuning
    model.config.use_cache = False

    if use_forced_decoder_ids:
        # Explicitly hardcodes the target language token to stop Whisper from guessing
        forced_ids = processor.get_decoder_prompt_ids(
            language=language, task=task
        )
        model.config.forced_decoder_ids = forced_ids
        model.config.suppress_tokens = []

    model.to(device)
    return model, feature_extractor, tokenizer, processor