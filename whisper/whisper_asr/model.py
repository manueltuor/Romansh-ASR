import torch
import types
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

def make_causal_forward(original_forward):
    """Create a forward wrapper that inserts a causal mask, calling original_forward."""
    def causal_forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        # Only apply causal mask for self‑attention (key_value_states is None)
        if key_value_states is None:
            bsz, tgt_len, _ = hidden_states.shape
            causal_mask = torch.triu(
                torch.ones(
                    (tgt_len, tgt_len),
                    device=hidden_states.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            if attention_mask is not None:
                attention_mask = (
                    attention_mask[:, None, None, :] * causal_mask[None, None, :, :]
                )
            else:
                attention_mask = causal_mask[None, None, :, :]

        # Call the original forward of *this specific* layer
        return original_forward(
            hidden_states,
            key_value_states=key_value_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
    return causal_forward


def apply_causal_attention_mask(model):
    """
    Replace every encoder self‑attention layer with a causally‑masked version.

    The model is modified in‑place.  After calling this function, the encoder
    will only attend to previous time steps (no look‑ahead), making it suitable
    for streaming / low‑latency ASR.

    Args:
        model: a HuggingFace WhisperForConditionalGeneration instance.
    """
    encoder = model.model.encoder
    for layer in encoder.layers:
        attn = layer.self_attn
        attn.forward = types.MethodType(make_causal_forward(attn.forward), attn)