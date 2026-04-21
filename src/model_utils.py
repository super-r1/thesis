import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
from .config import MODEL_ID_MAP, DEFAULT_MODEL, HF_TOKEN, device


class TextOnlyProcessor:
    """
    Wrapper to make text only model tokenizer behave like AutoProcessor
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)
    
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


def load_model_and_processor(checkpoint_path=None, model_name=DEFAULT_MODEL):
    model_info = MODEL_ID_MAP[model_name]
    model_id = model_info["id"]
    modality = model_info.get("modality", "multimodal")

    if modality == "text":
        # gemma1b is text-only
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=HF_TOKEN,
            use_fast=True,
        )

        # padding for batch processing
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            token=HF_TOKEN,
        )

        processor = TextOnlyProcessor(tokenizer)

    else:
        # multimodal models
        processor = AutoProcessor.from_pretrained(
            model_id,
            token=HF_TOKEN,
            use_fast=True,
        )

        processor.tokenizer.padding_side = "left"
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            token=HF_TOKEN,
        )

    # load checkpoint if provided
    if checkpoint_path:
        print(f"Loading adapter checkpoint from {checkpoint_path}...")
        model = PeftModel.from_pretrained(model, checkpoint_path)

    return model, processor
