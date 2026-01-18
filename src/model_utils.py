import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from .config import model_id, HF_TOKEN, device

def load_model_and_processor():
    processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
    
    # padding for batch processing
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.pad_token = processor.tokenizer.eos_token 

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map=device, 
        token=HF_TOKEN
    )
    return model, processor