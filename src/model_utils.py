import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from .config import model_id, HF_TOKEN, device

def load_model_and_processor(checkpoint_path=None):
    #processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
    processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN, use_fast=True)
    
    # padding for batch processing
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.pad_token = processor.tokenizer.eos_token 

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, 
        dtype=torch.bfloat16,
        device_map=device, 
        token=HF_TOKEN
    )

    # load checkpoint if provided
    if checkpoint_path:
        print(f"Loading adapter checkpoint from {checkpoint_path}...")
        model = PeftModel.from_pretrained(model, checkpoint_path)

        # MAYBE ADD LATER
        # model = model.merge_and_unload()
        
    return model, processor