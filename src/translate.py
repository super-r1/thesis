import torch
from tqdm import tqdm

def batch_translate(model, processor, sources, target_lang="nl-NL", batch_size=4):
    """
    Translates a list of strings using the provided model and processor.
    Uses batching and a progress bar (tqdm) for CPU efficiency and feedback.
    """
    all_translations = []
    
    # tqdm creates the progress bar based on the number of batches
    for i in tqdm(range(0, len(sources), batch_size), desc="Translating", unit="batch"):
        batch_texts = sources[i : i + batch_size]
        
        # Format the batch for the TranslateGemma chat template
        batch_messages = [
            [{"role": "user", "content": [{"type": "text", "source_lang_code": "en", 
              "target_lang_code": target_lang, "text": txt}]}]
            for txt in batch_texts
        ]

        # Tokenize the current batch
        inputs = processor.apply_chat_template(
            batch_messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_dict=True, 
            return_tensors="pt",
            padding=True
        ).to(model.device)

        # Run inference
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=False
            )

        # Decode the output (skipping the input tokens)
        decoded = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Clean up and add to the final list
        all_translations.extend([t.strip() for t in decoded])
        
    return all_translations