import torch
from tqdm import tqdm

def batch_translate(model, processor, sources, target_lang="nl-NL", batch_size=4, num_samples=1):
    """
    Translates a list of strings using the provided model and processor.
    Uses batching and a progress bar (tqdm) for CPU efficiency and feedback.
    
    Returns list of {source, translation, likelihood} dictionaries.
    Likelihood is calculated as the average log-probability per generated token.
    """
    all_results = []
    
    # tqdm creates the progress bar based on the number of batches
    for i in tqdm(range(0, len(sources), batch_size), desc="Translating", unit="batch"):
        batch_texts = sources[i : i + batch_size]
        
        # format the batch for the TranslateGemma chat template
        batch_messages = [
            [{"role": "user", "content": [{"type": "text", "source_lang_code": "en", 
              "target_lang_code": target_lang, "text": txt}]}]
            for txt in batch_texts
        ]

        # tokenize the current batch
        inputs = processor.apply_chat_template(
            batch_messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_dict=True, 
            return_tensors="pt",
            padding=True
        ).to(model.device)

        # run inference
        with torch.inference_mode():
            # produce num_samples outputs per source
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=False,
                num_beams=num_samples,
                num_return_sequences=num_samples,
                return_dict_in_generate=True,
                output_scores=True
            )

        # decode the output (skipping the input tokens)
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_len:]
        decoded = processor.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # compute transition scores (raw log-probs for each generated token)
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        # mask out padding tokens so they don't affect the average likelihood
        mask = (generated_tokens != processor.tokenizer.pad_token_id)
        
        # sum the log-probs of valid tokens and divide by the number of valid tokens
        sum_scores = torch.sum(transition_scores * mask, dim=-1)
        token_counts = mask.sum(dim=-1)
        
        # calculate average log-probability per token
        # use a small epsilon (1e-9) to avoid zero division
        likelihoods = (sum_scores / (token_counts + 1e-9)).tolist()
        
        # make everything the right size and index etc
        for idx, src_text in enumerate(batch_texts):
            for s_idx in range(num_samples):
                global_idx = idx * num_samples + s_idx
                all_results.append({
                    "source": src_text,
                    "translation": decoded[global_idx].strip(),
                    "likelihood": likelihoods[global_idx]
                })
        
    return all_results