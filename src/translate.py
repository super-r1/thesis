import torch
from tqdm import tqdm
import torch.nn.functional as F

def batch_translate(model, processor, sources, target_lang="nl-NL", batch_size=4, num_samples=1, likelihood_mode="generation"):
    """
    Translates a list of strings using the provided model and processor.
    Uses batching and a progress bar (tqdm) for CPU efficiency and feedback.
    
    Returns list of {source, translation, likelihood} dictionaries.
    Likelihood is calculated as the average log-probability per generated token.
    """
    all_results = []
    
    # tqdm creates the progress bar based on the number of batches
    for i in tqdm(range(0, len(sources), batch_size), desc=f"Translating ({likelihood_mode})", unit="batch"):
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
            # settings depend on likelihood calculation mode
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256,
                do_sample=(likelihood_mode == "reeval"), 
                top_p=0.9 if likelihood_mode == "reeval" else None,
                temperature=0.7 if likelihood_mode == "reeval" else None,
                num_beams=num_samples if likelihood_mode == "generation" else 1,
                num_return_sequences=num_samples,
                return_dict_in_generate=True,
                output_scores=True,
                #no_repeat_ngram_size=3,
                length_penalty=0.8,
                early_stopping=True
            )

        # decode the output (skipping the input tokens)
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_len:]
        decoded = processor.batch_decode(generated_tokens, skip_special_tokens=True)
        
        if likelihood_mode == "generation":
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

        else:
            # run another forward pass to calculate likelihoods using teacher forcing
            # smaller batches for memory efficiency
            likelihoods = []
            reeval_sub_batch_size = 2 
            
            with torch.inference_mode():
                # repeat inputs so they align with num_samples
                expanded_input_ids = inputs.input_ids.repeat_interleave(num_samples, dim=0)
                expanded_attention_mask = inputs.attention_mask.repeat_interleave(num_samples, dim=0)

                # process in smaller chunks (to save memory)
                for j in range(0, expanded_input_ids.shape[0], reeval_sub_batch_size):
                    sub_input_ids = expanded_input_ids[j : j + reeval_sub_batch_size]
                    sub_attn_mask = expanded_attention_mask[j : j + reeval_sub_batch_size]
                    sub_gen_tokens = generated_tokens[j : j + reeval_sub_batch_size]

                    # concatenate prompt and generated tokens
                    full_input_ids = torch.cat([sub_input_ids, sub_gen_tokens], dim=-1)
                    
                    # create full attention mask
                    gen_mask = (sub_gen_tokens != processor.tokenizer.pad_token_id).long()
                    full_attention_mask = torch.cat([sub_attn_mask, gen_mask], dim=-1)

                    # run forward pass
                    outputs_tf = model(
                        input_ids=full_input_ids,
                        attention_mask=full_attention_mask,
                        return_dict=True
                    )

                    # shift logits and labels to align
                    logits = outputs_tf.logits[:, input_len - 1 : -1, :]
                    log_probs = F.log_softmax(logits, dim=-1)

                    sub_token_log_probs = torch.gather(
                        log_probs,
                        dim=-1,
                        index=sub_gen_tokens.unsqueeze(-1)
                    ).squeeze(-1)

                    # mask and average for this sub-batch
                    sub_mask = (sub_gen_tokens != processor.tokenizer.pad_token_id)
                    sub_sum_scores = torch.sum(sub_token_log_probs * sub_mask, dim=-1)
                    sub_token_counts = sub_mask.sum(dim=-1)
                    
                    sub_likelihoods = (sub_sum_scores / (sub_token_counts + 1e-9)).tolist()
                    likelihoods.extend(sub_likelihoods)
        
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