import torch
from tqdm import tqdm
import torch.nn.functional as F
from .config import MODEL_ID_MAP, LANG_MAP

def make_messages(text, target_lang, model_name, source_lang="en", mode="standard", hypo=None, again_instr="refine_instr"):
    """
    creates message with correct formatting, depending on model and mode
    """
    model_info = MODEL_ID_MAP[model_name]
    target_lang_code = LANG_MAP[target_lang]["gemma"]

    # translate-only gemma
    if model_info["type"] == "translate_only":
        user_text = text
        if mode == "again" and hypo:
            instr = model_info[again_instr]
            user_text = (
                f"Source: {text}\n"
                f"Initial Hypothesis: {hypo}\n"
                f"Instruction: {instr}"
            )
        
        return [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": user_text,
                "source_lang_code": source_lang,
                "target_lang_code": target_lang_code
            }]
        }]

    # general gemma
    if mode == "again":
        instr = model_info[again_instr]
        user_text = f"{instr}\n\nSource: {text}\nDraft: {hypo}"
    else:
        instr = model_info["system_instr"]
        user_text = f"{instr}\n\nTranslate this to {LANG_MAP[target_lang]['name']}: {text}"

    return [{
        "role": "user", 
        "content": [{"type": "text", "text": user_text}]
    }]

def batch_translate(model, processor, sources, model_name, lang_key="nl", 
                    batch_size=4, num_samples=1, mode="standard", hypos=None):
    """
    Translates a list of strings using the provided model and processor.
    Uses batching and a progress bar (tqdm) for CPU efficiency and feedback.
    
    Returns list of {source, translation, likelihood} dictionaries.
    Likelihood is calculated as the average log-probability per generated token.
    """
    all_results = []
    
    # tqdm creates the progress bar based on the number of batches
    for i in tqdm(range(0, len(sources), batch_size), desc="Translating in batches", unit="batch"):
        batch_texts = sources[i : i + batch_size]

        # create prompt message (using helper function)
        batch_hypos = hypos[i : i + batch_size] if hypos is not None else [None] * len(batch_texts)
        batch_messages = []
        for txt, h in zip(batch_texts, batch_hypos):
            conversation = make_messages(txt, lang_key, model_name, mode=mode, hypo=h)
            batch_messages.append(conversation)

        # tokenize the current batch
        # print(f"DEBUG: {batch_messages[0]}")
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
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                num_return_sequences=num_samples,
                #temperature=None,
                #num_beams=1,
                #num_beam_groups=num_samples,
                #trust_remote_code=True,
                #diversity_penalty=1.0,
                #length_penalty=0.8,
                #early_stopping=True
            )

        # decode the output (skipping the input tokens)
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded = processor.batch_decode(generated_tokens, skip_special_tokens=True)
        
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
                
                # create attention mask over full sequence
                gen_mask = (sub_gen_tokens != processor.tokenizer.pad_token_id).long()
                full_attention_mask = torch.cat([sub_attn_mask, gen_mask], dim=-1)

                # run forward pass with teacher forcing
                outputs_tf = model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    return_dict=True
                )

                # shift logits and labels to align (only use logits for generated tokens)
                gen_len = sub_gen_tokens.size(1)
                logits = outputs_tf.logits[:, input_len - 1 : input_len - 1 + gen_len, :]

                # softmax for normalized log-probs
                log_probs = F.log_softmax(logits, dim=-1)

                # gather log-probs for generated tokens (out of entire vocab)
                sub_token_log_probs = torch.gather(
                    log_probs,
                    dim=-1,
                    index=sub_gen_tokens.unsqueeze(-1)
                ).squeeze(-1)

                # sum mask to get amount of non-padding tokens
                sub_mask = (sub_gen_tokens != processor.tokenizer.pad_token_id)
                sub_sum_scores = torch.sum(sub_token_log_probs * sub_mask, dim=-1)
                sub_token_counts = sub_mask.sum(dim=-1)
                
                # average over probs for this sub-batch
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
