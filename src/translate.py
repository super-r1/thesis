import torch

def batch_translate(model, processor, sources, target_lang="nl-NL"):
    batch_messages = [
        [{"role": "user", "content": [{"type": "text", "source_lang_code": "en", 
          "target_lang_code": target_lang, "text": txt}]}]
        for txt in sources
    ]

    inputs = processor.apply_chat_template(
        batch_messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_dict=True, 
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    decoded = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    return [t.strip() for t in decoded]