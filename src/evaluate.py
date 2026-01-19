import torch
from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer, T5TokenizerFast
from tqdm import tqdm

from . import metricx_models
from .config import device


def comet22_eval(sources, translations, references):
    """
    calculate comet-22 scores (with reference)
    """
    print("Evaluating with COMET-22...")
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
    
    comet_data = [
        {"src": s, "mt": t, "ref": r} 
        for s, t, r in zip(sources, translations, references)
    ]
    
    num_gpus = 1 if device == "cuda" else 0
    return comet_model.predict(comet_data, batch_size=16, gpus=num_gpus)


def metricx24_eval(sources, translations, model_name="google/metricx-24-hybrid-large-v2p6", batch_size=4):
    """
    calculate metricX-24 QE scores (without reference) in batches
    based on https://github.com/google-research/metricx/blob/main/metricx24/predict.py
    """
    print(f"Loading MetricX-24 model: {model_name}...")
    
    tokenizer = T5TokenizerFast.from_pretrained("google/mt5-large", legacy=False)
    
    model = metricx_models.MT5ForRegression.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )
    model.to(device)
    model.eval()

    all_scores = []

    print(f"Running MetricX inference on {len(sources)} sentences in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(sources), batch_size)):
        batch_src = sources[i:i+batch_size]
        batch_mt = translations[i:i+batch_size]
        
        # format input text
        input_texts = [
            f"source: {s} candidate: {m}" for s, m in zip(batch_src, batch_mt)
        ]
        
        # tokenize batch and add padding to the longest sequence in this batch
        inputs = tokenizer(
            input_texts,
            max_length=1024,
            truncation=True,
            padding="longest",
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors="pt"
        ).to(device)

        # get predictions
        with torch.no_grad():
            output = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            batch_predictions = output.predictions.flatten().tolist()
            all_scores.extend(batch_predictions)

    return all_scores