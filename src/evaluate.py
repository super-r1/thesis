import torch
import pandas as pd
import os
from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer, T5TokenizerFast
from tqdm import tqdm

from . import metricx_models
from .config import device
from .config import DATA_DIR

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

def analyze_hypos(in_csv, lang, out_folder=f"{DATA_DIR}/translate_again", remove_canary=True):
    """
    takes translation results csv and finds the most likely and highest comet translations
    returns csv with source, hypo_a (most likely), hypo_b (highest comet), comet score difference
    """

    df = pd.read_csv(in_csv)

    # give rank to translations for each source for likelihood, comet and metricx
    df['likelihood_rank'] = df.groupby('source')['likelihood'].rank(ascending=False, method='first').astype(int)
    df['comet_rank'] = df.groupby('source')['comet22_score'].rank(ascending=False, method='first').astype(int)
    df['metricx_rank'] = df.groupby('source')['metricx24_score'].rank(ascending=True, method='first').astype(int)

    # first 5 rows are canary
    if remove_canary:
        df = df.iloc[5:]

    # make 2 dfs with the best translations according to likelihood and comet
    model_best_df = df[df['likelihood_rank'] == 1].copy()
    comet_best_df = df[df['comet_rank'] == 1].copy()

    # merge into one df on source
    vs_df = pd.merge(
        model_best_df[['source', 'translation', 'comet22_score']], 
        comet_best_df[['source', 'translation', 'comet22_score', 'target']], 
        on='source', 
        suffixes=('_model', '_comet')
    )

    # rename hypothesis columns
    vs_df = vs_df.rename(columns={
        'translation_model': 'hypo_a', 
        'translation_comet': 'hypo_b'})

    # calculate comet score difference
    vs_df['comet_diff'] = vs_df['comet22_score_comet'] - vs_df['comet22_score_model']

    # save output csv
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{lang}.csv")
    vs_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")