import pandas as pd
from datasets import load_dataset

def load_wmt_data(lang, limit):
    ds = load_dataset("google/wmt24pp", lang)
    df = pd.DataFrame(ds['train'])
    
    sources = df['source'].tolist()[:limit]
    targets = df['target'].tolist()[:limit]
    
    return sources, targets