import pandas as pd
from datasets import load_dataset

def load_wmt_data(lang, limit=False):
    ds = load_dataset("google/wmt24pp", lang)
    df = pd.DataFrame(ds['train'])
    
    if limit:
        # only use part of data
        sources = df['source'].tolist()[:limit]
        targets = df['target'].tolist()[:limit]
    else:
        sources = df['source'].tolist()
        targets = df['target'].tolist()
    
    return sources, targets