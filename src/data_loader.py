import pandas as pd
from datasets import load_dataset
import os
import requests
import tarfile
from dotenv import load_dotenv

DATA_DIR = os.path.abspath("../../data")
FLORES_DIR = os.path.join(DATA_DIR, "flores200_dataset")
DOWNLOAD_URL = "https://tinyurl.com/flores200dataset"
TAR_PATH = os.path.join(DATA_DIR, "flores200.tar.gz")

def load_wmt_data(lang, limit):
    ds = load_dataset("google/wmt24pp", lang)
    df = pd.DataFrame(ds['train'])
    
    sources = df['source'].tolist()[:limit]
    targets = df['target'].tolist()[:limit]
    
    return sources, targets

def load_flores_data(limit):
    if not os.path.exists(FLORES_DIR):
        print("FLORES dataset not found. Downloading...")
        os.makedirs(DATA_DIR, exist_ok=True)
        
        response = requests.get(DOWNLOAD_URL, stream=True)
        if response.status_code == 200:
            with open(TAR_PATH, 'wb') as f:
                f.write(response.content)
            print("Download complete. Extracting...")
            
            with tarfile.open(TAR_PATH, "r:gz") as tar:
                tar.extractall(path=DATA_DIR)
            
            os.remove(TAR_PATH)
            print("Extraction finished.")
        else:
            raise Exception(f"Failed to download dataset. Status code: {response.status_code}")

    eng_path = os.path.join(FLORES_DIR, "dev", "eng_Latn.dev")
    nld_path = os.path.join(FLORES_DIR, "dev", "nld_Latn.dev")

    with open(eng_path, "r", encoding="utf-8") as f:
        english_sentences = [line.strip() for line in f]

    with open(nld_path, "r", encoding="utf-8") as f:
        dutch_sentences = [line.strip() for line in f]

    sources = english_sentences[:limit]
    targets = dutch_sentences[:limit]

    return sources, targets

def load_bouquet_data(limit):
    load_dotenv()
    access_token = os.getenv("HF_TOKEN")

    bouquet = load_dataset("facebook/bouquet", "nld_Latn", token=access_token)

    bouquet_df = bouquet["dev"].to_pandas()

    bouquet_df = bouquet_df[['tgt_text', 'src_text']].rename(
        columns={
            'tgt_text': 'english',
            'src_text': 'dutch'
        }
    )

    sources = bouquet_df['english'].tolist()[:limit]
    targets = bouquet_df['dutch'].tolist()[:limit]

    return sources, targets