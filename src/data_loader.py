import pandas as pd
from datasets import load_dataset
import os
import requests
import tarfile
import random
from dotenv import load_dotenv
import gzip
import json
from .config import LANG_MAP

# file paths and urls
from .config import DATA_DIR
FLORES_DIR = os.path.join(DATA_DIR, "flores200_dataset")
DOWNLOAD_URL = "https://tinyurl.com/flores200dataset"
TAR_PATH = os.path.join(DATA_DIR, "flores200.tar.gz")

MADLAD_DIR = os.path.join(DATA_DIR, "madlad400")
MADLAD_URL = "https://huggingface.co/datasets/allenai/MADLAD-400/resolve/main/data/nl/nl_clean_0000.jsonl.gz"
MADLAD_FILE = os.path.join(MADLAD_DIR, "nl_clean_0000.jsonl.gz")

def load_wmt_data(lang, limit=None):
    """
    Loads WMT dataset, filtering out canary lines
    """
    ds = load_dataset("google/wmt24pp", lang)
    df = pd.DataFrame(ds['train'])
    
    # discard canary rows
    df_clean = df[df['domain'] != 'canary']
    
    # optionally apply limit and return source+target sentences
    sources = df_clean['source'].tolist()[:limit]
    targets = df_clean['target'].tolist()[:limit]
    
    return sources, targets

def load_flores_data(lang, limit=None):
    """
    Loads FLORES-200 dataset for specified language, combining dev and devtest,
    and shuffles the resulting pairs.
    """

    # check if dataset is already downloaded
    if not os.path.exists(FLORES_DIR):
        print("FLORES dataset not found. Downloading...")
        os.makedirs(DATA_DIR, exist_ok=True)
        
        response = requests.get(DOWNLOAD_URL, stream=True)
        if response.status_code == 200:
            with open(TAR_PATH, 'wb') as f:
                f.write(response.content)
            print("Download complete. Extracting...")
            
            # extract tar file
            with tarfile.open(TAR_PATH, "r:gz") as tar:
                tar.extractall(path=DATA_DIR)
            
            os.remove(TAR_PATH)
            print("Extraction finished.")
        else:
            raise Exception(f"Failed to download dataset. Status code: {response.status_code}")

    # use both dev and devtest splits
    splits = ["dev", "devtest"]

    all_english = []
    all_tgt = []

    for split in splits:
        eng_path = os.path.join(FLORES_DIR, f"{split}", f"eng_Latn.{split}")
        tgt_path = os.path.join(FLORES_DIR, f"{split}", f"{lang}.{split}")

        if os.path.exists(eng_path) and os.path.exists(tgt_path):
            with open(eng_path, "r", encoding="utf-8") as f:
                all_english.extend([line.strip() for line in f])
            with open(tgt_path, "r", encoding="utf-8") as f:
                all_tgt.extend([line.strip() for line in f])
        else:
            print(f"Warning: Could not find files in {split} split.")

    # shuffle (combine so that pairs stay together)
    combined = list(zip(all_english, all_tgt))
    random.seed(42) # for reproducibility
    random.shuffle(combined)
    
    # back to 2 lists and optionally apply limit
    sources, targets = zip(*combined)
    sources = list(sources[:limit])
    targets = list(targets[:limit])

    return sources, targets

def load_translate_again_data(lang, data_folder=None, limit=None):
    """
    load translate-again data for specified language
    returns list of dicts with keys: source, hypo_a, hypo_b, comet_diff
            where each list entry represents one df row
    """

    if not data_folder:
        data_folder = f"{DATA_DIR}/translate_again"

    # load csv of this language
    csv_path = os.path.join(data_folder, f"{lang}.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping.")
        return []
    df = pd.read_csv(csv_path)

    # TBD look into why NA values exist
    df = df.fillna("")
    
    # optionally apply limit
    if limit:
        df = df.head(limit)

    # convert to list of dicts
    return df.to_dict(orient="records")

def load_bouquet_data(limit=None):
    """
    Loads BOUQUET dataset for English-Dutch language pair.
    """
    load_dotenv()
    access_token = os.getenv("HF_TOKEN")

    bouquet = load_dataset("facebook/bouquet", "nld_Latn", token=access_token)

    # get dev split and convert to pandas dataframe
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

def load_madlad_data(limit=None):
    """
    Loads Dutch monolingual data from MADLAD-400.
    """

    # check if dataset is already downloaded
    if not os.path.exists(MADLAD_FILE):
        print("MADLAD-400 dataset not found. Downloading...")
        os.makedirs(MADLAD_DIR, exist_ok=True)
        
        # stream=True for large file download
        response = requests.get(MADLAD_URL, stream=True)
        if response.status_code == 200:
            with open(MADLAD_FILE, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download successful.")
        else:
            raise Exception(f"Failed to download MADLAD-400. Status code: {response.status_code}")

    sources = []
    print(f"Reading {limit if limit else 'all'} sentences from MADLAD-400...")
    
    # open file and read line by line
    with gzip.open(MADLAD_FILE, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            # get text field from json line
            data = json.loads(line)
            sources.append(data['text'].strip())
            
    # only return sources because dataset is monolingual
    return sources, None