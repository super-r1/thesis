import os
import torch
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
device = "cuda" if torch.cuda.is_available() else "cpu"

# on snellius, use scratch. otherwise local
if os.path.exists("/scratch-shared"):
    DATA_DIR = "/scratch-shared/bveenman/data"
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    OUTPUT_DIR = f"/scratch-shared/bveenman/{job_id}/outputs"
else:
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs"))

# create folder if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# various ways of defining language codes
LANG_MAP = {
 "nl": {
 "name": "Dutch",
 "gemma": "nl",
 "wmt": "en-nl_NL",
 "flores": "nld_Latn",
 "madlad": "nl"
 },
 "zh": {
 "name": "Chinese",
 "gemma": "zh",
 "wmt": "en-zh_CN", 
 "flores": "zho_Hans",
 "madlad": "zh"
 }
}

MODEL_ID_MAP = {
 "translategemma": {
 "id": "google/translategemma-4b-it",
 "type": "translate_only",
 "system_instr": None,
 "refine_instr": "This initial hypothesis needs improvement. Please refine it for accuracy and fluency.",
 "keep_instr": "This initial hypothesis is already high-quality. Please provide the final version."
 },
 "gemma": {
 "id": "google/gemma-3-4b-it",
 "type": "general",
 "system_instr": "You are a professional translator. Produce only the translation, without any additional explanations or commentary.",
 "refine_instr": "You are an expert editor. Below is a translation draft. Please refine it to be more accurate and natural."
 }
}

DEFAULT_MODEL = "translategemma"