import torch
import os
from dotenv import load_dotenv

load_dotenv()

model_id = "google/translategemma-4b-it"
HF_TOKEN = os.getenv("HF_TOKEN")
device = "cuda" if torch.cuda.is_available() else "cpu"