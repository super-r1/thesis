from comet import download_model, load_from_checkpoint
from .config import device

def comet22_eval(sources, translations, references):
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
    
    comet_data = [
        {"src": s, "mt": t, "ref": r} 
        for s, t, r in zip(sources, translations, references)
    ]
    
    # use gpu if available
    num_gpus = 1 if device == "cuda" else 0
    return comet_model.predict(comet_data, batch_size=16, gpus=num_gpus)