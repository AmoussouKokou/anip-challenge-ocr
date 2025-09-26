
import os, json, random, numpy as np, torch

def seed_everything(seed=42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_images(root, exts=(".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
    out=[]
    for dp,_,files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(dp,f))
    return out

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as w:
        json.dump(obj, w, ensure_ascii=False, indent=2)
