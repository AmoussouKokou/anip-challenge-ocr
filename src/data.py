
import os, glob, random, cv2, numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .constants import CLASSES

def make_transforms(train=True, size=768):
    aug = [A.LongestMaxSize(size), A.PadIfNeeded(size,size, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255))]
    if train:
        aug += [
            A.ImageCompression(quality_lower=40,quality_upper=90,p=0.5),
            A.MotionBlur(3,p=0.2), A.GaussianBlur(3,p=0.2),
            A.RandomBrightnessContrast(0.2,0.2,p=0.5),
            A.Rotate(limit=7, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255),p=0.5),
        ]
    aug += [A.Normalize(), ToTensorV2()]
    return A.Compose(aug)

def infer_country_from_path(p):
    p=p.lower()
    if "estonia" in p or "ee" in p: return "ee"
    if "spain" in p or "es" in p: return "es"
    if "russia" in p or "ru" in p: return "ru"
    if "arizona" in p or "az" in p or "usa" in p: return "az"
    return "unknown"

class IdDocsDataset(Dataset):
    def __init__(self, root, train=True, size=768):
        self.root = root
        self.items=[]
        for country in os.listdir(root):
            cdir = os.path.join(root, country)
            if not os.path.isdir(cdir): continue
            for cls in CLASSES:
                img_dir = os.path.join(cdir, cls if cls!="normal" else "normal")
                if not os.path.isdir(img_dir): continue
                for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff"):
                    for imgp in glob.glob(os.path.join(img_dir,ext)):
                        self.items.append({"img": imgp, "label": cls, "country": infer_country_from_path(imgp)})
        random.shuffle(self.items)
        self.transforms = make_transforms(train=train, size=size)
        self.class_to_idx = {c:i for i,c in enumerate(CLASSES)}

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        x = self.items[i]
        img = cv2.imread(x["img"]); 
        if img is None:
            raise FileNotFoundError(f"Impossible de lire: {x['img']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = self.transforms(image=img)["image"]
        y = self.class_to_idx[x["label"]]
        return {"image": t, "label": y, "country": x["country"], "path": x["img"]}
