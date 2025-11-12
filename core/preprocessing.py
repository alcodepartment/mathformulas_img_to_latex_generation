# core/preprocessing.py
#Никита, я тут пока заглукшу накидал везде

import numpy as np, torch
from PIL import Image, ImageFilter

def preprocess(path: str, h: int = 160, w_max: int = 1024):
    img = Image.open(path).convert("L")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    w = int(img.width * (h / img.height))
    w = min(max(8, w), w_max)
    img = img.resize((w, h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)/255.0
    pad = np.zeros((h, w_max), dtype=np.float32)
    pad[:, :w] = arr
    return torch.from_numpy(pad[None, :, :])  # (1,H,W)
