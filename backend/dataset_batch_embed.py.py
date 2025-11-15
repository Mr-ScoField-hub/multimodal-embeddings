import os
import json
import numpy as np
from PIL import Image
import torch
import clip

DATASET_DIR = "multi_modal_dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
CAPTIONS_FILE = os.path.join(DATASET_DIR, "captions.json")
EMBED_DIR = os.path.join(DATASET_DIR, "embeddings")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

with open(CAPTIONS_FILE, "r") as f:
    captions = json.load(f)

counter = 1

for img_name, text in captions.items():
    img_path = os.path.join(IMAGES_DIR, img_name)
    if not os.path.exists(img_path):
        continue

    idx = str(counter).zfill(4)

    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_embed = model.encode_image(image)
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        txt_embed = model.encode_text(text_tokens)

    img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
    txt_embed = txt_embed / txt_embed.norm(dim=-1, keepdim=True)

    np.save(os.path.join(EMBED_DIR, f"embedding_{idx}_img.npy"), img_embed.cpu().numpy())
    np.save(os.path.join(EMBED_DIR, f"embedding_{idx}_txt.npy"), txt_embed.cpu().numpy())

    counter += 1

print("Embeddings generated.")
