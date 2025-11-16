from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import json
import torch
from PIL import Image
import clip

app = FastAPI()

# Allow your frontend and localhost during testing
origins = [
    "https://frontend-6zqct85x2-scofields-projects-b3359916.vercel.app",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

DATA_DIR = "data"
UPLOAD_DIR = "uploads"
EMBED_DIR = "embeddings"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

captions_file = os.path.join(DATA_DIR, "captions.json")
if not os.path.exists(captions_file):
    with open(captions_file, "w") as f:
        json.dump({}, f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_next_embedding_name():
    existing = [f for f in os.listdir(EMBED_DIR) if f.startswith("embedding_") and f.endswith(".pt")]
    if not existing:
        return "embedding_001.pt"
    existing_numbers = [int(f.split("_")[1].split(".")[0]) for f in existing]
    next_number = max(existing_numbers) + 1
    return f"embedding_{next_number:03d}.pt"

@app.post("/upload/")
async def upload_image(file: UploadFile):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "success", "filename": file.filename}

@app.post("/embed/")
async def generate_embedding(filename: str = Form(...), caption: str = Form(...)):
    with open(captions_file, "r") as f:
        captions = json.load(f)
    captions[filename] = caption
    with open(captions_file, "w") as f:
        json.dump(captions, f, indent=2)

    image = preprocess(Image.open(os.path.join(UPLOAD_DIR, filename))).unsqueeze(0).to(device)
    text = clip.tokenize([caption]).to(device)

    with torch.no_grad():
        image_embedding = model.encode_image(image)
        text_embedding = model.encode_text(text)

    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    combined = torch.cat([image_embedding, text_embedding], dim=-1)
    emb_filename = get_next_embedding_name()
    emb_path = os.path.join(EMBED_DIR, emb_filename)
    torch.save(combined.cpu(), emb_path)

    matrix = combined.cpu().numpy().tolist()

    return {"filename": filename, "embedding_file": emb_path, "matrix": matrix}

# --- ENTRY POINT FOR FLY.IO ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
