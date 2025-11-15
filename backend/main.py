from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, json, torch
from PIL import Image
from clip import clip

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
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
    # Save caption
    with open(captions_file, "r") as f:
        captions = json.load(f)
    captions[filename] = caption
    with open(captions_file, "w") as f:
        json.dump(captions, f, indent=2)

    # Load image
    image = preprocess(Image.open(os.path.join(UPLOAD_DIR, filename))).unsqueeze(0).to(device)
    text = clip.tokenize([caption]).to(device)

    with torch.no_grad():
        image_embedding = model.encode_image(image)
        text_embedding = model.encode_text(text)

    # Normalize
    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    # Combine embeddings
    combined = torch.cat([image_embedding, text_embedding], dim=-1)
    emb_filename = get_next_embedding_name()
    emb_path = os.path.join(EMBED_DIR, emb_filename)
    torch.save(combined.cpu(), emb_path)

    # Convert tensor to 2D array for frontend display
    matrix = combined.cpu().numpy().tolist()

    return {"filename": filename, "embedding_file": emb_path, "matrix": matrix}
