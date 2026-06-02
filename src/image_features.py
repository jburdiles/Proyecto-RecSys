import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings")
PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "photos")

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageFeatureExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def _load_model(self):
        if self._model is None:
            print(f"Loading ResNet50 on {self.device}...")
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self._model = nn.Sequential(*list(backbone.children())[:-1])
            self._model.eval().to(self.device)

    def _embed_single(self, photo_path):
        try:
            img = Image.open(photo_path).convert("RGB")
            tensor = TRANSFORM(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self._model(tensor).squeeze().cpu().numpy()
            return feat
        except Exception as e:
            print(f"  Warning: could not process {photo_path}: {e}")
            return None

    def extract(self, photos_df, photos_dir=PHOTOS_DIR):
        self._load_model()
        business_embeddings = {}

        for _, row in tqdm(photos_df.iterrows(), total=len(photos_df), desc="Extracting image features"):
            path = os.path.join(photos_dir, f"{row['photo_id']}.jpg")
            if not os.path.exists(path):
                continue
            feat = self._embed_single(path)
            if feat is not None:
                business_embeddings.setdefault(row["business_id"], []).append(feat)

        result = {bid: np.mean(vecs, axis=0) for bid, vecs in business_embeddings.items()}
        print(f"Image embeddings computed for {len(result):,} restaurants")
        return result

    def save(self, embeddings, filename="image_embeddings.npz"):
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        path = os.path.join(EMBEDDINGS_DIR, filename)
        ids = list(embeddings.keys())
        vecs = np.stack([embeddings[i] for i in ids])
        np.savez(path, ids=ids, embeddings=vecs)
        print(f"Saved {len(ids)} image embeddings -> {path}")

    @staticmethod
    def load(filename="image_embeddings.npz"):
        path = os.path.join(EMBEDDINGS_DIR, filename)
        data = np.load(path, allow_pickle=True)
        return {bid: emb for bid, emb in zip(data["ids"], data["embeddings"])}

