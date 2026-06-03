import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings")
PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "photos")


class CLIPFeatureExtractor:
    CLIP_MODEL = "openai/clip-vit-base-patch32"

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor
            print(f"Loading CLIP ViT-B/32 on {self.device}...")
            self._processor = CLIPProcessor.from_pretrained(self.CLIP_MODEL)
            self._model = CLIPModel.from_pretrained(self.CLIP_MODEL).to(self.device)
            self._model.eval()

    def _embed_batch(self, images):
        inputs = self._processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def extract(self, photos_df, photos_dir=PHOTOS_DIR, batch_size=32):
        self._load_model()
        business_embeddings = {}
        grouped = photos_df.groupby("business_id")["photo_id"].apply(list).to_dict()

        for bid, photo_ids in tqdm(grouped.items(), desc="CLIP: extracting per restaurant"):
            batch_imgs = []
            for pid in photo_ids:
                path = os.path.join(photos_dir, f"{pid}.jpg")
                if not os.path.exists(path):
                    continue
                try:
                    batch_imgs.append(Image.open(path).convert("RGB"))
                except Exception:
                    continue

            if not batch_imgs:
                continue

            all_feats = []
            for i in range(0, len(batch_imgs), batch_size):
                all_feats.append(self._embed_batch(batch_imgs[i:i + batch_size]))

            business_embeddings[bid] = np.mean(np.vstack(all_feats), axis=0)

        print(f"CLIP embeddings computed for {len(business_embeddings):,} restaurants (512d)")
        return business_embeddings

    def save(self, embeddings, filename="clip_embeddings.npz"):
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        path = os.path.join(EMBEDDINGS_DIR, filename)
        ids = list(embeddings.keys())
        vecs = np.stack([embeddings[i] for i in ids])
        np.savez(path, ids=ids, embeddings=vecs)
        print(f"Saved {len(ids)} CLIP embeddings -> {path}")

    @staticmethod
    def load(filename="clip_embeddings.npz"):
        path = os.path.join(EMBEDDINGS_DIR, filename)
        data = np.load(path, allow_pickle=True)
        return {bid: emb for bid, emb in zip(data["ids"], data["embeddings"])}
