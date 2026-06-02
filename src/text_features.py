import os
import numpy as np

EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings")


class TextFeatureExtractor:
    def __init__(self, model_name="all-MiniLM-L6-v2", mode="dense"):
        self.model_name = model_name
        self.mode = mode
        self._model = None
        self._tfidf = None

    def _load_model(self):
        if self.mode == "dense" and self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading sentence-transformer: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        elif self.mode == "tfidf" and self._tfidf is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(max_features=10_000, min_df=5, stop_words="english")

    def _aggregate_reviews(self, reviews_df):
        return (
            reviews_df.groupby("business_id")["text"]
            .apply(lambda texts: " ".join(texts.astype(str).str.strip()))
            .to_dict()
        )

    def fit_transform(self, reviews_df):
        self._load_model()
        texts_by_business = self._aggregate_reviews(reviews_df)
        business_ids = list(texts_by_business.keys())
        texts = [texts_by_business[bid] for bid in business_ids]

        if self.mode == "dense":
            print("Encoding texts with sentence-transformers...")
            embeddings = self._model.encode(texts, batch_size=64, show_progress_bar=True)
        else:
            print("Fitting TF-IDF...")
            embeddings = self._tfidf.fit_transform(texts).toarray()

        return {bid: emb for bid, emb in zip(business_ids, embeddings)}

    def save(self, embeddings, filename="text_embeddings.npz"):
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        path = os.path.join(EMBEDDINGS_DIR, filename)
        ids = list(embeddings.keys())
        vecs = np.stack([embeddings[i] for i in ids])
        np.savez(path, ids=ids, embeddings=vecs)
        print(f"Saved {len(ids)} text embeddings -> {path}")

    @staticmethod
    def load(filename="text_embeddings.npz"):
        path = os.path.join(EMBEDDINGS_DIR, filename)
        data = np.load(path, allow_pickle=True)
        return {bid: emb for bid, emb in zip(data["ids"], data["embeddings"])}
