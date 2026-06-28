import os
import numpy as np
import pandas as pd

PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "photos")


def pick_sample_users(reviews_df, n=5, seed=42):
    """
    Devuelve n user_ids de forma determinista: mismo seed -> mismos usuarios
    en todos los notebooks (ordena los ids antes de muestrear, asi no depende
    del orden de filas del DataFrame).
    """
    users = np.array(sorted(reviews_df["user_id"].unique()))
    rng = np.random.default_rng(seed)
    n = min(n, len(users))
    return rng.choice(users, size=n, replace=False).tolist()


def _short(text, n=160):
    text = str(text).replace("\n", " ").strip()
    return text[:n] + ("..." if len(text) > n else "")


def _first_photo(business_ids, photos, photos_dir):
    """(business_id, path) de la primera foto existente por restaurante."""
    pmap = photos.groupby("business_id")["photo_id"].apply(list).to_dict()
    found = []
    for bid in business_ids:
        for pid in pmap.get(bid, []):
            path = os.path.join(photos_dir, f"{pid}.jpg")
            if os.path.exists(path):
                found.append((bid, path))
                break
    return found


def _show_images(business_ids, photos, rest, photos_dir):
    import matplotlib.pyplot as plt
    from PIL import Image

    found = _first_photo(business_ids, photos, photos_dir)
    if not found:
        print("  (sin imagenes disponibles para estas recomendaciones)")
        return
    fig, axes = plt.subplots(1, len(found), figsize=(4 * len(found), 4))
    axes = np.atleast_1d(axes)
    for ax, (bid, path) in zip(axes, found):
        try:
            ax.imshow(Image.open(path).convert("RGB"))
        except Exception:
            ax.set_visible(False)
            continue
        name = rest.loc[bid, "name"] if bid in rest.index else bid
        ax.set_title(_short(name, 25), fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_user_recommendations(user_id, recommend_fn, train_reviews, restaurants,
                              photos=None, top_k=5, n_history=3, show_images=True,
                              photos_dir=PHOTOS_DIR):
    """
    Imprime el historial del usuario (top reviews por rating, con texto) y sus
    top_k recomendaciones (nombre, ciudad, categorias). Si se pasan `photos`
    y show_images=True, muestra una foto por restaurante recomendado.
    """
    rest = restaurants.set_index("business_id")

    hist = train_reviews[train_reviews["user_id"] == user_id].sort_values(
        "stars", ascending=False)
    print(f"\n{'=' * 72}")
    print(f"Usuario {user_id}  |  {len(hist)} reviews en train")
    print(f"{'-' * 72}\nHistorial (top {n_history} por rating):")
    text_col = next((c for c in ("text", "text_clean") if c in hist.columns), None)
    for _, r in hist.head(n_history).iterrows():
        name = rest.loc[r["business_id"], "name"] if r["business_id"] in rest.index else r["business_id"]
        line = f"  [{r['stars']:.0f}*] {name}"
        if text_col:
            line += f' - "{_short(r[text_col])}"'
        print(line)

    recs = recommend_fn(user_id, top_k=top_k)
    print(f"{'-' * 72}\nTop {top_k} recomendaciones:")
    for i, bid in enumerate(recs, 1):
        if bid in rest.index:
            row = rest.loc[bid]
            print(f"  {i}. {row['name']} ({row.get('city', '?')}) - "
                  f"{row.get('stars', '?')}* - {_short(row.get('categories', ''), 60)}")
        else:
            print(f"  {i}. {bid} (sin metadata)")

    if show_images and photos is not None:
        _show_images(recs, photos, rest, photos_dir)
    return recs
