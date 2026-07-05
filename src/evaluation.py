import numpy as np
import pandas as pd
from collections import defaultdict


def temporal_train_test_split(reviews_df, test_fraction=0.2):
    """
    Split temporal: para cada usuario se toma el último test_fraction
    de sus reviews (ordenadas por fecha) como test.
    Usuarios con menos de 5 reviews se excluyen.
    """
    reviews_df = reviews_df.sort_values(["user_id", "date"])
    train_rows, test_rows = [], []

    for user_id, group in reviews_df.groupby("user_id"):
        if len(group) < 5:
            train_rows.append(group)
            continue
        n_test = max(1, int(len(group) * test_fraction))
        train_rows.append(group.iloc[:-n_test])
        test_rows.append(group.iloc[-n_test:])

    train = pd.concat(train_rows).reset_index(drop=True)
    test = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame()
    print(f"Train: {len(train)} reviews | Test: {len(test)} reviews")
    return train, test


def precision_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def item_popularity(train_df):
    """Popularidad = fraccion de usuarios que interactuaron con cada item en train."""
    n_users = train_df["user_id"].nunique()
    counts = train_df.groupby("business_id")["user_id"].nunique()
    return (counts / n_users).to_dict(), n_users


def novelty_at_k(recommended, popularity, n_users, k):
    """Novedad media (self-information) de la lista top-k. Mayor = items menos populares."""
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    floor = 1.0 / n_users  # items no vistos en train: tratados como vistos por ~nadie
    return float(np.mean([
        -np.log2(max(popularity.get(item, floor), floor)) for item in top_k
    ]))


def hit_rate_at_k(recommended, relevant, k):
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def intra_list_diversity(recommended, item_features, k):
    """
    Diversidad intra-lista (ILD): disimilitud coseno media entre todos los
    pares de items de la lista top-k. 0 = items identicos, 1 = ortogonales.
    Solo usa items que tienen embedding; devuelve None si quedan <2.
    Para comparar modelos hay que pasar SIEMPRE el mismo espacio de features.
    """
    top_k = [i for i in recommended[:k] if i in item_features]
    if len(top_k) < 2:
        return None
    V = np.stack([item_features[i] for i in top_k]).astype(float)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    sims = V @ V.T
    iu = np.triu_indices(len(top_k), k=1)  # solo pares (i<j), sin diagonal
    return float(np.mean(1.0 - sims[iu]))


def evaluate_model(recommend_fn, test_df, train_df, k_values=[5, 10, 20],
                   min_test_items=1, item_features=None):
    """
    Evalúa un modelo sobre el conjunto de test.
    recommend_fn recibe user_id y top_k, devuelve lista de business_ids.
    Solo evalúa usuarios con al menos `min_test_items` items en test.
    Si se pasa `item_features` (dict business_id -> vector), agrega la
    columna `ild` (diversidad intra-lista). Devuelve DataFrame por K.
    """
    results = defaultdict(list)
    ild_scores = defaultdict(list)
    recommended_items = defaultdict(set)  # union de recomendaciones por K -> coverage

    popularity, n_users = item_popularity(train_df)
    # catalogo evaluable = items con interacciones (train ∪ test), fijo entre modelos
    catalog = set(train_df["business_id"]) | set(test_df["business_id"])
    catalog_size = len(catalog)

    test_users = test_df.groupby("user_id")["business_id"].apply(set)
    test_users = test_users[test_users.apply(len) >= min_test_items]

    for user_id, relevant in test_users.items():
        max_k = max(k_values)
        try:
            recommended = recommend_fn(user_id, top_k=max_k)
        except Exception:
            continue
        if not recommended:
            continue
        for k in k_values:
            results[k].append({
                "precision": precision_at_k(recommended, relevant, k),
                "recall": recall_at_k(recommended, relevant, k),
                "ndcg": ndcg_at_k(recommended, relevant, k),
                "novelty": novelty_at_k(recommended, popularity, n_users, k),
                "hit_rate": hit_rate_at_k(recommended, relevant, k),
            })
            recommended_items[k].update(recommended[:k])
            if item_features is not None:
                ild = intra_list_diversity(recommended, item_features, k)
                if ild is not None:
                    ild_scores[k].append(ild)

    rows = []
    for k in k_values:
        if not results[k]:
            continue
        metrics = pd.DataFrame(results[k]).mean()
        row = {"K": k, **metrics.to_dict()}
        # coverage: fraccion del catalogo evaluable que el modelo llega a recomendar
        covered = recommended_items[k] & catalog
        row["coverage"] = len(covered) / catalog_size if catalog_size else 0.0
        if item_features is not None and ild_scores[k]:
            row["ild"] = float(np.mean(ild_scores[k]))
        rows.append(row)

    return pd.DataFrame(rows).set_index("K")


def evaluate_rmse(predict_fn, test_df):
    """
    RMSE y MAE de predicción de rating sobre el test set.
    predict_fn(user_id, item_id) -> float | None (None si el modelo no cubre el par).
    """
    preds, actuals = [], []
    for _, row in test_df.iterrows():
        p = predict_fn(row['user_id'], row['business_id'])
        if p is not None:
            preds.append(p)
            actuals.append(row['stars'])
    if not preds:
        return {'rmse': float('nan'), 'mae': float('nan')}
    errors = np.array(preds) - np.array(actuals)
    print(f'Evaluado sobre {len(preds):,} pares ({len(preds)/len(test_df)*100:.1f}% del test set)')
    return {'rmse': float(np.sqrt(np.mean(errors**2))), 'mae': float(np.mean(np.abs(errors)))}


def compare_models(models, test_df, train_df, k_values=[5, 10, 20]):
    """Evalúa varios modelos y devuelve una tabla resumen comparativa."""
    frames = []
    for name, rec_fn in models.items():
        df = evaluate_model(rec_fn, test_df, train_df, k_values)
        df.columns = pd.MultiIndex.from_product([[name], df.columns])
        frames.append(df)
    return pd.concat(frames, axis=1)
