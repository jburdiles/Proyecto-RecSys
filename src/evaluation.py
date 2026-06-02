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


def evaluate_model(recommend_fn, test_df, train_df, k_values=[5, 10, 20], min_test_items=1):
    """
    Evalúa un modelo sobre el conjunto de test.
    recommend_fn recibe user_id y top_k, devuelve lista de business_ids.
    Solo evalúa usuarios con al menos `min_test_items` items en test.
    Devuelve DataFrame con métricas promediadas por K.
    """
    results = defaultdict(list)

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
            })

    rows = []
    for k in k_values:
        if not results[k]:
            continue
        metrics = pd.DataFrame(results[k]).mean()
        rows.append({"K": k, **metrics.to_dict()})

    return pd.DataFrame(rows).set_index("K")


def compare_models(models, test_df, train_df, k_values=[5, 10, 20]):
    """Evalúa varios modelos y devuelve una tabla resumen comparativa."""
    frames = []
    for name, rec_fn in models.items():
        df = evaluate_model(rec_fn, test_df, train_df, k_values)
        df.columns = pd.MultiIndex.from_product([[name], df.columns])
        frames.append(df)
    return pd.concat(frames, axis=1)
