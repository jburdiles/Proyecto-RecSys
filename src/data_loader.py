import json
import os
import pandas as pd
from tqdm import tqdm

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _iter_json_lines(filepath, max_rows=None):
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            yield json.loads(line)


def load_restaurants(max_rows=None):
    path = os.path.join(RAW_DIR, "yelp_academic_dataset_business.json")
    records = []
    for row in tqdm(_iter_json_lines(path, max_rows), desc="Loading businesses"):
        if row.get("categories") and "Restaurant" in row["categories"]:
            records.append({
                "business_id": row["business_id"],
                "name": row["name"],
                "city": row["city"],
                "state": row["state"],
                "stars": row["stars"],
                "review_count": row["review_count"],
                "categories": row["categories"],
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "is_open": row.get("is_open", 1),
            })
    df = pd.DataFrame(records)
    print(f"Restaurants loaded: {len(df)}")
    return df


def load_reviews(restaurant_ids=None, max_rows=None):
    path = os.path.join(RAW_DIR, "yelp_academic_dataset_review.json")
    records = []
    for row in tqdm(_iter_json_lines(path, max_rows), desc="Loading reviews"):
        if restaurant_ids and row["business_id"] not in restaurant_ids:
            continue
        records.append({
            "review_id": row["review_id"],
            "user_id": row["user_id"],
            "business_id": row["business_id"],
            "stars": row["stars"],
            "text": row["text"],
            "date": row["date"],
            "useful": row.get("useful", 0),
        })
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Reviews loaded: {len(df)}")
    return df


def load_photos_metadata(restaurant_ids=None):
    path = os.path.join(RAW_DIR, "photos.json")
    records = []
    for row in _iter_json_lines(path):
        if restaurant_ids and row["business_id"] not in restaurant_ids:
            continue
        records.append({
            "photo_id": row["photo_id"],
            "business_id": row["business_id"],
            "label": row.get("label", ""),
        })
    df = pd.DataFrame(records)
    print(f"Photos metadata loaded: {len(df)}")
    return df


def load_users(user_ids=None, max_rows=None):
    path = os.path.join(RAW_DIR, "yelp_academic_dataset_user.json")
    records = []
    for row in tqdm(_iter_json_lines(path, max_rows), desc="Loading users"):
        if user_ids and row["user_id"] not in user_ids:
            continue
        records.append({
            "user_id": row["user_id"],
            "name": row["name"],
            "review_count": row["review_count"],
            "average_stars": row["average_stars"],
            "yelping_since": row.get("yelping_since"),
        })
    return pd.DataFrame(records)


def build_dataset(min_reviews_per_business=20, min_photos_per_business=5, max_review_rows=None):
    restaurants = load_restaurants()
    restaurants = restaurants[restaurants["review_count"] >= min_reviews_per_business]

    rest_ids = set(restaurants["business_id"])
    reviews = load_reviews(restaurant_ids=rest_ids, max_rows=max_review_rows)
    photos = load_photos_metadata(restaurant_ids=rest_ids)

    photo_counts = photos.groupby("business_id").size()
    valid_ids = set(photo_counts[photo_counts >= min_photos_per_business].index)
    restaurants = restaurants[restaurants["business_id"].isin(valid_ids)]
    reviews = reviews[reviews["business_id"].isin(valid_ids)]
    photos = photos[photos["business_id"].isin(valid_ids)]

    print(f"\nFinal dataset: {len(restaurants)} restaurants | "
          f"{len(reviews)} reviews | {len(photos)} photos")
    return restaurants, reviews, photos
