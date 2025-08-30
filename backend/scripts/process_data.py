# scripts/process_data.py
import os
import time
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---
JIKAN_API_URL = "https://api.jikan.moe/v4/top/anime"
DATA_DIR = "data"
CSV_FILE_PATH = os.path.join(DATA_DIR, "anime_data.csv")
EMBEDDINGS_FILE_PATH = os.path.join(DATA_DIR, "anime_embeddings.npy")
IDS_FILE_PATH = os.path.join(DATA_DIR, "anime_ids.npy")
TARGET_ANIME_COUNT = 2000


def fetch_anime_data():
    """
    Fetches data for the top anime from the Jikan API, processes it,
    and saves it to a CSV file.
    """
    print("--- Starting to fetch anime data from Jikan API ---")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    all_anime_data = []
    page = 1
    has_next_page = True

    while has_next_page and len(all_anime_data) < TARGET_ANIME_COUNT:
        print(f"Fetching page {page}...")
        try:
            response = requests.get(JIKAN_API_URL, params={"page": page})
            response.raise_for_status()
            data = response.json()

            for anime in data.get("data", []):
                genres = [genre['name'] for genre in anime.get('genres', [])]

                processed_anime = {
                    'mal_id': anime.get('mal_id'),
                    'title': anime.get('title'),
                    'synopsis': anime.get('synopsis'),
                    'genres': ", ".join(genres),
                    'image_url': anime.get('images', {}).get('jpg', {}).get('image_url'),
                    'score': anime.get('score', 0.0)
                }
                all_anime_data.append(processed_anime)

            has_next_page = data.get("pagination", {}).get("has_next_page", False)
            page += 1
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching data: {e}")
            has_next_page = False

    print(f"\nFetched data for {len(all_anime_data)} anime.")
    df = pd.DataFrame(all_anime_data)

    print("Cleaning data...")
    df.dropna(subset=['mal_id', 'title', 'synopsis'], inplace=True)
    df['mal_id'] = df['mal_id'].astype(int)
    df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0.0)
    df.drop_duplicates(subset=['mal_id'], inplace=True)

    # --- FIX: Added encoding='utf-8' to handle special characters ---
    df.to_csv(CSV_FILE_PATH, index=False, encoding='utf-8')
    print(f"Successfully saved cleaned data to {CSV_FILE_PATH}")
    return df


def compute_and_save_embeddings():
    """
    Loads the anime data, computes sentence embeddings for the synopses,
    and saves the embeddings and corresponding IDs.
    """
    print("\n--- Starting to compute and save embeddings ---")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE_PATH} not found. Please run fetch_anime_data() first.")
        return

    print("Loading sentence-transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    synopses = df['synopsis'].fillna('').tolist()

    print("Computing embeddings for all synopses... (This may take a while)")
    embeddings = model.encode(synopses, show_progress_bar=True)

    anime_ids = df['mal_id'].values

    np.save(EMBEDDINGS_FILE_PATH, embeddings)
    np.save(IDS_FILE_PATH, anime_ids)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Successfully saved embeddings to {EMBEDDINGS_FILE_PATH}")
    print(f"Successfully saved anime IDs to {IDS_FILE_PATH}")


if __name__ == "__main__":
    fetch_anime_data()
    compute_and_save_embeddings()

