import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pydantic import BaseModel
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from contextlib import asynccontextmanager

# --- Data Loading ---
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Loading data and model ---")
    try:
        anime_df = pd.read_csv("data/anime_data.csv")

        # --- Clean data ---
        anime_df['synopsis'] = anime_df['synopsis'].fillna('')
        anime_df['genres'] = anime_df['genres'].fillna('')
        anime_df['score'] = anime_df['score'].fillna(0.0)

        embeddings = np.load("data/anime_embeddings.npy")
        anime_ids = np.load("data/anime_ids.npy")
        id_to_index = {mal_id: i for i, mal_id in enumerate(anime_ids)}

        ml_models["anime_df"] = anime_df
        ml_models["embeddings"] = embeddings
        ml_models["anime_ids"] = anime_ids
        ml_models["id_to_index"] = id_to_index
        print("Successfully loaded and cleaned data and model.")
    except FileNotFoundError:
        print("ERROR: Data files not found. Ensure data files are in the data/ directory.")
        ml_models["anime_df"] = pd.DataFrame()
    yield
    ml_models.clear()

# --- Application Setup ---
app = FastAPI(
    title="AI Anime Recommendation API",
    lifespan=lifespan
)

# --- CORS Middleware ---
origins = [
    "http://localhost:5173",
    "https://ani-sugg.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Connection ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "anime_recommendations"
client = None
users_collection = None

try:
    if "mongodb://localhost" in MONGO_URI:
        print("Connecting to local MongoDB...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client[DB_NAME]
    users_collection = db["users"]
    print("Successfully connected to MongoDB.")
except Exception as e:
    print(f"WARNING: Could not connect to MongoDB: {e}. Database features will be disabled.")
    client = None

# --- Pydantic Models ---
class Anime(BaseModel):
    mal_id: int
    title: str
    image_url: str
    score: float
    synopsis: Optional[str] = ""
    genres: Optional[str] = ""

class Recommendation(Anime):
    similarity_score: float

class WatchedItem(BaseModel):
    user_id: str
    anime_id: int

# --- API Endpoints ---
@app.get("/anime/{anime_id}", response_model=Anime)
def get_anime_details(anime_id: int):
    anime_df = ml_models.get("anime_df")
    if anime_df is None or anime_df.empty:
         raise HTTPException(status_code=503, detail="Model and data not loaded.")
    details = anime_df[anime_df["mal_id"] == anime_id]
    if details.empty:
        raise HTTPException(status_code=404, detail="Anime not found")
    return details.iloc[0].to_dict()

@app.get("/anime", response_model=List[Anime])
def get_all_anime():
    if ml_models.get("anime_df") is None or ml_models["anime_df"].empty:
        raise HTTPException(status_code=503, detail="Model and data not loaded.")
    return ml_models["anime_df"].to_dict(orient="records")

@app.get("/recommend/{anime_id}", response_model=List[Recommendation])
def get_content_recommendations(anime_id: int, user_id: Optional[str] = None, limit: int = 10):
    id_to_index = ml_models["id_to_index"]
    anime_df = ml_models["anime_df"]

    if anime_id not in id_to_index:
        raise HTTPException(status_code=404, detail="Anime ID not found in model.")

    anime_idx = id_to_index[anime_id]
    anime_vector = ml_models["embeddings"][anime_idx].reshape(1, -1)

    similarity_scores = cosine_similarity(anime_vector, ml_models["embeddings"])[0]
    similar_indices = np.argsort(similarity_scores)[-limit-50:-1][::-1]

    recommendations = []
    watched_ids = set()
    if user_id and users_collection is not None:
        user_data = users_collection.find_one({"user_id": user_id})
        if user_data:
            watched_ids = set(user_data.get("watched_list", []))

    for idx in similar_indices:
        if len(recommendations) >= limit:
            break
        rec_id = int(ml_models["anime_ids"][idx])
        if rec_id != anime_id and rec_id not in watched_ids:
            anime_details = anime_df[anime_df["mal_id"] == rec_id].iloc[0]
            rec_data = anime_details.to_dict()
            rec_data["similarity_score"] = similarity_scores[idx]
            recommendations.append(rec_data)

    return recommendations

@app.get("/profile/recommend/{user_id}", response_model=List[Recommendation])
def get_profile_recommendations(user_id: str, limit: int = 10):
    if users_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available.")

    user_data = users_collection.find_one({"user_id": user_id})
    if not user_data or not user_data.get("watched_list"):
        raise HTTPException(status_code=404, detail="User has no watched anime.")

    watched_ids = user_data["watched_list"]
    id_to_index = ml_models["id_to_index"]

    watched_vectors = [ml_models["embeddings"][id_to_index[wid]] for wid in watched_ids if wid in id_to_index]
    if not watched_vectors:
        raise HTTPException(status_code=404, detail="None of the watched anime found in model.")

    taste_profile = np.mean(watched_vectors, axis=0).reshape(1, -1)

    similarity_scores = cosine_similarity(taste_profile, ml_models["embeddings"])[0]
    similar_indices = np.argsort(similarity_scores)[-limit-len(watched_ids)-20:][::-1]

    recommendations = []
    for idx in similar_indices:
        if len(recommendations) >= limit:
            break
        rec_id = int(ml_models["anime_ids"][idx])
        if rec_id not in watched_ids:
            anime_details = ml_models["anime_df"][ml_models["anime_df"]["mal_id"] == rec_id].iloc[0]
            rec_data = anime_details.to_dict()
            rec_data["similarity_score"] = similarity_scores[idx]
            recommendations.append(rec_data)

    return recommendations

@app.get("/user/watched/{user_id}", response_model=List[int])
def get_watched_list(user_id: str):
    if users_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available.")
    user_data = users_collection.find_one({"user_id": user_id})
    return user_data.get("watched_list", []) if user_data else []

@app.post("/user/watched")
def add_to_watched_list(item: WatchedItem):
    if users_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available.")
    try:
        users_collection.update_one(
            {"user_id": item.user_id},
            {"$addToSet": {"watched_list": item.anime_id}},
            upsert=True
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

# --- Reset User Profile ---
@app.delete("/user/reset/{user_id}")
def reset_user(user_id: str):
    if users_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available.")
    try:
        result = users_collection.delete_one({"user_id": user_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="User not found.")
        return {"status": "success", "message": f"User {user_id} profile reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
