import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pydantic import BaseModel, Field
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from contextlib import asynccontextmanager

# --- Data Loading ---
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Loading data and model ---")
    try:
        # CORRECTED FILE PATHS FOR RENDER DEPLOYMENT
        # The root directory on Render is 'backend', so we use relative paths from there.
        anime_df = pd.read_csv("data/anime_data.csv")
        embeddings = np.load("data/anime_embeddings.npy")
        anime_ids = np.load("data/anime_ids.npy")
        id_to_index = {mal_id: i for i, mal_id in enumerate(anime_ids)}
        
        ml_models["anime_df"] = anime_df
        ml_models["embeddings"] = embeddings
        ml_models["anime_ids"] = anime_ids
        ml_models["id_to_index"] = id_to_index
        print("Successfully loaded data and model.")
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
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "anime_recommendations"
try:
    if not MONGO_URI:
        print("ERROR: MONGO_URI environment variable not set.")
        client = None
    else:
        client = MongoClient(MONGO_URI)
        client.server_info() 
        db = client[DB_NAME]
        users_collection = db["users"]
        print("Successfully connected to MongoDB.")
except Exception as e:
    print(f"ERROR: Failed to connect to MongoDB: {e}")
    client = None

# --- Pydantic Models ---
class Anime(BaseModel):
    mal_id: int
    title: str
    image_url: str
    score: float
    synopsis: Optional[str] = None
    genres: Optional[str] = None

class Recommendation(Anime):
    similarity_score: float

class WatchedItem(BaseModel):
    user_id: str
    anime_id: int

# --- API Endpoints ---
@app.get("/anime/{anime_id}", response_model=Anime)
def get_anime_details(anime_id: int):
    anime_df = ml_models.get("anime_df")
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
    
    similar_indices = np.argsort(similarity_scores)[-limit-20:-1][::-1] 

    recommendations = []
    watched_ids = set()
    if user_id and client:
        user_data = users_collection.find_one({"user_id": user_id})
        if user_data:
            watched_ids = set(user_data.get("watched_list", []))

    for idx in similar_indices:
        if len(recommendations) >= limit:
            break
        rec_id = int(ml_models["anime_ids"][idx])
        if rec_id != anime_id and rec_id not in watched_ids:
            anime_details = anime_df[anime_df["mal_id"] == rec_id].iloc[0]
            rec = {
                **anime_details.to_dict(),
                "similarity_score": similarity_scores[idx],
            }
            recommendations.append(rec)
            
    return recommendations

@app.get("/profile/recommend/{user_id}", response_model=List[Recommendation])
def get_profile_recommendations(user_id: str, limit: int = 10):
    if not client:
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
    
    similar_indices = np.argsort(similarity_scores)[-limit-len(watched_ids):][::-1]

    recommendations = []
    for idx in similar_indices:
        if len(recommendations) >= limit:
            break
        rec_id = int(ml_models["anime_ids"][idx])
        if rec_id not in watched_ids:
            anime_details = ml_models["anime_df"][ml_models["anime_df"]["mal_id"] == rec_id].iloc[0]
            rec = {
                **anime_details.to_dict(),
                "similarity_score": similarity_scores[idx],
            }
            recommendations.append(rec)

    return recommendations

@app.get("/user/watched/{user_id}", response_model=List[int])
def get_watched_list(user_id: str):
    if not client:
        raise HTTPException(status_code=503, detail="Database connection not available.")
    user_data = users_collection.find_one({"user_id": user_id})
    return user_data.get("watched_list", []) if user_data else []

@app.post("/user/watched")
def add_to_watched_list(item: WatchedItem):
    if not client:
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

