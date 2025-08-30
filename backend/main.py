# main.py
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pydantic import BaseModel, Field
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from contextlib import asynccontextmanager

# --- Data Loading ---
# This dictionary will hold our data and model, loaded at startup
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model and data
    print("--- Loading data and model ---")
    try:
        # Load the anime data
        anime_df = pd.read_csv("data/anime_data.csv")
        # Load embeddings and IDs
        embeddings = np.load("data/anime_embeddings.npy")
        anime_ids = np.load("data/anime_ids.npy")
        
        # Create a mapping from anime ID to its index in the embeddings array
        id_to_index = {mal_id: i for i, mal_id in enumerate(anime_ids)}

        # Store in our dictionary
        ml_models["anime_df"] = anime_df
        ml_models["embeddings"] = embeddings
        ml_models["anime_ids"] = anime_ids
        ml_models["id_to_index"] = id_to_index
        print("Successfully loaded data and model.")
    except FileNotFoundError:
        print("ERROR: Data files not found. Please run scripts/process_data.py first.")
        ml_models["anime_df"] = pd.DataFrame() # Avoid errors on startup
    yield
    # Clean up the ML models and data
    ml_models.clear()

# --- Application Setup ---
app = FastAPI(
    title="AI Anime Recommendation API",
    description="An API for providing AI-powered anime recommendations.",
    version="1.0.0",
    lifespan=lifespan # Use the new lifespan context manager
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Connection (Placeholder) ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "anime_recommendations"
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    # A simple check to confirm connection
    client.server_info() 
    print("Successfully connected to MongoDB.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    client = None
    db = None


# --- Pydantic Models ---
class Recommendation(BaseModel):
    mal_id: int
    title: str
    synopsis: str
    genres: str
    image_url: str
    similarity_score: float = Field(..., alias="score")
    mal_score: float # The overall MyAnimeList score

class AnimeInfo(BaseModel):
    mal_id: int
    title: str
    synopsis: str | None = None
    image_url: str | None = None
    score: float | None = None


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Anime Recommendation API!"}

@app.get("/recommend/{anime_id}", response_model=List[Recommendation])
def get_recommendations_for_anime(anime_id: int, limit: int = 10):
    if ml_models.get("anime_df") is None or ml_models["anime_df"].empty:
        raise HTTPException(status_code=503, detail="Model and data not loaded.")

    id_to_index = ml_models["id_to_index"]
    if anime_id not in id_to_index:
        raise HTTPException(status_code=404, detail="Anime ID not found.")

    anime_idx = id_to_index[anime_id]
    anime_vector = ml_models["embeddings"][anime_idx].reshape(1, -1)
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(anime_vector, ml_models["embeddings"])[0]
    
    # Get top N similar anime indices, excluding the anime itself
    similar_indices = similarity_scores.argsort()[-limit-1:-1][::-1]
    
    recommendations = []
    for idx in similar_indices:
        rec_id = ml_models["anime_ids"][idx]
        anime_details = ml_models["anime_df"][ml_models["anime_df"]["mal_id"] == rec_id].iloc[0]
        rec = {
            "mal_id": int(rec_id),
            "title": anime_details["title"],
            "synopsis": anime_details["synopsis"],
            "genres": anime_details["genres"],
            "image_url": anime_details["image_url"],
            "score": similarity_scores[idx],
            "mal_score": anime_details.get("score", 0.0)
        }
        recommendations.append(rec)
        
    return recommendations

@app.get("/anime", response_model=List[AnimeInfo])
def get_all_anime():
    """Returns a list of all anime for the homepage/search."""
    if ml_models.get("anime_df") is None or ml_models["anime_df"].empty:
        raise HTTPException(status_code=503, detail="Model and data not loaded.")
    
    # Convert dataframe to list of dictionaries for Pydantic validation
    return ml_models["anime_df"].to_dict(orient="records")

class WatchedItem(BaseModel):
    user_id: str
    anime_id: int

@app.post("/user/watched")
def add_to_watched_list(item: WatchedItem):
    # This is placeholder logic. In a real app, you would save this to MongoDB.
    print(f"User {item.user_id} watched anime {item.anime_id}")
    return {"status": "success", "user_id": item.user_id, "added_anime_id": item.anime_id}

