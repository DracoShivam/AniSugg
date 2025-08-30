# main.py
# Import necessary libraries
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pydantic import BaseModel, Field
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity
import logging

# --- Application Setup ---
# Initialize the FastAPI application
app = FastAPI(
    title="AI Anime Recommendation API",
    description="An API for providing AI-powered anime recommendations.",
    version="1.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CORS Middleware ---
origins = [
    "http://localhost:5173",
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
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    users_collection = db.users
    client.server_info()
    logger.info("Successfully connected to MongoDB.")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    client = None

# --- Pydantic Models ---
class Anime(BaseModel):
    mal_id: int
    title: str
    image_url: str
    score: float
    synopsis: Optional[str] = None

# Explicit model for recommendations to avoid ambiguity
class Recommendation(BaseModel):
    mal_id: int
    title: str
    image_url: str
    score: float
    synopsis: Optional[str] = None
    similarity_score: float

class WatchedAnime(BaseModel):
    user_id: str
    anime_id: int

# --- Global Variables for AI Model ---
anime_df = pd.DataFrame()
anime_embeddings = np.array([])
anime_ids = np.array([])
id_to_index = {}

# --- AI Model Loading ---
@app.on_event("startup")
def load_model_and_data():
    global anime_df, anime_embeddings, anime_ids, id_to_index
    logger.info("Loading AI model and data...")
    try:
        anime_df = pd.read_csv("data/anime_data.csv")
        anime_embeddings = np.load("data/anime_embeddings.npy")
        anime_ids = np.load("data/anime_ids.npy")
        id_to_index = {mal_id: i for i, mal_id in enumerate(anime_ids)}
        logger.info("Successfully loaded all data files.")
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}. Please run scripts/process_data.py")

# --- API Endpoints ---
@app.get("/anime", response_model=List[Anime])
def get_all_anime():
    return anime_df.to_dict(orient="records")

@app.get("/recommend/{anime_id}", response_model=List[Recommendation])
def get_recommendations_for_anime(anime_id: int, user_id: str):
    if anime_id not in id_to_index:
        raise HTTPException(status_code=404, detail="Anime not found in the dataset.")

    user_watched_list = users_collection.find_one({"user_id": user_id}, {"watched_list": 1})
    watched_ids = set(user_watched_list.get("watched_list", [])) if user_watched_list else set()

    idx = id_to_index[anime_id]
    sim_scores = cosine_similarity([anime_embeddings[idx]], anime_embeddings)[0]
    
    similar_indices = sim_scores.argsort()[::-1][1:]
    
    recs = []
    for i in similar_indices:
        if len(recs) >= 10: break
        rec_id = int(anime_ids[i])
        if rec_id != anime_id and rec_id not in watched_ids:
            anime_details = anime_df.iloc[i]
            recs.append({
                "mal_id": rec_id,
                "title": anime_details["title"],
                "image_url": anime_details["image_url"],
                "synopsis": anime_details["synopsis"],
                "score": anime_details["score"],
                "similarity_score": sim_scores[i]
            })
    return recs

@app.get("/profile/recommend/{user_id}", response_model=List[Recommendation])
def get_recommendations_for_profile(user_id: str):
    user_data = users_collection.find_one({"user_id": user_id})
    if not user_data or not user_data.get("watched_list"):
        raise HTTPException(status_code=404, detail="User has no watched anime.")
    
    watched_indices = [id_to_index[wid] for wid in user_data["watched_list"] if wid in id_to_index]
    if not watched_indices:
        raise HTTPException(status_code=404, detail="None of the user's watched anime are in the recommendation dataset.")

    taste_profile_vector = np.mean(anime_embeddings[watched_indices], axis=0)
    sim_scores = cosine_similarity([taste_profile_vector], anime_embeddings)[0]

    similar_indices = sim_scores.argsort()[::-1]
    
    recs = []
    for i in similar_indices:
        if len(recs) >= 10: break
        rec_id = int(anime_ids[i])
        if rec_id not in user_data["watched_list"]:
            anime_details = anime_df.iloc[i]
            recs.append({
                "mal_id": rec_id,
                "title": anime_details["title"],
                "image_url": anime_details["image_url"],
                "synopsis": anime_details["synopsis"],
                "score": anime_details["score"],
                "similarity_score": sim_scores[i]
            })
    return recs
    
@app.post("/user/watched")
def add_to_watched_list(data: WatchedAnime):
    try:
        users_collection.update_one(
            {"user_id": data.user_id},
            {"$addToSet": {"watched_list": data.anime_id}},
            upsert=True
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Database update failed: {e}")
        raise HTTPException(status_code=500, detail="Could not update watched list.")

@app.get("/user/watched/{user_id}", response_model=List[int])
def get_watched_list(user_id: str):
    user_data = users_collection.find_one({"user_id": user_id}, {"watched_list": 1})
    if user_data:
        return user_data.get("watched_list", [])
    return []

