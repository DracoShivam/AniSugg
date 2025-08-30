# main.py
# Import necessary libraries
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pydantic import BaseModel, Field
from typing import List

# --- Application Setup ---
# Initialize the FastAPI application
app = FastAPI(
    title="AI Anime Recommendation API",
    description="An API for providing AI-powered anime recommendations.",
    version="1.0.0"
)

# --- CORS Middleware ---
# Define the origins that are allowed to make requests to this API
# This is crucial for connecting a frontend application.
origins = [
    "http://localhost:5173",
]

# Add the CORS middleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# --- Database Connection ---
# It's a best practice to load sensitive info from environment variables.
# We'll use a default value for local development if the variable isn't set.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "anime_recommendations"

# Establish a connection to the MongoDB database
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    # You can define collections here, e.g., watched_collection = db["watched"]
    print("Successfully connected to MongoDB.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    # In a real application, you might want to handle this more gracefully
    client = None
    db = None


# --- Pydantic Models ---
# Pydantic models define the structure of the request and response bodies.
# This provides data validation and documentation automatically.

class WatchedAnime(BaseModel):
    """Model for the data sent when a user marks an anime as watched."""
    user_id: str = Field(..., example="user123", description="The unique identifier for the user.")
    anime_id: int = Field(..., example=5114, description="The unique identifier for the anime (e.g., MyAnimeList ID).")

class Recommendation(BaseModel):
    """Model representing a single recommended anime."""
    anime_id: int = Field(..., example=21)
    title: str = Field(..., example="One Piece")
    score: float = Field(..., example=0.95)

# --- API Endpoints ---

@app.get("/", tags=["Root"])
def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Welcome to the AI Anime Recommendation API!"}

@app.get("/recommend/{anime_id}", response_model=List[Recommendation], tags=["Recommendations"])
def get_recommendations_for_anime(anime_id: int):
    """
    Get recommendations based on a single anime.

    This endpoint takes an anime ID and returns a list of similar anime.
    **Placeholder Logic:** Currently returns a fixed dummy list.
    """
    print(f"Received recommendation request for anime_id: {anime_id}")
    # In a real application, you would implement your recommendation logic here.
    # This might involve querying a pre-computed similarity matrix or running a model.
    dummy_recommendations = [
        {"anime_id": 1, "title": "Cowboy Bebop", "score": 0.98},
        {"anime_id": 30, "title": "Neon Genesis Evangelion", "score": 0.95},
        {"anime_id": 205, "title": "Samurai Champloo", "score": 0.92},
    ]
    return dummy_recommendations

@app.get("/profile/recommend", response_model=List[Recommendation], tags=["Recommendations"])
def get_recommendations_for_user():
    """
    Get personalized recommendations for the current user.

    This would typically require user authentication to identify the user.
    **Placeholder Logic:** Currently returns a fixed dummy list.
    """
    # In a real scenario, you would get the user_id from an auth token (e.g., JWT).
    user_id = "default_user" # Hardcoded for now
    print(f"Received profile recommendation request for user: {user_id}")
    # The logic here would fetch the user's watch history from the database,
    # and generate recommendations based on their unique taste profile.
    dummy_recommendations = [
        {"anime_id": 9253, "title": "Steins;Gate", "score": 0.99},
        {"anime_id": 11061, "title": "Hunter x Hunter (2011)", "score": 0.97},
        {"anime_id": 1535, "title": "Death Note", "score": 0.96},
    ]
    return dummy_recommendations

@app.post("/user/watched", tags=["User Profile"])
def add_to_watched_list(watched: WatchedAnime):
    """
    Add an anime to a user's watched list.

    This endpoint receives a user ID and an anime ID and should store
    this information in the database.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database connection not available.")

    print(f"Adding anime {watched.anime_id} to watched list for user {watched.user_id}")
    # In a real implementation, you would insert this into your MongoDB collection.
    # Example:
    # watched_collection = db["watched_history"]
    # result = watched_collection.insert_one(watched.dict())
    # print(f"Inserted document with id: {result.inserted_id}")

    return {"status": "success", "message": f"Anime {watched.anime_id} added to watched list for user {watched.user_id}."}


# To run this application:
# 1. Make sure you have the dependencies from requirements.txt installed.
# 2. Set the MONGO_URI environment variable if you're not using the default.
# 3. In your terminal, run: uvicorn main:app --reload
