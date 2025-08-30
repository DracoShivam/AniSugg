import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient, errors
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from contextlib import asynccontextmanager
import logging
from datetime import datetime
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants ---
class Config:
    MAX_RECOMMENDATIONS = 50
    DEFAULT_RECOMMENDATION_LIMIT = 10
    MIN_RECOMMENDATION_LIMIT = 1
    MAX_SEARCH_BUFFER = 100
    CACHE_TTL = 3600  # 1 hour
    MAX_WATCHED_LIST_SIZE = 10000
    MIN_SIMILARITY_THRESHOLD = 0.3
    
class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"

# --- Thread pool for CPU-bound operations ---
executor = ThreadPoolExecutor(max_workers=4)

# --- Data Loading and Caching ---
ml_models = {}
recommendation_cache = {}

def validate_data_integrity(df: pd.DataFrame, embeddings: np.ndarray, ids: np.ndarray) -> bool:
    """Validate loaded data integrity"""
    try:
        if len(df) != len(embeddings) or len(df) != len(ids):
            logger.error(f"Data size mismatch: df={len(df)}, embeddings={len(embeddings)}, ids={len(ids)}")
            return False
        
        if df['mal_id'].nunique() != len(df):
            logger.error("Duplicate anime IDs found in dataset")
            return False
            
        required_cols = ['mal_id', 'title', 'image_url', 'synopsis', 'genres', 'score']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Data validation error: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting application...")
    try:
        # Load data
        anime_df = pd.read_csv("data/anime_data.csv")
        embeddings = np.load("data/anime_embeddings.npy")
        anime_ids = np.load("data/anime_ids.npy")
        
        # Clean and validate data
        anime_df['synopsis'] = anime_df['synopsis'].fillna('')
        anime_df['genres'] = anime_df['genres'].fillna('')
        anime_df['score'] = pd.to_numeric(anime_df['score'], errors='coerce').fillna(0.0)
        anime_df['image_url'] = anime_df['image_url'].fillna('')
        anime_df['title'] = anime_df['title'].fillna('Unknown')
        
        # Remove invalid entries
        anime_df = anime_df[anime_df['mal_id'].notna()]
        anime_df['mal_id'] = anime_df['mal_id'].astype(int)
        
        if not validate_data_integrity(anime_df, embeddings, anime_ids):
            raise ValueError("Data integrity check failed")
        
        # Create efficient lookup structures
        id_to_index = {int(mal_id): i for i, mal_id in enumerate(anime_ids)}
        anime_dict = anime_df.set_index('mal_id').to_dict('index')
        
        # Normalize embeddings for faster cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        ml_models["anime_df"] = anime_df
        ml_models["anime_dict"] = anime_dict
        ml_models["embeddings"] = embeddings
        ml_models["normalized_embeddings"] = normalized_embeddings
        ml_models["anime_ids"] = anime_ids
        ml_models["id_to_index"] = id_to_index
        
        logger.info(f"Successfully loaded {len(anime_df)} anime with {embeddings.shape[1]}-dim embeddings")
        
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        ml_models["anime_df"] = pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        ml_models["anime_df"] = pd.DataFrame()
    
    yield
    
    logger.info("Shutting down application...")
    ml_models.clear()
    recommendation_cache.clear()

# --- Application Setup ---
app = FastAPI(
    title="AniSuggest - AI Anime Recommendation API",
    version="2.0.0",
    description="Advanced anime recommendation system using semantic similarity",
    lifespan=lifespan
)

# --- CORS Middleware ---
origins = [
    "http://localhost:5173",
    "http://localhost:5174",
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
class DatabaseConnection:
    def __init__(self):
        self.client = None
        self.db = None
        self.users_collection = None
        self.connect()
    
    def connect(self):
        """Establish MongoDB connection with retry logic"""
        MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        DB_NAME = "anime_recommendations"
        
        try:
            self.client = MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000,
                maxPoolSize=50,
                minPoolSize=10
            )
            self.client.admin.command('ping')
            self.db = self.client[DB_NAME]
            self.users_collection = self.db["users"]
            
            # Create indexes for better performance
            self.users_collection.create_index("user_id", unique=True)
            self.users_collection.create_index("updated_at")
            
            logger.info("Successfully connected to MongoDB")
        except errors.ServerSelectionTimeoutError:
            logger.warning("MongoDB connection timeout - running in offline mode")
            self.client = None
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            self.client = None
    
    def is_connected(self):
        """Check if database is connected"""
        return self.client is not None

# Initialize database connection
db_conn = DatabaseConnection()

# --- Dependency Injection ---
def get_db():
    """Database dependency"""
    if not db_conn.is_connected():
        raise HTTPException(status_code=503, detail="Database unavailable")
    return db_conn.users_collection

def get_ml_models():
    """ML models dependency"""
    if not ml_models.get("anime_df") is not None or ml_models["anime_df"].empty:
        raise HTTPException(status_code=503, detail="ML models not loaded")
    return ml_models

# --- Enhanced Pydantic Models ---
class Anime(BaseModel):
    mal_id: int = Field(..., description="MyAnimeList ID")
    title: str = Field(..., min_length=1, max_length=500)
    image_url: str = Field(default="")
    score: float = Field(ge=0, le=10)
    synopsis: Optional[str] = Field(default="", max_length=5000)
    genres: Optional[str] = Field(default="", max_length=500)
    
    @validator('score')
    def validate_score(cls, v):
        return round(float(v), 2) if v else 0.0

class Recommendation(Anime):
    similarity_score: float = Field(ge=0, le=1)
    
    @validator('similarity_score')
    def validate_similarity(cls, v):
        return round(float(v), 4)

class WatchedItem(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    anime_id: int = Field(..., gt=0)

class UserProfile(BaseModel):
    user_id: str
    watched_list: List[int]
    created_at: datetime
    updated_at: datetime
    preferences: Optional[Dict[str, Any]] = {}

class ApiResponse(BaseModel):
    status: ResponseStatus
    data: Optional[Any] = None
    message: Optional[str] = None

# --- Utility Functions ---
def compute_recommendations(
    target_vector: np.ndarray,
    embeddings: np.ndarray,
    anime_ids: np.ndarray,
    anime_dict: dict,
    exclude_ids: set,
    limit: int,
    min_similarity: float = Config.MIN_SIMILARITY_THRESHOLD
) -> List[Dict]:
    """Compute recommendations using optimized vector operations"""
    
    # Use normalized embeddings for faster computation
    if target_vector.ndim == 1:
        target_vector = target_vector.reshape(1, -1)
    
    target_norm = target_vector / np.linalg.norm(target_vector)
    similarities = np.dot(embeddings, target_norm.T).flatten()
    
    # Filter by minimum similarity threshold
    valid_indices = np.where(similarities >= min_similarity)[0]
    
    if len(valid_indices) == 0:
        return []
    
    # Sort and filter
    sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
    
    recommendations = []
    for idx in sorted_indices[:limit + Config.MAX_SEARCH_BUFFER]:
        rec_id = int(anime_ids[idx])
        if rec_id not in exclude_ids and rec_id in anime_dict:
            anime_data = anime_dict[rec_id].copy()
            anime_data["mal_id"] = rec_id
            anime_data["similarity_score"] = float(similarities[idx])
            recommendations.append(anime_data)
            
            if len(recommendations) >= limit:
                break
    
    return recommendations

# --- API Endpoints ---
@app.get("/", response_model=ApiResponse)
async def health_check():
    """Health check endpoint"""
    return ApiResponse(
        status=ResponseStatus.SUCCESS,
        data={
            "api": "healthy",
            "database": db_conn.is_connected(),
            "models_loaded": bool(ml_models.get("anime_df") is not None)
        },
        message="AniSuggest API is running"
    )

@app.get("/anime/{anime_id}", response_model=Anime)
async def get_anime_details(
    anime_id: int = Field(..., gt=0),
    models: dict = Depends(get_ml_models)
):
    """Get detailed information for a specific anime"""
    anime_dict = models["anime_dict"]
    
    if anime_id not in anime_dict:
        raise HTTPException(status_code=404, detail=f"Anime with ID {anime_id} not found")
    
    anime_data = anime_dict[anime_id].copy()
    anime_data["mal_id"] = anime_id
    return Anime(**anime_data)

@app.get("/anime", response_model=List[Anime])
async def get_all_anime(
    limit: Optional[int] = Query(None, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    models: dict = Depends(get_ml_models)
):
    """Get all anime with pagination support"""
    anime_df = models["anime_df"]
    
    # Apply pagination
    end_idx = offset + limit if limit else len(anime_df)
    paginated_df = anime_df.iloc[offset:end_idx]
    
    return [
        Anime(**row.to_dict())
        for _, row in paginated_df.iterrows()
    ]

@app.get("/recommend/{anime_id}", response_model=List[Recommendation])
async def get_content_recommendations(
    anime_id: int = Field(..., gt=0),
    user_id: Optional[str] = Query(None, max_length=100),
    limit: int = Query(Config.DEFAULT_RECOMMENDATION_LIMIT, ge=1, le=Config.MAX_RECOMMENDATIONS),
    models: dict = Depends(get_ml_models)
):
    """Get content-based recommendations for a specific anime"""
    
    # Check cache
    cache_key = f"{anime_id}_{user_id}_{limit}"
    if cache_key in recommendation_cache:
        cached_time, cached_data = recommendation_cache[cache_key]
        if (datetime.now() - cached_time).seconds < Config.CACHE_TTL:
            return cached_data
    
    id_to_index = models["id_to_index"]
    
    if anime_id not in id_to_index:
        raise HTTPException(status_code=404, detail=f"Anime ID {anime_id} not found in model")
    
    # Get watched list if user is logged in
    watched_ids = {anime_id}
    if user_id and db_conn.is_connected():
        try:
            user_data = db_conn.users_collection.find_one({"user_id": user_id})
            if user_data:
                watched_ids.update(user_data.get("watched_list", []))
        except Exception as e:
            logger.warning(f"Failed to fetch user data: {e}")
    
    # Compute recommendations
    anime_idx = id_to_index[anime_id]
    target_vector = models["normalized_embeddings"][anime_idx]
    
    recommendations = await asyncio.get_event_loop().run_in_executor(
        executor,
        compute_recommendations,
        target_vector,
        models["normalized_embeddings"],
        models["anime_ids"],
        models["anime_dict"],
        watched_ids,
        limit
    )
    
    result = [Recommendation(**rec) for rec in recommendations]
    
    # Cache results
    recommendation_cache[cache_key] = (datetime.now(), result)
    
    return result

@app.get("/profile/recommend/{user_id}", response_model=List[Recommendation])
async def get_profile_recommendations(
    user_id: str = Field(..., min_length=1, max_length=100),
    limit: int = Query(Config.DEFAULT_RECOMMENDATION_LIMIT, ge=1, le=Config.MAX_RECOMMENDATIONS),
    models: dict = Depends(get_ml_models),
    db: Any = Depends(get_db)
):
    """Get personalized recommendations based on user's watch history"""
    
    user_data = db.find_one({"user_id": user_id})
    if not user_data or not user_data.get("watched_list"):
        raise HTTPException(
            status_code=404,
            detail="User has no watched anime. Add some anime to get recommendations."
        )
    
    watched_ids = set(user_data["watched_list"])
    id_to_index = models["id_to_index"]
    
    # Get vectors for watched anime
    watched_vectors = []
    for wid in watched_ids:
        if wid in id_to_index:
            watched_vectors.append(models["normalized_embeddings"][id_to_index[wid]])
    
    if not watched_vectors:
        raise HTTPException(
            status_code=404,
            detail="None of the watched anime found in model"
        )
    
    # Create taste profile using weighted average (recent items weighted higher)
    weights = np.linspace(0.5, 1.0, len(watched_vectors))
    weights = weights / weights.sum()
    taste_profile = np.average(watched_vectors, axis=0, weights=weights)
    
    recommendations = await asyncio.get_event_loop().run_in_executor(
        executor,
        compute_recommendations,
        taste_profile,
        models["normalized_embeddings"],
        models["anime_ids"],
        models["anime_dict"],
        watched_ids,
        limit
    )
    
    return [Recommendation(**rec) for rec in recommendations]

@app.get("/user/watched/{user_id}", response_model=List[int])
async def get_watched_list(
    user_id: str = Field(..., min_length=1, max_length=100),
    db: Any = Depends(get_db)
):
    """Get user's watched anime list"""
    user_data = db.find_one({"user_id": user_id})
    return user_data.get("watched_list", []) if user_data else []

@app.post("/user/watched", response_model=ApiResponse)
async def add_to_watched_list(
    item: WatchedItem,
    db: Any = Depends(get_db),
    models: dict = Depends(get_ml_models)
):
    """Add anime to user's watched list"""
    
    # Validate anime exists
    if item.anime_id not in models["anime_dict"]:
        raise HTTPException(status_code=404, detail="Anime not found")
    
    try:
        result = db.update_one(
            {"user_id": item.user_id},
            {
                "$addToSet": {"watched_list": item.anime_id},
                "$set": {"updated_at": datetime.utcnow()},
                "$setOnInsert": {"created_at": datetime.utcnow()}
            },
            upsert=True
        )
        
        # Clear cache for this user
        cache_keys_to_remove = [k for k in recommendation_cache.keys() if item.user_id in k]
        for key in cache_keys_to_remove:
            del recommendation_cache[key]
        
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Added anime {item.anime_id} to watched list"
        )
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update watched list")

@app.delete("/user/watched/{user_id}/{anime_id}", response_model=ApiResponse)
async def remove_from_watched_list(
    user_id: str = Field(..., min_length=1, max_length=100),
    anime_id: int = Field(..., gt=0),
    db: Any = Depends(get_db)
):
    """Remove anime from user's watched list"""
    try:
        result = db.update_one(
            {"user_id": user_id},
            {
                "$pull": {"watched_list": anime_id},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Anime not in watched list")
        
        # Clear cache
        cache_keys_to_remove = [k for k in recommendation_cache.keys() if user_id in k]
        for key in cache_keys_to_remove:
            del recommendation_cache[key]
        
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Removed anime {anime_id} from watched list"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update watched list")

@app.delete("/user/profile/{user_id}/reset", response_model=ApiResponse)
async def reset_user_profile(
    user_id: str = Field(..., min_length=1, max_length=100),
    db: Any = Depends(get_db)
):
    """Reset user's profile by clearing their watched list"""
    try:
        result = db.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "watched_list": [],
                    "updated_at": datetime.utcnow(),
                    "reset_at": datetime.utcnow()
                }
            }
        )
        
        if result.matched_count == 0:
            # Create new empty profile if doesn't exist
            db.insert_one({
                "user_id": user_id,
                "watched_list": [],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "reset_at": datetime.utcnow()
            })
        
        # Clear all caches for this user
        cache_keys_to_remove = [k for k in recommendation_cache.keys() if user_id in k]
        for key in cache_keys_to_remove:
            del recommendation_cache[key]
        
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            message="Profile reset successfully. Your watched list has been cleared."
        )
    except Exception as e:
        logger.error(f"Failed to reset profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset profile")

@app.get("/user/stats/{user_id}", response_model=ApiResponse)
async def get_user_stats(
    user_id: str = Field(..., min_length=1, max_length=100),
    db: Any = Depends(get_db),
    models: dict = Depends(get_ml_models)
):
    """Get user statistics and preferences"""
    user_data = db.find_one({"user_id": user_id})
    
    if not user_data:
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            data={
                "total_watched": 0,
                "favorite_genres": [],
                "average_score": 0,
                "profile_created": None
            }
        )
    
    watched_ids = user_data.get("watched_list", [])
    if not watched_ids:
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            data={
                "total_watched": 0,
                "favorite_genres": [],
                "average_score": 0,
                "profile_created": user_data.get("created_at")
            }
        )
    
    # Calculate statistics
    anime_dict = models["anime_dict"]
    genres_count = {}
    total_score = 0
    valid_scores = 0
    
    for anime_id in watched_ids:
        if anime_id in anime_dict:
            anime = anime_dict[anime_id]
            
            # Count genres
            if anime.get("genres"):
                for genre in anime["genres"].split(","):
                    genre = genre.strip()
                    genres_count[genre] = genres_count.get(genre, 0) + 1
            
            # Sum scores
            if anime.get("score", 0) > 0:
                total_score += anime["score"]
                valid_scores += 1
    
    # Get top genres
    top_genres = sorted(genres_count.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return ApiResponse(
        status=ResponseStatus.SUCCESS,
        data={
            "total_watched": len(watched_ids),
            "favorite_genres": [g[0] for g in top_genres],
            "average_score": round(total_score / valid_scores, 2) if valid_scores > 0 else 0,
            "profile_created": user_data.get("created_at"),
            "last_updated": user_data.get("updated_at")
        }
    )

# --- Error Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": ResponseStatus.ERROR,
            "message": exc.detail,
            "data": None
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": ResponseStatus.ERROR,
            "message": "An unexpected error occurred",
            "data": None
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")