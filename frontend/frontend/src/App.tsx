import React, { useState, useEffect, createContext, useContext, useCallback, useMemo, memo } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useParams, useNavigate } from 'react-router-dom';

// --- Configuration ---
const getApiBaseUrl = (): string => {
  if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  return 'http://127.0.0.1:8000';
};

const API_BASE_URL = getApiBaseUrl();
const DEBOUNCE_DELAY = 300;
const RETRY_ATTEMPTS = 3;
const RETRY_DELAY = 1000;

// --- Type Definitions ---
interface Anime {
  mal_id: number;
  title: string;
  image_url: string;
  score: number;
  synopsis?: string;
  genres?: string;
}

interface Recommendation extends Anime {
  similarity_score: number;
}

interface UserStats {
  total_watched: number;
  favorite_genres: string[];
  average_score: number;
  profile_created?: string;
  last_updated?: string;
}

interface ApiResponse<T = any> {
  status: 'success' | 'error' | 'warning';
  data?: T;
  message?: string;
}

// --- Custom Hooks ---
const useDebounce = <T,>(value: T, delay: number): T => {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => clearTimeout(handler);
  }, [value, delay]);

  return debouncedValue;
};

const useLocalStorage = <T,>(key: string, initialValue: T): [T, (value: T) => void] => {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(`Error loading ${key} from localStorage:`, error);
      return initialValue;
    }
  });

  const setValue = useCallback((value: T) => {
    try {
      setStoredValue(value);
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error(`Error saving ${key} to localStorage:`, error);
    }
  }, [key]);

  return [storedValue, setValue];
};

// --- API Service ---
class ApiService {
  private static async fetchWithRetry(url: string, options?: RequestInit, retries = RETRY_ATTEMPTS): Promise<Response> {
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      if (!response.ok && retries > 0) {
        await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
        return this.fetchWithRetry(url, options, retries - 1);
      }

      return response;
    } catch (error) {
      if (retries > 0) {
        await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
        return this.fetchWithRetry(url, options, retries - 1);
      }
      throw error;
    }
  }

  static async getAllAnime(): Promise<Anime[]> {
    const response = await this.fetchWithRetry(`${API_BASE_URL}/anime`);
    if (!response.ok) throw new Error('Failed to fetch anime list');
    return response.json();
  }

  static async getAnimeDetails(id: string): Promise<Anime> {
    const response = await this.fetchWithRetry(`${API_BASE_URL}/anime/${id}`);
    if (!response.ok) throw new Error('Failed to fetch anime details');
    return response.json();
  }

  static async getRecommendations(animeId: string, userId: string): Promise<Recommendation[]> {
    const response = await this.fetchWithRetry(
      `${API_BASE_URL}/recommend/${animeId}?user_id=${encodeURIComponent(userId)}`
    );
    if (!response.ok) throw new Error('Failed to fetch recommendations');
    return response.json();
  }

  static async getProfileRecommendations(userId: string): Promise<Recommendation[]> {
    const response = await this.fetchWithRetry(
      `${API_BASE_URL}/profile/recommend/${encodeURIComponent(userId)}`
    );
    if (response.status === 404) return [];
    if (!response.ok) throw new Error('Failed to fetch profile recommendations');
    return response.json();
  }

  static async getWatchedList(userId: string): Promise<number[]> {
    const response = await this.fetchWithRetry(
      `${API_BASE_URL}/user/watched/${encodeURIComponent(userId)}`
    );
    if (!response.ok) throw new Error('Failed to fetch watched list');
    return response.json();
  }

  static async addToWatched(userId: string, animeId: number): Promise<ApiResponse> {
    const response = await this.fetchWithRetry(
      `${API_BASE_URL}/user/watched`,
      {
        method: 'POST',
        body: JSON.stringify({ user_id: userId, anime_id: animeId }),
      }
    );
    if (!response.ok) throw new Error('Failed to add to watched list');
    return response.json();
  }

  static async removeFromWatched(userId: string, animeId: number): Promise<ApiResponse> {
    const response = await this.fetchWithRetry(
      `${API_BASE_URL}/user/watched/${encodeURIComponent(userId)}/${animeId}`,
      { method: 'DELETE' }
    );
    if (!response.ok) throw new Error('Failed to remove from watched list');
    return response.json();
  }

  static async resetProfile(userId: string): Promise<ApiResponse> {
    const response = await this.fetchWithRetry(
      `${API_BASE_URL}/user/profile/${encodeURIComponent(userId)}/reset`,
      { method: 'DELETE' }
    );
    if (!response.ok) throw new Error('Failed to reset profile');
    return response.json();
  }

  static async getUserStats(userId: string): Promise<UserStats> {
    const response = await this.fetchWithRetry(
      `${API_BASE_URL}/user/stats/${encodeURIComponent(userId)}`
    );
    if (!response.ok) throw new Error('Failed to fetch user stats');
    const apiResponse: ApiResponse<UserStats> = await response.json();
    return apiResponse.data || { total_watched: 0, favorite_genres: [], average_score: 0 };
  }
}

// --- Watched List Context ---
interface WatchedListContextType {
  userId: string;
  watchedIds: Set<number>;
  userStats: UserStats | null;
  isLoading: boolean;
  error: string | null;
  addWatchedId: (id: number) => Promise<void>;
  removeWatchedId: (id: number) => Promise<void>;
  resetProfile: () => Promise<void>;
  refreshWatchedList: () => Promise<void>;
}

const WatchedListContext = createContext<WatchedListContextType | null>(null);

const useWatchedList = (): WatchedListContextType => {
  const context = useContext(WatchedListContext);
  if (!context) throw new Error("useWatchedList must be used within WatchedListProvider");
  return context;
};

const WatchedListProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [watchedIds, setWatchedIds] = useState<Set<number>>(new Set());
  const [userStats, setUserStats] = useState<UserStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [userId] = useLocalStorage<string>('anisugg_user_id', '');

  // Initialize userId if not exists
  useEffect(() => {
    if (!userId) {
      const newUserId = `user_${crypto.randomUUID()}`;
      localStorage.setItem('anisugg_user_id', JSON.stringify(newUserId));
      window.location.reload();
    }
  }, [userId]);

  const refreshWatchedList = useCallback(async () => {
    if (!userId) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const [watchedList, stats] = await Promise.all([
        ApiService.getWatchedList(userId),
        ApiService.getUserStats(userId)
      ]);
      
      setWatchedIds(new Set(watchedList));
      setUserStats(stats);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load watched list';
      setError(errorMessage);
      console.error('Error fetching watched list:', err);
    } finally {
      setIsLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    refreshWatchedList();
  }, [refreshWatchedList]);

  const addWatchedId = useCallback(async (id: number) => {
    try {
      await ApiService.addToWatched(userId, id);
      setWatchedIds(prev => new Set(prev).add(id));
      
      // Update stats
      const stats = await ApiService.getUserStats(userId);
      setUserStats(stats);
    } catch (err) {
      console.error('Failed to add to watched list:', err);
      throw err;
    }
  }, [userId]);

  const removeWatchedId = useCallback(async (id: number) => {
    try {
      await ApiService.removeFromWatched(userId, id);
      setWatchedIds(prev => {
        const newSet = new Set(prev);
        newSet.delete(id);
        return newSet;
      });
      
      // Update stats
      const stats = await ApiService.getUserStats(userId);
      setUserStats(stats);
    } catch (err) {
      console.error('Failed to remove from watched list:', err);
      throw err;
    }
  }, [userId]);

  const resetProfile = useCallback(async () => {
    try {
      await ApiService.resetProfile(userId);
      setWatchedIds(new Set());
      setUserStats({ total_watched: 0, favorite_genres: [], average_score: 0 });
    } catch (err) {
      console.error('Failed to reset profile:', err);
      throw err;
    }
  }, [userId]);

  const value = useMemo(() => ({
    userId,
    watchedIds,
    userStats,
    isLoading,
    error,
    addWatchedId,
    removeWatchedId,
    resetProfile,
    refreshWatchedList
  }), [userId, watchedIds, userStats, isLoading, error, addWatchedId, removeWatchedId, resetProfile, refreshWatchedList]);

  return (
    <WatchedListContext.Provider value={value}>
      {children}
    </WatchedListContext.Provider>
  );
};

// --- Components ---
const LoadingSpinner: React.FC = () => (
  <div className="flex justify-center items-center p-8">
    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
  </div>
);

const ErrorMessage: React.FC<{ message: string }> = ({ message }) => (
  <div className="bg-red-900/20 border border-red-500 text-red-300 px-4 py-3 rounded-lg">
    <p className="flex items-center">
      <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
      </svg>
      {message}
    </p>
  </div>
);

const ConfirmDialog: React.FC<{
  isOpen: boolean;
  title: string;
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
}> = ({ isOpen, title, message, onConfirm, onCancel }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4 shadow-xl border border-gray-700">
        <h3 className="text-xl font-bold mb-3 text-white">{title}</h3>
        <p className="text-gray-300 mb-6">{message}</p>
        <div className="flex gap-3 justify-end">
          <button
            onClick={onCancel}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
          >
            Reset Profile
          </button>
        </div>
      </div>
    </div>
  );
};

const UserStatsCard: React.FC = () => {
  const { userStats, isLoading } = useWatchedList();

  if (isLoading || !userStats) return null;

  return (
    <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
      <h3 className="text-lg font-semibold mb-3 text-indigo-400">Your Profile Stats</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div>
          <p className="text-gray-400">Anime Watched</p>
          <p className="text-2xl font-bold">{userStats.total_watched}</p>
        </div>
        <div>
          <p className="text-gray-400">Avg Score</p>
          <p className="text-2xl font-bold">{userStats.average_score || 'N/A'}</p>
        </div>
        <div className="col-span-2">
          <p className="text-gray-400 mb-1">Top Genres</p>
          <div className="flex flex-wrap gap-1">
            {userStats.favorite_genres.length > 0 ? (
              userStats.favorite_genres.map(genre => (
                <span key={genre} className="text-xs bg-gray-700 px-2 py-1 rounded">
                  {genre}
                </span>
              ))
            ) : (
              <span className="text-gray-500">Watch more anime to see your preferences</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const AnimeCard: React.FC<{ anime: Anime | Recommendation }> = memo(({ anime }) => {
  const isRecommendation = 'similarity_score' in anime;
  const [imageError, setImageError] = useState(false);

  const handleImageError = useCallback(() => {
    setImageError(true);
  }, []);

  const imageSrc = imageError || !anime.image_url 
    ? 'https://placehold.co/500x700/1f2937/7c3aed?text=No+Image'
    : anime.image_url;

  return (
    <Link
      to={`/anime/${anime.mal_id}`}
      className="group relative block bg-gray-800 rounded-lg overflow-hidden shadow-lg hover:shadow-indigo-500/30 transition-all duration-300 hover:scale-105"
    >
      <div className="relative aspect-[3/4]">
        <img
          src={imageSrc}
          alt={anime.title}
          className="w-full h-full object-cover"
          onError={handleImageError}
          loading="lazy"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent opacity-60 group-hover:opacity-80 transition-opacity"></div>
        
        {/* Score Badge */}
        {anime.score > 0 && (
          <div className="absolute top-2 right-2 bg-black/70 backdrop-blur-sm px-2 py-1 rounded-md">
            <span className="text-yellow-400 font-bold text-sm">★ {anime.score.toFixed(1)}</span>
          </div>
        )}
      </div>
      
      <div className="absolute bottom-0 left-0 right-0 p-3">
        <h3 className="text-white font-semibold text-sm truncate group-hover:text-indigo-400 transition-colors">
          {anime.title}
        </h3>
        {isRecommendation && (
          <p className="text-xs text-green-400 mt-1">
            {((anime as Recommendation).similarity_score * 100).toFixed(0)}% match
          </p>
        )}
      </div>

      {/* Hover Overlay */}
      <div className="absolute inset-0 p-4 bg-gray-900/95 backdrop-blur-sm text-white opacity-0 group-hover:opacity-100 transition-opacity duration-300 overflow-y-auto">
        <h4 className="font-bold text-indigo-400 mb-2">{anime.title}</h4>
        
        {anime.genres && (
          <div className="flex flex-wrap gap-1 mb-2">
            {anime.genres.split(',').slice(0, 3).map(genre => (
              <span key={genre.trim()} className="text-xs bg-gray-700 px-2 py-1 rounded">
                {genre.trim()}
              </span>
            ))}
          </div>
        )}
        
        <p className="text-xs text-gray-300 leading-relaxed line-clamp-6">
          {anime.synopsis || 'No synopsis available.'}
        </p>
      </div>
    </Link>
  );
});

AnimeCard.displayName = 'AnimeCard';

const RecommendationCarousel: React.FC = () => {
  const [profileRecs, setProfileRecs] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { userId, watchedIds, isLoading: isWatchedListLoading } = useWatchedList();

  useEffect(() => {
    if (isWatchedListLoading) return;

    const fetchProfileRecs = async () => {
      setLoading(true);
      setError(null);

      if (watchedIds.size === 0) {
        setError("Start adding anime to your watched list to get personalized recommendations!");
        setProfileRecs([]);
        setLoading(false);
        return;
      }

      try {
        const recs = await ApiService.getProfileRecommendations(userId);
        setProfileRecs(recs);
        
        if (recs.length === 0) {
          setError("Add more anime to improve your recommendations!");
        }
      } catch (err) {
        setError("Could not load personalized recommendations. Please try again later.");
        console.error('Failed to fetch profile recommendations:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchProfileRecs();
  }, [userId, watchedIds, isWatchedListLoading]);

  if (loading || isWatchedListLoading) {
    return (
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-4 border-l-4 border-indigo-500 pl-3">
          Recommended For You
        </h2>
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <div className="mb-8">
      <h2 className="text-2xl font-bold mb-4 border-l-4 border-indigo-500 pl-3">
        Recommended For You
      </h2>
      {error ? (
        <div className="bg-gray-800/50 backdrop-blur-sm p-6 rounded-lg text-center border border-gray-700">
          <p className="text-gray-400">{error}</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
          {profileRecs.map(rec => (
            <AnimeCard key={rec.mal_id} anime={rec} />
          ))}
        </div>
      )}
    </div>
  );
};

// --- Page Components ---
const HomePage: React.FC = () => {
  const [animeList, setAnimeList] = useState<Anime[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showResetDialog, setShowResetDialog] = useState(false);
  
  const { watchedIds, resetProfile, userStats } = useWatchedList();
  const debouncedSearchTerm = useDebounce(searchTerm, DEBOUNCE_DELAY);

  const filteredAnime = useMemo(() => {
    return animeList
      .filter(anime => anime.title.toLowerCase().includes(debouncedSearchTerm.toLowerCase()))
      .filter(anime => !watchedIds.has(anime.mal_id));
  }, [animeList, debouncedSearchTerm, watchedIds]);

  useEffect(() => {
    const fetchAllAnime = async () => {
      try {
        const data = await ApiService.getAllAnime();
        setAnimeList(data);
      } catch (err) {
        setError('Failed to load anime library. Please refresh the page.');
        console.error('Failed to fetch anime list:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchAllAnime();
  }, []);

  const handleReset = async () => {
    try {
      await resetProfile();
      setShowResetDialog(false);
      window.location.reload(); // Refresh to update all components
    } catch (err) {
      console.error('Failed to reset profile:', err);
      alert('Failed to reset profile. Please try again.');
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {/* Header */}
      <header className="text-center mb-12">
        <h1 className="text-5xl md:text-6xl font-bold mb-3 logo-font bg-gradient-to-r from-purple-400 via-indigo-500 to-cyan-400 text-transparent bg-clip-text animate-gradient">
          AniSuggest
        </h1>
        <p className="text-lg text-gray-400">AI-Powered Anime Discovery</p>
      </header>

      {/* User Stats and Actions */}
      <div className="mb-8 space-y-4">
        <UserStatsCard />
        
        {userStats && userStats.total_watched > 0 && (
          <div className="flex justify-end">
            <button
              onClick={() => setShowResetDialog(true)}
              className="px-4 py-2 bg-gray-700 hover:bg-red-600 rounded-lg transition-all duration-300 text-sm font-medium flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Reset Recommendations
            </button>
          </div>
        )}
      </div>

      {/* Personalized Recommendations */}
      <RecommendationCarousel />

      {/* Browse Section */}
      <div className="mt-12">
        <h2 className="text-2xl font-bold mb-4 border-l-4 border-indigo-500 pl-3">
          Browse All Anime
        </h2>
        
        {/* Search Bar */}
        <div className="mb-8 max-w-2xl">
          <div className="relative">
            <input
              type="text"
              placeholder="Search for anime..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              className="w-full p-4 pl-12 bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all"
            />
            <svg className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
        </div>

        {/* Anime Grid */}
        {error ? (
          <ErrorMessage message={error} />
        ) : loading ? (
          <LoadingSpinner />
        ) : (
          <>
            <p className="text-gray-400 mb-4">
              Showing {filteredAnime.length} anime {debouncedSearchTerm && `matching "${debouncedSearchTerm}"`}
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
              {filteredAnime.map(anime => (
                <AnimeCard key={anime.mal_id} anime={anime} />
              ))}
            </div>
          </>
        )}
      </div>

      {/* Reset Confirmation Dialog */}
      <ConfirmDialog
        isOpen={showResetDialog}
        title="Reset Your Profile?"
        message="This will clear your entire watched list and reset all personalized recommendations. This action cannot be undone."
        onConfirm={handleReset}
        onCancel={() => setShowResetDialog(false)}
      />
    </div>
  );
};

const DetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [animeDetails, setAnimeDetails] = useState<Anime | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved'>('idle');

  const { userId, watchedIds, addWatchedId, removeWatchedId } = useWatchedList();
  const isWatched = id ? watchedIds.has(Number(id)) : false;

  useEffect(() => {
    setSaveStatus(isWatched ? 'saved' : 'idle');
  }, [isWatched]);

  useEffect(() => {
    const fetchData = async () => {
      if (!id) return;

      setLoading(true);
      setError(null);

      try {
        const [details, recs] = await Promise.all([
          ApiService.getAnimeDetails(id),
          ApiService.getRecommendations(id, userId)
        ]);

        setAnimeDetails(details);
        setRecommendations(recs);
      } catch (err) {
        setError('Failed to load anime details. Please try again.');
        console.error('Failed to fetch anime data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [id, userId]);

  const handleWatchedToggle = async () => {
    if (!id) return;
    
    const animeId = Number(id);
    setSaveStatus('saving');

    try {
      if (isWatched) {
        await removeWatchedId(animeId);
        setSaveStatus('idle');
      } else {
        await addWatchedId(animeId);
        setSaveStatus('saved');
      }
    } catch (err) {
      console.error('Failed to update watched status:', err);
      setSaveStatus(isWatched ? 'saved' : 'idle');
    }
  };

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="container mx-auto px-4 py-8"><ErrorMessage message={error} /></div>;
  if (!animeDetails) return <div className="container mx-auto px-4 py-8"><ErrorMessage message="Anime not found" /></div>;

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {/* Back Button */}
      <button
        onClick={() => navigate(-1)}
        className="mb-6 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors inline-flex items-center gap-2"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        Back
      </button>

      {/* Anime Details */}
      <div className="grid md:grid-cols-3 gap-8 mb-12">
        <div className="md:col-span-1">
          <img
            src={animeDetails.image_url || 'https://placehold.co/500x700/1f2937/7c3aed?text=No+Image'}
            alt={animeDetails.title}
            className="w-full rounded-lg shadow-2xl"
            onError={(e) => {
              const target = e.target as HTMLImageElement;
              target.onerror = null;
              target.src = 'https://placehold.co/500x700/1f2937/7c3aed?text=No+Image';
            }}
          />
          
          {/* Action Buttons */}
          <div className="mt-4 space-y-3">
            <button
              onClick={handleWatchedToggle}
              disabled={saveStatus === 'saving'}
              className={`w-full py-3 px-4 rounded-lg font-semibold transition-all duration-300 flex items-center justify-center gap-2 ${
                isWatched
                  ? 'bg-green-600 hover:bg-red-600'
                  : saveStatus === 'saving'
                  ? 'bg-gray-600 cursor-wait'
                  : 'bg-indigo-600 hover:bg-indigo-700'
              }`}
            >
              {saveStatus === 'saving' ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                  Processing...
                </>
              ) : isWatched ? (
                <>
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  Watched (Click to Remove)
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  Add to Watched
                </>
              )}
            </button>
          </div>
        </div>

        <div className="md:col-span-2">
          <h1 className="text-4xl font-bold mb-4">{animeDetails.title}</h1>
          
          {/* Score and Genres */}
          <div className="flex flex-wrap items-center gap-4 mb-6">
            {animeDetails.score > 0 && (
              <div className="flex items-center gap-2 bg-yellow-500/20 px-3 py-1 rounded-full">
                <span className="text-yellow-400 text-lg">★</span>
                <span className="font-bold">{animeDetails.score.toFixed(2)}</span>
              </div>
            )}
            
            {animeDetails.genres && (
              <div className="flex flex-wrap gap-2">
                {animeDetails.genres.split(',').map(genre => (
                  <span key={genre.trim()} className="bg-gray-700 text-gray-300 text-sm px-3 py-1 rounded-full">
                    {genre.trim()}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Synopsis */}
          <div className="prose prose-invert max-w-none">
            <h2 className="text-xl font-semibold mb-3 text-indigo-400">Synopsis</h2>
            <p className="text-gray-300 leading-relaxed whitespace-pre-wrap">
              {animeDetails.synopsis || 'No synopsis available.'}
            </p>
          </div>
        </div>
      </div>

      <div>
        <h2 className="text-2xl font-bold mb-4 border-l-4 border-indigo-500 pl-3">
          Similar Anime You Might Like
        </h2>
        {recommendations.length === 0 ? (
          <p className="text-gray-400">
            No recommendations available for this anime. Try watching or adding more anime to improve results!
          </p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
            {recommendations.map((rec) => (
              <AnimeCard key={rec.mal_id} anime={rec} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// --- Main App Component ---
const App: React.FC = () => {
  return (
    <Router>
      <WatchedListProvider>
        <div className="min-h-screen bg-gray-900 text-gray-100">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/anime/:id" element={<DetailPage />} />
          </Routes>
        </div>
      </WatchedListProvider>
    </Router>
  );
};

export default App;