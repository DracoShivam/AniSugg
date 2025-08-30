import React, { useState, useEffect, createContext, useContext } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useParams } from 'react-router-dom';

// --- Configuration ---
const getApiBaseUrl = () => {
  if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  return 'http://127.0.0.1:8000';
};
const API_BASE_URL = getApiBaseUrl();

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

// --- Watched List Context for Global State ---
interface WatchedListContextType {
  userId: string;
  watchedIds: Set<number>;
  addWatchedId: (id: number) => void;
  isLoading: boolean;
}

const WatchedListContext = createContext<WatchedListContextType | null>(null);

const useWatchedList = () => {
  const context = useContext(WatchedListContext);
  if (!context) throw new Error("useWatchedList must be used within a WatchedListProvider");
  return context;
};

const WatchedListProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [watchedIds, setWatchedIds] = useState<Set<number>>(new Set());
  const [isLoading, setIsLoading] = useState(true);

  // Generate or load unique userId for each browser
  const [userId] = useState<string>(() => {
    let existing = localStorage.getItem("anisugg_user_id");
    if (!existing) {
      existing = `user_${crypto.randomUUID()}`;
      localStorage.setItem("anisugg_user_id", existing);
    }
    return existing;
  });

  // Fetch watched list from backend
  useEffect(() => {
    const fetchWatchedList = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/user/watched/${userId}`);
        if (!response.ok) throw new Error("Could not fetch watched list.");
        const data: number[] = await response.json();
        setWatchedIds(new Set(data));
      } catch (error) {
        console.error(error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchWatchedList();
  }, [userId]);

  const addWatchedId = (id: number) => {
    setWatchedIds(prev => new Set(prev).add(id));
  };

  return (
    <WatchedListContext.Provider value={{ userId, watchedIds, addWatchedId, isLoading }}>
      {children}
    </WatchedListContext.Provider>
  );
};

// --- Main App Component ---
function App() {
  return (
    <WatchedListProvider>
      <Router>
        <div className="bg-gray-900 text-white min-h-screen font-sans">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/anime/:id" element={<DetailPage />} />
          </Routes>
        </div>
      </Router>
    </WatchedListProvider>
  );
}

// --- Page Components ---
const HomePage: React.FC = () => {
  const [animeList, setAnimeList] = useState<Anime[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const { watchedIds } = useWatchedList();

  const filteredAnime = animeList
    .filter(anime => anime.title.toLowerCase().includes(searchTerm.toLowerCase()))
    .filter(anime => !watchedIds.has(anime.mal_id));

  useEffect(() => {
    const fetchAllAnime = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/anime`);
        if (!response.ok) throw new Error('Network response was not ok');
        const data: Anime[] = await response.json();
        setAnimeList(data);
      } catch (error) {
        console.error("Failed to fetch anime list:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchAllAnime();
  }, []);

  return (
    <div className="container mx-auto px-4 py-8">
      <header className="text-center mb-12">
        <h1 className="text-6xl font-bold mb-2 logo-font bg-gradient-to-r from-purple-400 to-indigo-600 text-transparent bg-clip-text">AniSugg</h1>
        <p className="text-lg text-gray-400">Find your next favorite show</p>
      </header>

      <RecommendationCarousel />

      <div className="mt-12">
        <h2 className="text-2xl font-bold mb-4 border-l-4 border-indigo-500 pl-3">Browse All Anime</h2>
        <div className="mb-8 max-w-2xl">
          <input
            type="text"
            placeholder="Search for an anime..."
            className="w-full p-4 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
            onChange={e => setSearchTerm(e.target.value)}
          />
        </div>
        {loading ? (
          <p className="text-center">Loading anime library...</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
            {filteredAnime.map(anime => (
              <AnimeCard key={anime.mal_id} anime={anime} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const DetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [animeDetails, setAnimeDetails] = useState<Anime | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { userId, watchedIds, addWatchedId } = useWatchedList();
  const isAlreadySaved = id ? watchedIds.has(Number(id)) : false;

  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved'>(
    isAlreadySaved ? 'saved' : 'idle'
  );

  useEffect(() => {
    setSaveStatus(isAlreadySaved ? 'saved' : 'idle');
  }, [isAlreadySaved]);

  useEffect(() => {
    setLoading(true);
    const fetchDetailsAndRecommendations = async () => {
      if (!id) return;
      try {
        const [detailsRes, recsRes] = await Promise.all([
          fetch(`${API_BASE_URL}/anime/${id}`),
          fetch(`${API_BASE_URL}/recommend/${id}?user_id=${userId}`)
        ]);

        if (!detailsRes.ok) throw new Error('Failed to fetch anime details.');
        if (!recsRes.ok) throw new Error('Failed to fetch recommendations.');
        
        const detailsData = await detailsRes.json();
        const recsData = await recsRes.json();
        
        setAnimeDetails(detailsData);
        setRecommendations(recsData);

      } catch (err) {
        setError(err instanceof Error ? err.message : 'An unknown error occurred');
      } finally {
        setLoading(false);
      }
    };
    fetchDetailsAndRecommendations();
  }, [id, userId]);

  const handleAddToWatched = async () => {
    if (!id || saveStatus !== 'idle') return;
    setSaveStatus('saving');
    try {
      const response = await fetch(`${API_BASE_URL}/user/watched`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, anime_id: Number(id) }),
      });
      if (!response.ok) throw new Error('Failed to save.');
      addWatchedId(Number(id));
      setSaveStatus('saved');
    } catch (err) {
      console.error("Failed to add to watched list:", err);
      setSaveStatus('idle');
    }
  };
  
  if (loading) return <p className="text-center p-8">Loading details...</p>;
  if (error) return <p className="text-center p-8 text-red-500">Error: {error}</p>;
  if (!animeDetails) return <p className="text-center p-8">Anime not found.</p>;

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex flex-col md:flex-row gap-8 mb-12">
        <img 
          src={animeDetails.image_url} 
          alt={animeDetails.title} 
          className="w-full md:w-1/4 h-auto object-cover rounded-lg shadow-lg"
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            target.onerror = null; 
            target.src = 'https://placehold.co/500x700/1f2937/7c3aed?text=No+Image';
          }}
        />
        <div className="flex-1">
          <div className="flex justify-between items-start">
            <h1 className="text-4xl font-bold mb-4">{animeDetails.title}</h1>
            <div className="flex items-center gap-4 flex-shrink-0">
              <Link to="/" className="px-6 py-2 rounded-lg font-semibold bg-gray-700 hover:bg-gray-600 transition-colors">&larr; Back to Home</Link>
              <button
                onClick={handleAddToWatched}
                disabled={saveStatus !== 'idle'}
                className={`px-6 py-2 rounded-lg font-semibold transition-colors w-36 text-center ${
                  saveStatus === 'saved' ? 'bg-green-600 cursor-not-allowed' :
                  saveStatus === 'saving' ? 'bg-gray-500 cursor-wait' :
                  'bg-indigo-600 hover:bg-indigo-700'
                }`}
              >
                {saveStatus === 'saved' ? 'Saved!' :
                 saveStatus === 'saving' ? 'Saving...' :
                 'Add to Watched'}
              </button>
            </div>
          </div>
          <div className="flex items-center gap-4 mb-4">
            <span className="text-yellow-400 font-bold text-lg">MAL Score: {animeDetails.score.toFixed(2)}</span>
          </div>
          <div className="flex flex-wrap gap-2 mb-4">
            {animeDetails.genres?.split(',').map(genre => (
              <span key={genre.trim()} className="bg-gray-700 text-gray-300 text-xs font-semibold px-3 py-1 rounded-full">{genre.trim()}</span>
            ))}
          </div>
          <p className="text-gray-300 leading-relaxed">{animeDetails.synopsis}</p>
        </div>
      </div>
      
      <div>
        <h2 className="text-3xl font-bold mb-6">Recommendations based on "{animeDetails.title}"</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
          {recommendations.map(rec => (
            <AnimeCard key={rec.mal_id} anime={rec} />
          ))}
        </div>
      </div>
    </div>
  );
};

// --- Reusable Components ---
const RecommendationCarousel: React.FC = () => {
  const [profileRecs, setProfileRecs] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { userId, watchedIds, isLoading: isWatchedListLoading } = useWatchedList();

  useEffect(() => {
    if (isWatchedListLoading) return;

    const fetchProfileRecs = async () => {
      setLoading(true);
      if (watchedIds.size === 0) {
        setError("Add some anime to your watched list to get personalized recommendations!");
        setProfileRecs([]);
        setLoading(false);
        return;
      }
      try {
        const response = await fetch(`${API_BASE_URL}/profile/recommend/${userId}`);
        if (response.status === 404) {
          setError("Add more anime to your watched list to improve personalized recommendations!");
          setProfileRecs([]);
          return;
        }
        if (!response.ok) throw new Error('Failed to fetch recommendations.');
        const data = await response.json();
        setProfileRecs(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Could not load recommendations.');
      } finally {
        setLoading(false);
      }
    };
    fetchProfileRecs();
  }, [watchedIds, isWatchedListLoading, userId]);

  if (loading || isWatchedListLoading) {
    return <p>Loading your personalized recommendations...</p>;
  }

  return (
    <div className="mb-8">
       <h2 className="text-2xl font-bold mb-4 border-l-4 border-indigo-500 pl-3">Recommended For You</h2>
      {error ? (
        <div className="bg-gray-800 p-4 rounded-lg text-center text-gray-400">{error}</div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
          {profileRecs.map(rec => (
            <AnimeCard key={rec.mal_id} anime={rec} />
          ))}
        </div>
      )}
    </div>
  );
};

const AnimeCard: React.FC<{ anime: Anime | Recommendation }> = ({ anime }) => {
  const isRecommendation = 'similarity_score' in anime;

  return (
    <Link
      to={`/anime/${anime.mal_id}`}
      state={{ title: anime.title }}
      className="group relative block bg-gray-800 rounded-lg overflow-hidden shadow-lg hover:shadow-indigo-500/50 transition-shadow duration-300"
    >
      <div className="relative">
        <img
          src={anime.image_url}
          alt={`Poster for ${anime.title}`}
          className="w-full h-72 object-cover transition-transform duration-300 group-hover:scale-105"
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            target.onerror = null; 
            target.src = 'https://placehold.co/500x700/1f2937/7c3aed?text=No+Image';
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent"></div>
      </div>
      <div className="p-3">
        <h3 className="text-white text-sm font-semibold truncate group-hover:text-indigo-400" title={anime.title}>
          {anime.title}
        </h3>
      </div>
      <div className="absolute inset-0 p-4 bg-black/80 backdrop-blur-sm text-white opacity-0 group-hover:opacity-100 transition-opacity duration-300 overflow-y-auto rounded-lg">
        <h4 className="font-bold text-indigo-400 mb-2">{anime.title}</h4>
        
        {anime.score > 0 && <p className="text-xs mb-2 text-yellow-400">MAL Score: {anime.score.toFixed(2)}</p>}
        
        {isRecommendation && (
          <p className="text-xs mb-2 text-green-400">
            Similarity: {((anime as Recommendation).similarity_score * 100).toFixed(1)}%
          </p>
        )}
        
        <p className="text-xs text-gray-300 leading-relaxed">
          {anime.synopsis?.substring(0, 250)}{anime.synopsis && anime.synopsis.length > 250 ? '...' : ''}
        </p>
      </div>
    </Link>
  );
};

// --- Styles ---
const style = document.createElement('style');
style.textContent = `
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');
  .logo-font {
    font-family: 'Orbitron', sans-serif;
  }
`;
document.head.appendChild(style);

export default App;