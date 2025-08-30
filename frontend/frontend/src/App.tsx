import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useParams, useLocation } from 'react-router-dom';

// --- Type Definitions ---
interface Anime {
  mal_id: number;
  title: string;
  synopsis: string;
  image_url: string;
  score: number; // This is the MAL score
}

interface Recommendation extends Anime {
  similarity_score: number;
}

// --- Component: AnimeCard ---
interface AnimeCardProps {
  anime: Anime | Recommendation;
}

const AnimeCard: React.FC<AnimeCardProps> = ({ anime }) => {
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
            target.src='https://placehold.co/500x700/1f2937/7c3aed?text=No+Image';
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
        {isRecommendation && <p className="text-xs mb-2 text-green-400">Similarity: {((anime as Recommendation).similarity_score * 100).toFixed(1)}%</p>}
        <p className="text-xs text-gray-300 leading-relaxed">
          {anime.synopsis?.substring(0, 250)}{anime.synopsis && anime.synopsis.length > 250 ? '...' : ''}
        </p>
      </div>
    </Link>
  );
};

// --- Page: HomePage ---
const HomePage: React.FC = () => {
  const [allAnime, setAllAnime] = useState<Anime[]>([]);
  const [filteredAnime, setFilteredAnime] = useState<Anime[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch all anime data on initial load
    const fetchAllAnime = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/anime');
        if (!response.ok) throw new Error('Network response was not ok');
        const data: Anime[] = await response.json();
        setAllAnime(data);
        setFilteredAnime(data.slice(0, 20)); // Show top 20 initially
      } catch (error) {
        console.error("Failed to fetch anime:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchAllAnime();
  }, []);

  useEffect(() => {
    // Filter anime based on search term
    if (!searchTerm) {
      setFilteredAnime(allAnime.slice(0, 20)); // Reset to initial view if search is cleared
      return;
    }
    const lowercasedTerm = searchTerm.toLowerCase();
    const results = allAnime.filter(anime =>
      anime.title.toLowerCase().includes(lowercasedTerm)
    );
    setFilteredAnime(results);
  }, [searchTerm, allAnime]);

  return (
    <div className="container mx-auto p-4 md:p-8">
      <header className="text-center mb-8">
        <div className="text-6xl font-bold text-white mb-2" style={{ fontFamily: "'Orbitron', sans-serif" }}>
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-indigo-600">
            AniSugg
          </span>
        </div>
        <p className="text-gray-400">Find your next favorite show</p>
      </header>

      <div className="mb-8 max-w-lg mx-auto">
        <input
          type="text"
          placeholder="Search for an anime..."
          className="w-full p-3 rounded-md bg-gray-800 text-white border border-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      {loading ? (
        <p className="text-center text-gray-400">Loading anime database...</p>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
          {filteredAnime.map((anime) => (
            <AnimeCard key={anime.mal_id} anime={anime} />
          ))}
        </div>
      )}
    </div>
  );
};

// --- Page: DetailPage ---
const DetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const { title } = location.state || { title: `Anime ID: ${id}` };
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!id) return;
    const fetchRecommendations = async () => {
      setLoading(true);
      try {
        const response = await fetch(`http://127.0.0.1:8000/recommend/${id}`);
        if (!response.ok) throw new Error('Request failed.');
        const data = await response.json();
        // The backend returns similarity_score as "score" and mal_score as "mal_score"
        const formattedData = data.map((rec: any) => ({ 
            ...rec, 
            similarity_score: rec.score, 
            score: rec.mal_score 
        }));
        setRecommendations(formattedData);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchRecommendations();
  }, [id]);

  return (
    <div className="container mx-auto p-4 md:p-8">
      <h2 className="text-3xl font-bold text-white mb-6">Recommendations based on "{title}"</h2>

      {loading ? (
        <p className="text-center text-gray-400">Loading recommendations...</p>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
          {recommendations.map((rec) => (
            <AnimeCard key={rec.mal_id} anime={rec} />
          ))}
        </div>
      )}
    </div>
  );
};

// --- Main App Component ---
function App() {
  return (
    <Router>
      <div className="bg-gray-900 min-h-screen text-white">
        <main>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/anime/:id" element={<DetailPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;

