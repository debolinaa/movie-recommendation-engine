import os
import json
import requests
import pandas as pd
import numpy as np
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from difflib import get_close_matches

# ---------------- Load Environment ----------------
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# ---------------- Paths ----------------
# Get the correct paths relative to the ui folder
SRC_DIR = os.path.dirname(os.path.dirname(__file__))
  # Go up from ui to src
DATA_DIR = os.path.join(SRC_DIR, "data")
MOVIES_CSV = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")
TAGS_CSV = os.path.join(DATA_DIR, "tags.csv")
LINKS_CSV = os.path.join(DATA_DIR, "links.csv")
CACHE_FILE = os.path.join(DATA_DIR, "tmdb_cache.json")
NO_COVER_PATH = os.path.join(SRC_DIR, "ui", "assets", "cover-not-found.jpg")

# Debug: Print the actual paths being used
print(f"Current working directory: {os.getcwd()}")
print(f"SRC_DIR: {SRC_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"MOVIES_CSV: {MOVIES_CSV}")
print(f"NO_COVER_PATH: {NO_COVER_PATH}")

# ---------------- Load Data ----------------
print("Loading movie data...")
try:
    # Load movies with optimized settings
    movies_df = pd.read_csv(MOVIES_CSV, usecols=['movieId', 'title', 'genres'])
    print(f"Loaded {len(movies_df)} movies")
    
    # Load ratings with chunking to avoid memory issues
    print("Loading ratings data...")
    ratings_df = pd.read_csv(RATINGS_CSV, usecols=['userId', 'movieId', 'rating'])
    print(f"Loaded {len(ratings_df)} ratings")
    
    # Load tags with optimized settings
    print("Loading tags data...")
    tags_df = pd.read_csv(TAGS_CSV, usecols=['movieId', 'tag'])
    print(f"Loaded {len(tags_df)} tags")
    
    # Load links with optimized settings
    print("Loading links data...")
    links_df = pd.read_csv(LINKS_CSV, usecols=['movieId', 'tmdbId'])
    print(f"Loaded {len(links_df)} links")
    
except Exception as e:
    print(f"Error loading data: {e}")
    # Fallback: create minimal dataframes
    movies_df = pd.DataFrame({'movieId': [], 'title': [], 'genres': []})
    ratings_df = pd.DataFrame({'userId': [], 'movieId': [], 'rating': []})
    tags_df = pd.DataFrame({'movieId': [], 'tag': []})
    links_df = pd.DataFrame({'movieId': [], 'tmdbId': []})

# Merge tags
print("Processing tags...")
tags_grouped = tags_df.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
movies_df = movies_df.merge(tags_grouped, on="movieId", how="left")
movies_df['tag'] = movies_df['tag'].fillna("")

# Merge TMDb links
print("Processing TMDb links...")
movies_df = movies_df.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')

# ---------------- Content for similarity ----------------
print("Preparing content for similarity...")
movies_df['genres'] = movies_df['genres'].fillna("")
movies_df['tag'] = movies_df['tag'].fillna("")
movies_df['content'] = movies_df['genres'] + " " + movies_df['tag']

# Limit content length to prevent memory issues
movies_df['content'] = movies_df['content'].str[:500]  # Limit to 500 characters

# ---------------- Content-based similarity ----------------
print("Computing TF-IDF matrix...")
try:
    # Use only genres for content similarity to reduce memory usage
    movies_df['content'] = movies_df['genres'].fillna("")
    
    # Limit features and use more memory-efficient settings
    tfidf = TfidfVectorizer(
        max_features=500,  # Reduced from 1000
        stop_words='english',
        ngram_range=(1, 2),  # Only unigrams and bigrams
        min_df=2,  # Minimum document frequency
        max_df=0.95  # Maximum document frequency
    )
    
    tfidf_matrix = tfidf.fit_transform(movies_df['content'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
except MemoryError as e:
    print(f"Memory error with TF-IDF: {e}")
    print("Using simplified genre-based approach...")
    # Fallback: simple genre matching without TF-IDF
    tfidf_matrix = None
    print("TF-IDF disabled, using simple genre matching")
except Exception as e:
    print(f"Error computing TF-IDF: {e}")
    tfidf_matrix = None
    print("TF-IDF disabled, using simple genre matching")

indices = pd.Series(movies_df.index, index=movies_df['title'].str.lower()).drop_duplicates()

def _highlight_match(title: str, query: str):
    """Return HTML-highlighted title where query matches, case-insensitive."""
    try:
        if not query:
            return title
        start = title.lower().find(query.lower())
        if start == -1:
            return title
        end = start + len(query)
        return f"{title[:start]}<strong>{title[start:end]}</strong>{title[end:]}"
    except Exception:
        return title

def get_live_suggestions(query: str, limit: int = 8):
    """Build live suggestions from movie titles with highlighted matches.
    Returns a tuple: (dropdown_update, html_list)
    """
    try:
        if not isinstance(query, str) or len(query.strip()) < 2:
            return gr.update(choices=[], value=None, visible=False), ""

        q = query.strip().lower()
        titles = movies_df['title'].astype(str)
        lower = titles.str.lower()

        # Prioritize: startswith > contains > fuzzy close matches
        starts = titles[lower.str.startswith(q)]
        contains = titles[(~lower.str.startswith(q)) & (lower.str.contains(q))]

        # Fallback fuzzy for diversity if needed
        remaining = max(0, limit - len(starts) - len(contains))
        fuzz = []
        if remaining > 0:
            try:
                fuzz = get_close_matches(q, lower.tolist(), n=remaining, cutoff=0.6)
                fuzz_map = {t: titles.iloc[i] for i, t in enumerate(lower.tolist())}
                fuzz = [fuzz_map[f] for f in fuzz if fuzz_map.get(f) not in set(starts).union(set(contains))]
            except Exception:
                fuzz = []

        combined = pd.concat([starts, contains, pd.Series(fuzz)], ignore_index=True).drop_duplicates()[:limit]
        choices = combined.tolist()

        # Build HTML list with highlight
        items = []
        for t in choices:
            items.append(f"<li class='suggest-item'><span class='icon'>🔍</span><span>{_highlight_match(t, query)}</span></li>")
        html = "" if not items else (
            "<div class='suggestions'><ul>" + "".join(items) + "</ul></div>"
        )

        return gr.update(choices=choices, value=None, visible=len(choices) > 0), html
    except Exception as e:
        print(f"get_live_suggestions error: {e}")
        return gr.update(choices=[], value=None, visible=False), ""

def compute_user_similarity(user_ratings, target_user_ratings):
    """
    Compute similarity between users based on their rating patterns
    """
    # Find common movies rated by both users
    common_movies = set(user_ratings.index) & set(target_user_ratings.index)
    if len(common_movies) < 3:  # Need at least 3 common movies
        return 0.0
    
    # Get ratings for common movies
    user_common = user_ratings.loc[list(common_movies)]
    target_common = target_user_ratings.loc[list(common_movies)]
    
    # Compute Pearson correlation
    correlation = np.corrcoef(user_common, target_common)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0

def find_similar_users(target_movie_id, ratings_df, top_k=50):
    """
    Find users who rated the target movie similarly to the current user
    """
    # Get users who rated the target movie
    target_ratings = ratings_df[ratings_df['movieId'] == target_movie_id]
    
    if len(target_ratings) == 0:
        return pd.DataFrame()
    
    # Get average rating for the target movie
    avg_target_rating = target_ratings['rating'].mean()
    
    # Find users with similar rating patterns
    similar_users = []
    
    for user_id in ratings_df['userId'].unique():
        if user_id in target_ratings['userId'].values:
            continue  # Skip the target user
            
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        if len(user_ratings) < 3:  # Reduced minimum ratings requirement
            continue
            
        # Check if user rated the target movie
        user_target_rating = user_ratings[user_ratings['movieId'] == target_movie_id]
        if len(user_target_rating) > 0:
            rating_diff = abs(user_target_rating['rating'].iloc[0] - avg_target_rating)
            # More flexible rating similarity (within 2.0 points instead of 1.5)
            if rating_diff <= 2.0:
                similar_users.append({
                    'userId': user_id,
                    'rating': user_target_rating['rating'].iloc[0],
                    'rating_diff': rating_diff,
                    'total_ratings': len(user_ratings)
                })
    
    similar_users_df = pd.DataFrame(similar_users)
    if len(similar_users_df) > 0:
        # Sort by rating similarity and total ratings
        similar_users_df['score'] = (1 / (1 + similar_users_df['rating_diff'])) * np.log(similar_users_df['total_ratings'] + 1)
        similar_users_df = similar_users_df.sort_values('score', ascending=False).head(top_k)
    
    return similar_users_df

def get_collaborative_recommendations(target_movie_id, similar_users_df, ratings_df, movies_df, top_k=20):
    """
    Get recommendations based on what similar users liked
    """
    if len(similar_users_df) == 0:
        return pd.DataFrame()
    
    # Get movies rated by similar users
    similar_user_ids = similar_users_df['userId'].tolist()
    similar_user_ratings = ratings_df[ratings_df['userId'].isin(similar_user_ids)]
    
    # Calculate weighted average ratings for movies
    movie_scores = []
    
    for movie_id in similar_user_ratings['movieId'].unique():
        if movie_id == target_movie_id:
            continue  # Skip the target movie
            
        movie_ratings = similar_user_ratings[similar_user_ratings['movieId'] == movie_id]
        if len(movie_ratings) < 2:  # Need at least 2 ratings
            continue
            
        # Calculate weighted average rating
        total_weight = 0
        weighted_sum = 0
        
        for _, rating_row in movie_ratings.iterrows():
            user_id = rating_row['userId']
            user_similarity = similar_users_df[similar_users_df['userId'] == user_id]['score'].iloc[0]
            total_weight += user_similarity
            weighted_sum += rating_row['rating'] * user_similarity
        
        if total_weight > 0:
            weighted_avg = weighted_sum / total_weight
            movie_scores.append({
                'movieId': movie_id,
                'weighted_rating': weighted_avg,
                'num_ratings': len(movie_ratings)
            })
    
    if not movie_scores:
        return pd.DataFrame()
    
    # Sort by weighted rating and number of ratings
    movie_scores_df = pd.DataFrame(movie_scores)
    movie_scores_df['score'] = movie_scores_df['weighted_rating'] * np.log(movie_scores_df['num_ratings'] + 1)
    movie_scores_df = movie_scores_df.sort_values('score', ascending=False).head(top_k)
    
    # Merge with movie information
    recommendations = movie_scores_df.merge(movies_df[['movieId', 'title', 'genres', 'tmdbId']], on='movieId')
    return recommendations

def get_genre_based_recommendations(target_genres, ratings_df, movies_df, target_movie_id, top_k=20):
    """
    Get recommendations based on what users who like similar genres rated highly
    """
    # Find movies with similar genres
    genre_matches = []
    for _, movie in movies_df.iterrows():
        if movie['movieId'] == target_movie_id:
            continue
            
        movie_genres = movie['genres'].split('|')
        # Calculate genre overlap
        overlap = len(set(target_genres) & set(movie_genres))
        if overlap > 0:
            genre_matches.append({
                'movieId': movie['movieId'],
                'title': movie['title'],
                'genres': movie['genres'],
                'tmdbId': movie['tmdbId'],
                'genre_overlap': overlap
            })
    
    if not genre_matches:
        return pd.DataFrame()
    
    # Get average ratings for these movies
    genre_movies_df = pd.DataFrame(genre_matches)
    movie_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    movie_ratings.columns = ['movieId', 'avg_rating', 'rating_count']
    
    # Merge with genre information
    recommendations = genre_movies_df.merge(movie_ratings, on='movieId', how='left')
    recommendations = recommendations.fillna({'avg_rating': 0, 'rating_count': 0})
    
    # Filter by minimum rating count and calculate score
    recommendations = recommendations[recommendations['rating_count'] >= 3]
    recommendations['score'] = (recommendations['genre_overlap'] * 0.4 + 
                              recommendations['avg_rating'] * 0.6) * np.log(recommendations['rating_count'] + 1)
    
    # Sort and return top k
    recommendations = recommendations.sort_values('score', ascending=False).head(top_k)
    recommendations['weighted_rating'] = recommendations['avg_rating']  # For compatibility
    
    return recommendations[['movieId', 'title', 'genres', 'tmdbId', 'weighted_rating', 'score']]

def compute_similarity_scores(target_idx, top_k=20):
    """
    Compute content-based similarity scores for a single movie.
    This is much more memory efficient.
    """
    if tfidf_matrix is not None:
        # Use TF-IDF similarity
        target_vector = tfidf_matrix[target_idx:target_idx+1]
        similarities = cosine_similarity(target_vector, tfidf_matrix).flatten()
        
        # Get top k similar movies (excluding the target movie itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        return similar_indices, similarities[similar_indices]
    else:
        # Fallback: simple genre-based similarity
        target_genres = set(movies_df.iloc[target_idx]['genres'].split('|'))
        
        genre_scores = []
        for idx, row in movies_df.iterrows():
            if idx == target_idx:
                continue
                
            movie_genres = set(row['genres'].split('|'))
            # Calculate Jaccard similarity
            intersection = len(target_genres & movie_genres)
            union = len(target_genres | movie_genres)
            similarity = intersection / union if union > 0 else 0
            
            genre_scores.append((idx, similarity))
        
        # Sort by similarity and return top k
        genre_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in genre_scores[:top_k]]
        top_scores = [score for _, score in genre_scores[:top_k]]
        
        return top_indices, top_scores

def clean_cache():
    """Clean up corrupted cache entries"""
    global poster_cache
    cleaned_cache = {}
    for key, value in poster_cache.items():
        # Only keep valid TMDB poster URLs
        if (isinstance(value, str) and 
            value.startswith("https://image.tmdb.org") and 
            not value.startswith("https://via.placeholder.com") and
            not value.startswith("C:\\") and
            not value.startswith("file://")):
            cleaned_cache[key] = value
    
    poster_cache = cleaned_cache
    # Save cleaned cache
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(poster_cache, f)
    except Exception as e:
        print(f"Error saving cleaned cache: {e}")

# ---------------- Poster cache ----------------
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        poster_cache = json.load(f)
else:
    poster_cache = {}

# Clean up corrupted cache entries on startup
clean_cache()

def fetch_tmdb_poster(tmdb_id, title):
    """
    Fetch movie poster from TMDB API with improved error handling and fallbacks.
    Returns a consistent URL format for the poster.
    """
    # Clean up the cache key
    cache_key = str(tmdb_id) if tmdb_id and not pd.isna(tmdb_id) else title
    
    # Check cache first
    if cache_key in poster_cache:
        cached_url = poster_cache[cache_key]
        # Validate cached URL - if it's a placeholder or invalid path, remove it
        if (cached_url.startswith("https://via.placeholder.com") or 
            cached_url.startswith("C:\\") or 
            not cached_url.startswith("http")):
            del poster_cache[cache_key]
        else:
            return cached_url
    
    poster_url = None
    
    # Try to fetch by TMDB ID first (more reliable)
    if tmdb_id and not pd.isna(tmdb_id):
        try:
            tmdb_id_str = str(int(tmdb_id))
            response = requests.get(
                f"{TMDB_BASE_URL}/movie/{tmdb_id_str}", 
                params={"api_key": TMDB_API_KEY}, 
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                poster_path = data.get("poster_path")
                if poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        except Exception as e:
            print(f"Error fetching poster for TMDB ID {tmdb_id}: {e}")
    
    # Fallback: search by title if no poster found by ID
    if not poster_url:
        try:
            response = requests.get(
                f"{TMDB_BASE_URL}/search/movie", 
                params={"api_key": TMDB_API_KEY, "query": title}, 
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if results:
                    poster_path = results[0].get("poster_path")
                    if poster_path:
                        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        except Exception as e:
            print(f"Error searching poster for title '{title}': {e}")
    
    # Final fallback: use local cover image
    if not poster_url:
        # Convert local path to file:// URL for web display
        if os.path.exists(NO_COVER_PATH):
            poster_url = f"file:///{NO_COVER_PATH.replace(os.sep, '/')}"
        else:
            # If local file doesn't exist, use a placeholder
            poster_url = "https://via.placeholder.com/200x300?text=No+Image"
    
    # Cache the result
    poster_cache[cache_key] = poster_url
    
    # Save cache to file
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(poster_cache, f)
    except Exception as e:
        print(f"Error saving cache: {e}")
    
    return poster_url
def get_mood_genres(mood):
    """
    Maps mood selections to genres.
    """
    mood_map = {
        "Happy": ["Comedy", "Family", "Animation"],
        "Sad": ["Drama", "Romance"],
        "Adventurous": ["Adventure", "Action"],
        "Romantic": ["Romance"],
        "Thriller": ["Thriller", "Horror", "Mystery"],
        "Chill": ["Comedy", "Drama"]
    }
    return mood_map.get(mood, [])

# ---------------- Recommendation Function ----------------
def recommend_movies(movie_name, min_rating,mood):
    try:
        print(f"Generating recommendations for: {movie_name}")
        
        # Fuzzy match
        all_titles = movies_df['title'].str.lower().tolist()
        matches = get_close_matches(movie_name.lower(), all_titles, n=1, cutoff=0.6)
        if not matches:
            return "<h3 style='text-align:center; color:white;'>No movie found with that name!</h3>"
        
        matched_title = matches[0]
        print(f"Matched movie: {matched_title}")
        
        idx = indices[matched_title]
        print(f"Movie index: {idx}")

        # Get movie ID for collaborative filtering
        target_movie_id = movies_df.loc[idx, 'movieId']
        print(f"Target movie ID: {target_movie_id}")

        # 1. Content-based recommendations (genres + tags)
        print("Computing content-based similarity...")
        similar_indices, similarity_scores = compute_similarity_scores(idx, top_k=15)
        content_recs = movies_df.iloc[similar_indices].copy()
        content_recs['content_score'] = similarity_scores
        print(f"Found {len(content_recs)} content-based recommendations")

        # 2. Collaborative filtering recommendations (similar users)
        print("Computing collaborative filtering...")
        similar_users = find_similar_users(target_movie_id, ratings_df, top_k=30)
        collab_recs = get_collaborative_recommendations(target_movie_id, similar_users, ratings_df, movies_df, top_k=15)
        print(f"Found {len(collab_recs)} collaborative filtering recommendations")
        
        # If no collaborative recommendations, try genre-based user preferences
        if len(collab_recs) == 0:
            print("Trying genre-based collaborative filtering...")
            target_genres = movies_df.loc[idx, 'genres'].split('|')
            genre_collab_recs = get_genre_based_recommendations(target_genres, ratings_df, movies_df, target_movie_id, top_k=10)
            if len(genre_collab_recs) > 0:
                collab_recs = genre_collab_recs
                print(f"Found {len(collab_recs)} genre-based collaborative recommendations")

        # 3. Combine both approaches
        print("Combining recommendations...")
        if len(collab_recs) > 0:
            # Merge content and collaborative recommendations
            content_recs['source'] = 'content'
            collab_recs['source'] = 'collaborative'
            
            # Combine and deduplicate
            all_recs = pd.concat([content_recs, collab_recs], ignore_index=True)
            all_recs = all_recs.drop_duplicates(subset=['movieId'])
            
            # Calculate final scores
            all_recs['final_score'] = 0.0
            for _, row in all_recs.iterrows():
                if row['source'] == 'content':
                    all_recs.loc[row.name, 'final_score'] = row['content_score'] * 0.6
                else:  # collaborative
                    all_recs.loc[row.name, 'final_score'] = row['weighted_rating'] * 0.4
            
            # Sort by final score
            all_recs = all_recs.sort_values('final_score', ascending=False)
            recs = all_recs.head(20)
        else:
            # Fallback to content-based only
            recs = content_recs.head(20).copy()  # Use .copy() to avoid SettingWithCopyWarning
            recs['final_score'] = recs['content_score']

        # Apply mood-based filtering only if a mood is explicitly selected
        if mood and mood.strip() and mood != "":
            mood_genres = get_mood_genres(mood)
            print(f"Applying mood-based filtering for: {mood}")
            print(f"Target genres: {mood_genres}")
            
            # Boost scores for movies matching mood genres
            mood_boosted_count = 0
            for idx, row in recs.iterrows():
                movie_genres = set(row['genres'].split('|'))
                mood_match = any(genre in movie_genres for genre in mood_genres)
                if mood_match:
                    recs.loc[idx, 'final_score'] *= 1.3  # Moderate boost for mood matches
                    mood_boosted_count += 1
                    print(f"Boosted score for '{row['title']}' (mood match: {mood})")
            
            print(f"Mood filtering applied: {mood_boosted_count} movies received mood boost")
        else:
            print("No mood selected - using standard recommendation scoring")
        
        # Apply rating filter only if slider > 0
        if min_rating > 0:
            avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
            recs = recs.merge(avg_ratings, on='movieId', how='left')
            recs['rating'] = recs['rating'].fillna(0)
            recs = recs[recs['rating'] >= min_rating]
            print(f"After rating filter: {len(recs)} movies")

        # Ensure minimum 5 recommendations
        if recs.empty or len(recs) < 5:
            missing = 5 - len(recs) if not recs.empty else 5
            placeholder = pd.DataFrame([{"title":"No Match", "genres":"", "tmdbId":None, "final_score":0.0}] * missing)
            recs = pd.concat([recs, placeholder], ignore_index=True)
        
        # Sort by final score and take top 5
        recs = recs.sort_values('final_score', ascending=False).head(5)
        print(f"Final recommendations: {len(recs)} movies")

        # Fetch posters with error handling
        poster_urls = []
        for tmdb, title in zip(recs['tmdbId'], recs['title']):
            try:
                poster_url = fetch_tmdb_poster(tmdb, title)
                poster_urls.append(poster_url)
                print(f"Poster loaded for '{title}': {poster_url}")
            except Exception as e:
                print(f"Error loading poster for '{title}': {e}")
                # Fallback to local image
                if os.path.exists(NO_COVER_PATH):
                    fallback_url = f"file:///{NO_COVER_PATH.replace(os.sep, '/')}"
                else:
                    fallback_url = "https://via.placeholder.com/200x300?text=No+Image"
                poster_urls.append(fallback_url)
        
        recs['poster_url'] = poster_urls
        
        print(f"Generated {len(recs)} recommendations with posters")

        # Build movie cards
        cards_html = ""
        for _, row in recs.iterrows():
            # Prepare fallback image URL
            fallback_url = f"file:///{NO_COVER_PATH.replace(os.sep, '/')}" if os.path.exists(NO_COVER_PATH) else "https://via.placeholder.com/200x300?text=No+Image"
            
            # Get recommendation source and score
            source = row.get('source', 'content')
            score = row.get('final_score', 0.0)
            source_icon = "🎯" if source == 'content' else "👥"
            source_text = "Content-based" if source == 'content' else "User-based"
            
            # Generate unique colors for each card
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff', '#5f27cd']
            card_color = colors[hash(row['title']) % len(colors)]
            
            card = f"""
            <div class="movie-card" style="--card-color: {card_color};">
                <div class="card-inner">
                    <div class="poster-container">
                <img src="{row['poster_url']}" 
                     alt="{row['title']}" 
                     onerror="this.src='{fallback_url}'; this.style.opacity='0.7';"
                     onload="this.style.opacity='1';"
                     loading="lazy"
                             class="poster-image"/>
                        <div class="rating-badge">
                            ⭐ {row.get('rating', 0):.1f}
                        </div>
                    </div>
                    <div class="card-content">
                        <h3 class="movie-title">{row['title']}</h3>
                        <p class="movie-genres">{row['genres']}</p>
                        <div class="source-info">
                            <span class="source-icon">{source_icon}</span>
                            <span class="source-text">{source_text}</span>
                        </div>
                    </div>
                </div>
            </div>
            """
            cards_html += card

        # Enhanced CSS for movie cards
        full_html = f"""
        <div class="results-section">
        <style>
        .movies-container {{
            display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 30px;
                margin: 0 auto;
                padding: 20px;
                max-width: 1200px;
        }}
        
        .movie-card {{
            position: relative;
            width: 100%;
                max-width: 280px;
            margin: 0 auto;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
                border-radius: 25px;
                overflow: hidden;
                background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.15) 50%, rgba(240,147,251,0.15) 100%);
                backdrop-filter: blur(15px);
                border: 2px solid rgba(255,255,255,0.3);
                box-shadow: 0 12px 40px rgba(102,126,234,0.2);
                animation: shaky3d 2.5s ease-in-out infinite;
                transform-style: preserve-3d;
                perspective: 1000px;
        }}
        
        .movie-card:hover {{
                transform: translateY(-12px) scale(1.03);
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                border-color: var(--card-color);
            }}
            
            .card-inner {{
                position: relative;
                height: 100%;
                display: flex;
                flex-direction: column;
            }}
            
            .poster-container {{
                position: relative;
            width: 100%;
                height: 320px;
                overflow: hidden;
            }}
            
            .poster-image {{
                width: 100%;
                height: 100%;
            object-fit: cover;
                transition: all 0.4s ease;
                filter: brightness(0.9);
        }}
        
            .movie-card:hover .poster-image {{
                transform: scale(1.1);
                filter: brightness(1.1);
        }}
            
            .rating-badge {{
            position: absolute;
                top: 15px;
                right: 15px;
                background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
                color: #333;
                padding: 8px 12px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 14px;
                box-shadow: 0 4px 15px rgba(255,215,0,0.4);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.3);
            }}
            
            .card-content {{
                padding: 20px;
                background: linear-gradient(135deg, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0.6) 100%);
                color: white;
                flex-grow: 1;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }}
            
            .movie-title {{
                font-size: 18px;
                font-weight: 700;
                margin: 0 0 10px 0;
                line-height: 1.3;
                color: #ffffff;
                text-shadow: 0 2px 4px rgba(0,0,0,0.5);
            }}
            
            .movie-genres {{
            font-size: 14px;
                color: rgba(255,255,255,0.8);
                margin: 0 0 15px 0;
                line-height: 1.4;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
                overflow: hidden;
            }}
            
            .source-info {{
                display: flex;
                align-items: center;
                gap: 8px;
            font-size: 12px;
                color: var(--card-color);
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .source-icon {{
                font-size: 16px;
            }}
            
            /* Responsive adjustments */
            @media (max-width: 768px) {{
                .movies-container {{
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    padding: 15px;
                }}
                
                .movie-card {{
                    max-width: 100%;
                }}
                
                .poster-container {{
                    height: 280px;
                }}
                
                .card-content {{
                    padding: 15px;
                }}
                
                .movie-title {{
                    font-size: 16px;
                }}
        }}
        </style>
        <div class="movies-container">{cards_html}</div>
        </div>
        """
        return full_html
        
    except Exception as e:
        print(f"Error in recommend_movies: {e}")
        return f"<h3 style='text-align:center; color:red;'>Error generating recommendations: {str(e)}</h3>"

# ---------------- Gradio UI ----------------
with gr.Blocks(
    css="""
    /* Global Styles */
    html, body {
        overflow-x: hidden;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: 
                    url('movie-recommender/src/data/assets/backgnd.jpg');
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
        min-height: 100vh;
    }

    /* Main Container - Full Width Layout with Background */
    .gradio-container {
        background: url('movie-recommender/src/data/assets/backgnd.jpg') !important;
        max-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        min-height: 100vh !important;
        position: relative !important;
        z-index: 1 !important;
    }

    /* Ensure content is above background */
    .gradio-container > * {
        position: relative !important;
        z-index: 2 !important;
    }

    /* Main Header - Colorful Centered Design with Background Overlay */
    .main-header {
        text-align: center !important;
        margin: 0 !important;
        padding: 50px 20px !important;
        background: linear-gradient(135deg, rgba(102,126,234,0.9) 0%, rgba(118,75,162,0.9) 50%, rgba(240,147,251,0.9) 100%) !important;
        border-radius: 0 0 40px 40px !important;
        box-shadow: 0 15px 40px rgba(102,126,234,0.4) !important;
        backdrop-filter: blur(5px) !important;
        position: relative !important;
        z-index: 3 !important;
    }

    .main-title {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        color: white !important;
        margin: 0 0 20px 0 !important;
        letter-spacing: -1px !important;
        line-height: 1.1 !important;
        text-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
    }

    .subtitle {
        font-size: 1.4rem !important;
        color: rgba(255,255,255,0.9) !important;
        margin: 0 !important;
        font-weight: 400 !important;
        letter-spacing: 0.5px !important;
    }

    /* Search Input Styling */
    .search-input {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        margin: 0 !important;
        padding: 0 !important;
        height: auto !important;
    }

    .search-input input,
    .search-input textarea {
        width: 100% !important;
        height: 48px !important;
        padding: 10px 20px !important;
        border: none !important;
        border-radius: 25px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        line-height: 1.4 !important;
        box-sizing: border-box !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        top: 0px !important;
    }

    .search-input input::placeholder,
    .search-input textarea::placeholder {
        color: rgba(255,255,255,0.85) !important;
        font-size: 16px !important;
        font-weight: 400 !important;
    }

    .search-input input:hover,
    .search-input textarea:hover,
    .search-input input:focus,
    .search-input textarea:focus {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3) !important;
    }

    /* Row Styling - Colorful Distributed Layout with Background */
    .gr-row {
        align-items: stretch !important;
        gap: 25px !important;
        margin: 40px 20px !important;
        padding: 30px !important;
        background: rgba(255,255,255,0.08) !important;
        border-radius: 25px !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
        position: relative !important;
        z-index: 2 !important;
    }

    .gr-row > .gr-column {
        display: flex !important;
        align-items: center !important;
        min-height: 80px !important;
        padding: 15px !important;
        border-radius: 20px !important;
        transition: all 0.3s ease !important;
        overflow: visible !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    .gr-row > .gr-column:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    }

    /* Mood Selector - Minimal Interference with Gradio */
    .mood-selector {
        position: relative !important;
        width: 100% !important;
    }

    .mood-selector label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        margin-bottom: 8px !important;
        display: block !important;
    }

    /* Target Gradio's actual dropdown classes */
    .mood-selector .gr-dropdown,
    .mood-selector .gr-dropdown-toggle {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
        border: none !important;
        border-radius: 25px !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        box-shadow: 0 6px 20px rgba(255,107,107,0.4) !important;
        transition: all 0.3s ease !important;
        min-height: 52px !important;
        width: 100% !important;
    }

    .mood-selector .gr-dropdown:hover,
    .mood-selector .gr-dropdown-toggle:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(255,107,107,0.5) !important;
        background: linear-gradient(135deg, #ff5252 0%, #d32f2f 100%) !important;
    }

    .mood-selector .gr-dropdown:focus,
    .mood-selector .gr-dropdown-toggle:focus {
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(255,107,107,0.3) !important;
    }

    /* Style the dropdown menu */
    .mood-selector .gr-dropdown-menu {
        background: white !important;
        border: 1px solid #ddd !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        padding: 5px 0 !important;
        max-height: 200px !important;
        overflow-y: auto !important;
        z-index: 1000 !important;
    }

    /* Style dropdown items */
    .mood-selector .gr-dropdown-item {
        padding: 12px 20px !important;
        color: #333 !important;
        cursor: pointer !important;
        transition: background-color 0.2s ease !important;
        font-weight: 600 !important;
    }

    .mood-selector .gr-dropdown-item:hover {
        background-color: #f8f9fa !important;
    }

    /* Fallback for other possible Gradio classes */
    .mood-selector .dropdown-toggle {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
        border: none !important;
        border-radius: 25px !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        box-shadow: 0 6px 20px rgba(255,107,107,0.4) !important;
        transition: all 0.3s ease !important;
        min-height: 52px !important;
        width: 100% !important;
    }

    .mood-selector .dropdown-menu {
        background: white !important;
        border: 1px solid #ddd !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        padding: 5px 0 !important;
        max-height: 200px !important;
        overflow-y: auto !important;
        z-index: 1000 !important;
    }

    .mood-selector .dropdown-item {
        padding: 12px 20px !important;
        color: #333 !important;
        cursor: pointer !important;
        transition: background-color 0.2s ease !important;
        font-weight: 600 !important;
    }

    .mood-selector .dropdown-item:hover {
        background-color: #f8f9fa !important;
    }

    /* Ensure dropdown container allows proper interaction */
    .mood-selector .wrap {
        position: relative !important;
        width: 100% !important;
        z-index: 1 !important;
    }

    /* Make sure the dropdown is clickable */
    .mood-selector * {
        pointer-events: auto !important;
    }

    /* Ensure proper z-index stacking */
    .mood-selector .gr-dropdown-menu,
    .mood-selector .dropdown-menu {
        position: absolute !important;
        top: 100% !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 1000 !important;
    }

    /* Rating Slider Styling - Colorful */
    .rating-slider {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 20px !important;
        box-shadow: 0 6px 20px rgba(78,205,196,0.4) !important;
        backdrop-filter: blur(10px) !important;
        min-height: 52px !important;
        width: 100% !important;
    }

    .rating-slider:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(78,205,196,0.5) !important;
    }

    /* Button Styling - Colorful and Distributed */
    .main-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 30px !important;
        color: white !important;
        padding: 18px 35px !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        box-shadow: 0 10px 30px rgba(102,126,234,0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        min-height: 60px !important;
        width: 100% !important;
    }

    .main-btn:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 15px 40px rgba(102,126,234,0.6) !important;
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
    }

    .surprise-btn {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        border: none !important;
        border-radius: 30px !important;
        color: white !important;
        padding: 18px 35px !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        box-shadow: 0 10px 30px rgba(240,147,251,0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        min-height: 60px !important;
        width: 100% !important;
    }

    .surprise-btn:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 15px 40px rgba(240,147,251,0.6) !important;
        background: linear-gradient(135deg, #e884f0 0%, #f4455a 100%) !important;
    }

    /* Results Section - Colorful */
    .results-section {
        margin: 40px 20px !important;
        padding: 40px !important;
        background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%) !important;
        border-radius: 30px !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        box-shadow: 0 15px 50px rgba(0,0,0,0.3) !important;
    }

    /* Movie Cards Container - Full Width Colorful */
    .movies-container {
        max-width: 100% !important;
        margin: 0 20px !important;
        display: grid !important;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)) !important;
        gap: 30px !important;
        justify-items: center !important;
        align-items: start !important;
        padding: 30px !important;
        background: rgba(255,255,255,0.05) !important;
        border-radius: 25px !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important;
    }

    /* Responsive Design */
    @media (max-width: 1200px) { 
        .movies-container { grid-template-columns: repeat(4, 1fr) !important; } 
        .main-title { font-size: 3.2rem !important; }
    }
    @media (max-width: 900px) { 
        .movies-container { grid-template-columns: repeat(3, 1fr) !important; } 
        .main-title { font-size: 2.8rem !important; }
        .gr-row { gap: 12px !important; }
    }
    @media (max-width: 680px) { 
        .movies-container { grid-template-columns: repeat(2, 1fr) !important; } 
        .main-title { font-size: 2.4rem !important; }
        .gr-row { 
            flex-direction: column !important; 
            gap: 15px !important; 
            margin: 15px 0 !important;
        }
        .gr-row > .gr-column {
            min-height: 60px !important;
        }
        .search-input input,
        .search-input textarea {
            height: 55px !important;
            font-size: 15px !important;
        }
    }
    @media (max-width: 460px) { 
        .movies-container { grid-template-columns: 1fr !important; } 
        .main-title { font-size: 2rem !important; }
        .subtitle { font-size: 1rem !important; }
        .gr-row { 
            padding: 0 5px !important;
            gap: 12px !important;
        }
    }

    /* Remove Gradio default styling */
    .gr-column,
    .gr-column *:where(.gr-box, .gr-block, .gr-group, .gr-panel, .gr-form, .container, .wrap) {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Label Styling */
    .gr-label {
        color: rgba(255,255,255,0.9) !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        margin-bottom: 10px !important;
    }
""",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    )
) as demo:
    # Main Title
    gr.HTML("""
        <div class="main-header">
            <h1 class="main-title">🎬 Movie Recommendation Engine</h1>
            <p class="subtitle">Discover your next favorite movie with AI-powered recommendations</p>
        </div>
    """)
    
    # Search Section - Colorful Distributed Layout
    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            movie_input = gr.Textbox(
                placeholder="🔍 Search for movies...",
                show_label=False,
                container=False,
                elem_classes="search-input",
                interactive=True
            )
        with gr.Column(scale=3):
            mood_selector = gr.Dropdown(
                choices=["", "Happy 😊", "Sad 😢", "Adventurous 🚀", "Romantic 💕", "Thriller 🎭", "Chill 😌"],
                label="Choose Your Mood",
                value="",
                elem_classes="mood-selector",
                interactive=True,
                allow_custom_value=False,
                multiselect=False,
                type="value",
                container=True,
                show_label=True,
                info="Select your mood to get personalized recommendations"
            )
        with gr.Column(scale=3):
            rating_slider = gr.Slider(
                label="Min Rating", 
                minimum=0, 
                maximum=5, 
                step=0.1, 
                value=0,
                elem_classes="rating-slider"
            )
    
    # Action buttons - Full width distributed
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            btn = gr.Button("✨ Get Recommendations ✨", size="lg", elem_classes="main-btn")
        with gr.Column(scale=1):
         surprise_btn = gr.Button("🎲 Surprise Me!", size="lg", elem_classes="surprise-btn")
    
    
    output = gr.HTML()

    # Surprise me function
    def surprise_me():
        """Pick a random movie and show recommendations"""
        import random
        random_movie = random.choice(movies_df['title'].tolist())
        return random_movie
    
    # Trigger recommendations only on submit and button click
    movie_input.submit(recommend_movies, inputs=[movie_input, rating_slider, mood_selector], outputs=output)
    btn.click(recommend_movies, inputs=[movie_input, rating_slider, mood_selector], outputs=output)
    surprise_btn.click(surprise_me, outputs=movie_input)

# ---------------- Launch ----------------
if __name__ == "__main__":

    demo.launch(debug=True, share=True)