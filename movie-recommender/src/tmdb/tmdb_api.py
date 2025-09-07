import os
import json
import requests
import pandas as pd
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
SRC_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(SRC_DIR, "data")
MOVIES_CSV = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")
TAGS_CSV = os.path.join(DATA_DIR, "tags.csv")
LINKS_CSV = os.path.join(DATA_DIR, "links.csv")
CACHE_FILE = os.path.join(DATA_DIR, "tmdb_cache.json")
NO_COVER_PATH = os.path.join(SRC_DIR, "assets", "cover_not_found.jpg")

# ---------------- Load Data ----------------
movies_df = pd.read_csv(MOVIES_CSV)
ratings_df = pd.read_csv(RATINGS_CSV)
tags_df = pd.read_csv(TAGS_CSV)
links_df = pd.read_csv(LINKS_CSV)

# Merge tags
tags_grouped = tags_df.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
movies_df = movies_df.merge(tags_grouped, on="movieId", how="left")
movies_df['tag'] = movies_df['tag'].fillna("")

# Merge TMDb links
movies_df = movies_df.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')

# ---------------- Create content for similarity: genres + tags ----------------
movies_df['genres'] = movies_df['genres'].fillna("")
movies_df['tag'] = movies_df['tag'].fillna("")
movies_df['content'] = movies_df['genres'] + " " + movies_df['tag']

# ---------------- Content-based similarity ----------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies_df.index, index=movies_df['title'].str.lower()).drop_duplicates()

# ---------------- Poster cache ----------------
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        poster_cache = json.load(f)
else:
    poster_cache = {}

def fetch_tmdb_poster(tmdb_id, title):
    if tmdb_id:
        tmdb_id = str(int(tmdb_id))
        if tmdb_id in poster_cache:
            return poster_cache[tmdb_id]
        try:
            url = f"{TMDB_BASE_URL}/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
            r = requests.get(url, timeout=5).json()
            poster_path = r.get("poster_path")
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else NO_COVER_PATH
        except:
            poster_url = NO_COVER_PATH
        poster_cache[tmdb_id] = poster_url
    else:
        if title in poster_cache:
            return poster_cache[title]
        try:
            response = requests.get(
                f"{TMDB_BASE_URL}/search/movie",
                params={"api_key": TMDB_API_KEY, "query": title},
                timeout=5
            ).json()
            results = response.get("results")
            if results:
                poster_path = results[0].get("poster_path")
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else NO_COVER_PATH
            else:
                poster_url = NO_COVER_PATH
        except:
            poster_url = NO_COVER_PATH
        poster_cache[title] = poster_url

    with open(CACHE_FILE, "w") as f:
        json.dump(poster_cache, f)
    return poster_cache.get(tmdb_id) if tmdb_id else poster_cache.get(title)

# ---------------- Recommendation Function ----------------
def recommend_movies(movie_name, min_rating):
    all_titles = movies_df['title'].str.lower().tolist()
    matches = get_close_matches(movie_name.lower(), all_titles, n=1, cutoff=0.6)
    if not matches:
        return "<h3 style='text-align:center; color:white;'>No movie found with that name!</h3>"
    
    idx = indices[matches[0]]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:20]
    movie_indices = [i[0] for i in sim_scores]
    recs = movies_df.iloc[movie_indices].copy()

    if min_rating > 0:
        avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
        recs = recs.merge(avg_ratings, on='movieId', how='left')
        recs['rating'] = recs['rating'].fillna(0)
        recs = recs[recs['rating'] >= min_rating]

    if recs.empty or len(recs) < 5:
        missing = 5 - len(recs) if not recs.empty else 5
        placeholder = pd.DataFrame([{"title": "No Match", "genres": "", "tmdbId": None}] * missing)
        recs = pd.concat([recs, placeholder], ignore_index=True)

    recs = recs.head(5)
    recs['poster_url'] = [fetch_tmdb_poster(tmdb, title) for tmdb, title in zip(recs['tmdbId'], recs['title'])]

    # Build HTML for horizontal cards
    cards_html = ""
    for _, row in recs.iterrows():
        card = f"""
        <div class="movie-card">
            <img src="{row['poster_url']}" alt="{row['title']}"/>
            <div class="overlay">
                <h3>{row['title']}</h3>
                <p>{row['genres']}</p>
                <p>⭐ {row.get('rating', 0):.2f}</p>
            </div>
        </div>
        """
        cards_html += card

    full_html = f"""
    <style>
        body {{
            background: #0b0c1c;
            font-family: Arial, sans-serif;
        }}
        .title-card {{
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            color: #00fff7;
            text-shadow: 0 0 8px #00fff7, 0 0 20px #00fff7, 0 0 40px #00fff7;
            margin-top: 20px;
            margin-bottom: 20px;
        }}
        .movies-container {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: nowrap;
            overflow-x: auto;
            padding: 10px;
        }}
        .movie-card {{
            position: relative;
            width: 200px;
            transition: transform 0.3s ease;
        }}
        .movie-card:hover {{
            transform: rotateY(5deg) rotateX(5deg) translateY(-5px);
        }}
        .movie-card img {{
            width: 100%;
            height: 300px;
            border-radius: 12px;
            object-fit: cover;
        }}
        .overlay {{
            position: absolute;
            bottom: 0;
            width: 100%;
            padding: 8px;
            background: rgba(0,0,0,0.6);
            color: #fff;
            text-align: center;
            border-radius: 0 0 12px 12px;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        .movie-card:hover .overlay {{
            opacity: 1;
        }}
        .overlay h3 {{
            font-size: 16px;
            margin: 2px 0;
        }}
        .overlay p {{
            font-size: 14px;
            margin: 0;
        }}
    </style>
    <div class="movies-container">{cards_html}</div>
    """
    return full_html
# ---------------- Gradio UI ----------------
with gr.Blocks(css="""
    .gradio-container {max-width: 1300px !important; margin: auto;}
    .search-bar {width: 80% !important;}
    .rating-slider {width: 200px !important;}
    .movies-container {justify-content: center !important; max-width: 100%;}
""") as demo:

    # Column to ensure vertical stacking: title → search → rating → button → output
    with gr.Column():
        # Neon title on top
        gr.HTML("<div class='title-card'>🎬 Movie Recommendation Engine</div>")

        # Search bar and rating filter
        with gr.Row():
            movie_input = gr.Textbox(label="", placeholder="Type a movie title...", elem_classes="search-bar", interactive=True)
            rating_slider = gr.Slider(label="Minimum Rating", minimum=0, maximum=5, step=0.1, value=0, elem_classes="rating-slider")

        # Get Recommendations button
        btn = gr.Button("✨ Get Recommendations ✨")
        output = gr.HTML()

        # Trigger recommendations
        movie_input.submit(recommend_movies, inputs=[movie_input, rating_slider], outputs=output)
        btn.click(recommend_movies, inputs=[movie_input, rating_slider], outputs=output)

# ---------------- Launch ----------------
if __name__ == "__main__":
    demo.launch(debug=True, share=True)