import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

def hybrid_recommend(movie_name, movies_df, ratings_df, top_n=5):
    """
    Recommends movies similar to the given movie using a hybrid approach:
    genre similarity (cosine similarity) + normalized average ratings.

    Args:
        movie_name (str): Name of the input movie
        movies_df (pd.DataFrame): DataFrame containing movies info (columns: movieId, title, genres)
        ratings_df (pd.DataFrame): DataFrame containing ratings info (columns: movieId, rating)
        top_n (int): Number of recommendations to return

    Returns:
        pd.DataFrame: Top-N recommended movies with movieId, title, genres, and score
    """
    # --- Fuzzy match movie name ---
    movie_row = movies_df[movies_df['title'].str.lower() == movie_name.lower()]
    if movie_row.empty:
        closest = get_close_matches(movie_name, movies_df['title'].tolist(), n=1, cutoff=0.5)
        if not closest:
            return pd.DataFrame(columns=['movieId', 'title', 'genres', 'score'])
        movie_row = movies_df[movies_df['title'] == closest[0]]

    movie_index = movie_row.index[0]

    # --- One-hot encode genres if not already done ---
    genre_cols = [col for col in movies_df.columns if col not in ['movieId', 'title', 'genres']]
    if not genre_cols:
        all_genres = set(g for sub in movies_df['genres'].str.split('|') for g in sub)
        for g in all_genres:
            movies_df[g] = movies_df['genres'].apply(lambda x: int(g in x.split('|')))
        genre_cols = list(all_genres)

    # --- Compute genre similarity ---
    genre_vector = movies_df.loc[movie_index, genre_cols].values.reshape(1, -1)
    genre_matrix = movies_df[genre_cols].values
    genre_sim = cosine_similarity(genre_vector, genre_matrix)[0]

    # --- Compute normalized average ratings ---
    avg_ratings = ratings_df.groupby('movieId')['rating'].mean()
    ratings_norm = (avg_ratings - avg_ratings.min()) / (avg_ratings.max() - avg_ratings.min())
    ratings_norm = ratings_norm.reindex(movies_df['movieId']).fillna(0).values

    # --- Combine scores ---
    final_score = 0.7 * genre_sim + 0.3 * ratings_norm
    final_score[movie_index] = -1  # exclude the input movie

    top_indices = np.argsort(final_score)[-top_n:][::-1]
    recommendations = movies_df.iloc[top_indices].copy()
    recommendations['score'] = final_score[top_indices]

    return recommendations[['movieId', 'title', 'genres', 'score']]
