import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__))

def load_data(movies_file="movies.csv", ratings_file="ratings.csv"):
    movies_path = os.path.join(DATA_DIR, movies_file)
    ratings_path = os.path.join(DATA_DIR, ratings_file)
    
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    
    return movies, ratings
