# Movie Recommender System

This project is a movie recommender system built using the MovieLens Small dataset. It leverages semantic search and embeddings to provide personalized movie recommendations. The application features a user-friendly interface built with Streamlit and integrates with the TMDB API for fetching movie details.

## Project Structure

```
movie-recommender
├── src
│   ├── main.py                # Entry point for the application
│   ├── data
│   │   └── movielens_loader.py # Functions to load and preprocess MovieLens datasets
│   ├── embeddings
│   │   └── embedding_utils.py   # Functions to generate and manage movie embeddings
│   ├── semantic_search
│   │   └── search.py            # Implements semantic search functionality
│   ├── tmdb
│   │   └── tmdb_api.py          # Functions to interact with the TMDB API
│   ├── ui
│   │   └── gradio_app.py     # Gradio UI for user interaction
│   └── utils
│       └── helpers.py           # Various helper functions
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
└── .env                         # Environment variables (e.g., TMDB API key)
```

## Features

- **Movie Recommendations**: Get personalized movie suggestions based on user input.
- **Semantic Search**: Search for movies using natural language queries.
- **TMDB Integration**: Fetch movie posters and additional information from the TMDB API.
- **User-Friendly UI**: A responsive and visually appealing interface built with gradio.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd movie-recommender
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your TMDB API key:
   - Create a `.env` file in the root directory and add your TMDB API key:
     ```
     TMDB_API_KEY=your_api_key_here
     ```

5. Run the application:
   ```
   python src/main.py
   ```
# 🎬 Movie Recommendation Engine  

A content-based movie recommender system with a modern Gradio-powered user interface.  

---

features 

![WhatsApp Image 2025-09-08 at 15 06 49_8b623e6f](https://github.com/user-attachments/assets/da5b961b-ccf3-424e-b065-a9cb40cbb08f)
1. searchbar
2. recommend button
3. choose your mood feature, for personalized recommendations
![WhatsApp Image 2025-09-08 at 15 07 33_b9d2c780](https://github.com/user-attachments/assets/4d1c51e0-93c0-4a21-b875-c6a6e2c91f1c)
![WhatsApp Image 2025-09-08 at 15 07 58_4309a5be](https://github.com/user-attachments/assets/b9f00723-c95a-4c35-8b31-4fc479b98078)

4. surprise me to get randomized reccomendation for hidden gems
![WhatsApp Image 2025-09-08 at 15 09 02_3316576a](https://github.com/user-attachments/assets/db11137a-cb1a-4d62-9c31-fd02481ae9ba)

5. minimum rating feature , filtering based on rating selected
![WhatsApp Image 2025-09-08 at 15 09 32_69276aae](https://github.com/user-attachments/assets/b2902a6c-c37c-46b1-aaab-3025f5f578fc)

sample searches

![WhatsApp Image 2025-09-08 at 15 11 12_1277f9e6](https://github.com/user-attachments/assets/3760a5ed-1393-4836-98e1-548ae6c17f6e)
![WhatsApp Image 2025-09-08 at 15 14 10_345390e5](https://github.com/user-attachments/assets/a5ef13e3-d48e-4266-90d8-7c13a1311788)
![WhatsApp Image 2025-09-08 at 15 14 39_53c11511](https://github.com/user-attachments/assets/fd842aa6-d98a-4c71-8cb2-e1453b03303b)
![WhatsApp Image 2025-09-08 at 15 16 07_229e0c1c](https://github.com/user-attachments/assets/82d2ddce-bd32-454e-b676-5e7a73e0108d)



## 1. 📊 Data Exploration Findings  

- **Dataset Source**: [ TMDB dataset / CSV file].    
- **Key Features Used**:  
  - Title  
  - Genres  
  - Overview/Description  
  - Ratings  

### 🔍 EDA Highlights  
- Distribution of ratings across movies.  
- Most frequent genres (e.g., Drama, Comedy, Action).  
- Popular directors/actors.  
- WordCloud of most common words in movie descriptions.  

*(Add plots/screenshots here later — e.g., genre distribution, ratings histogram)*  

---

## 2. 🤖 Recommendation Logic & Implementation  

### ✅ Approach  
- **Content-Based Filtering**  
  - Used **TF-IDF Vectorizer** on movie overviews.  
  - Computed **cosine similarity** between movie vectors.  

- **Surprise Me! Feature**  
  - Random movie suggestion for exploration.  

### 🔑 Core Implementation  

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


## Acknowledgments

This project utilizes the MovieLens dataset and the TMDB API for enhanced movie data. Special thanks to the developers of Streamligradio for providing an excellent framework for building interactive applications.
