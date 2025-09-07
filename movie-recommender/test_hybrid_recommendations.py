#!/usr/bin/env python3
"""
Test script to verify hybrid recommendation functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ui.gradio_app import recommend_movies

def test_hybrid_recommendations():
    """Test the hybrid recommendation system"""
    print("Testing hybrid recommendation system...")
    
    # Test cases
    test_movies = [
        "The Dark Knight",
        "Inception", 
        "Pulp Fiction"
    ]
    
    for movie in test_movies:
        print(f"\n{'='*50}")
        print(f"Testing recommendations for: {movie}")
        print(f"{'='*50}")
        
        try:
            # Get recommendations
            result = recommend_movies(movie, min_rating=0)
            print(f"✓ Successfully generated recommendations")
            
            # Check if result contains HTML
            if isinstance(result, str) and '<div class="movies-container">' in result:
                print("✓ Result contains valid HTML output")
            else:
                print("⚠ Result format may be unexpected")
                
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_hybrid_recommendations()
