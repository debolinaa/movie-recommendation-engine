#!/usr/bin/env python3
"""
Test script to verify poster loading functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ui.gradio_app import fetch_tmdb_poster, clean_cache

def test_poster_loading():
    """Test poster loading for various movies"""
    print("Testing poster loading functionality...")
    
    # Test cases
    test_cases = [
        (None, "The Dark Knight"),
        (None, "Inception"),
        (None, "Pulp Fiction"),
        (None, "The Shawshank Redemption"),
        (None, "Fight Club")
    ]
    
    for tmdb_id, title in test_cases:
        try:
            print(f"\nTesting: {title}")
            poster_url = fetch_tmdb_poster(tmdb_id, title)
            print(f"  Poster URL: {poster_url}")
            
            # Check if URL is valid
            if poster_url.startswith("https://image.tmdb.org"):
                print("  ✓ Valid TMDB poster URL")
            elif poster_url.startswith("file://"):
                print("  ✓ Using local fallback image")
            elif poster_url.startswith("https://via.placeholder.com"):
                print("  ✓ Using placeholder image")
            else:
                print(f"  ⚠ Unexpected URL format: {poster_url}")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\nTesting cache cleaning...")
    try:
        clean_cache()
        print("  ✓ Cache cleaned successfully")
    except Exception as e:
        print(f"  ✗ Cache cleaning error: {e}")

if __name__ == "__main__":
    test_poster_loading()
