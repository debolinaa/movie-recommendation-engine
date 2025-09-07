import os
import pandas as pd
import re

# ----------------- Paths -----------------
DATA_DIR = os.path.join("src", "data")  # or "small_data" if you moved files
EMBEDDING_UTILS_FILE = os.path.join("src", "semantic_search", "embedding_utils.py")

# ----------------- List CSVs -----------------
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

if not csv_files:
    print("⚠️ No CSV files found in", DATA_DIR)
else:
    print(f"✅ Found {len(csv_files)} CSV files:")
    for file in csv_files:
        file_path = os.path.join(DATA_DIR, file)
        try:
            df = pd.read_csv(file_path)
            print(f"\n📂 {file}:")
            print(df.head())
            print(f"Shape: {df.shape}")
        except Exception as e:
            print(f"\n❌ Could not read {file}: {e}")

# ----------------- Auto-update embedding_utils.py -----------------
if os.path.exists(EMBEDDING_UTILS_FILE):
    with open(EMBEDDING_UTILS_FILE, "r") as f:
        code = f.read()

    # Update MOVIES_CSV path
    movies_csv_path = os.path.join(DATA_DIR, "movies.csv").replace("\\", "/")
    code_new = re.sub(r'MOVIES_CSV\s*=.*', f'MOVIES_CSV = "{movies_csv_path}"', code)

    # Update EMBEDDINGS_CSV path
    embeddings_csv_path = os.path.join(DATA_DIR, "embeddings.csv").replace("\\", "/")
    code_new = re.sub(r'EMBEDDINGS_CSV\s*=.*', f'EMBEDDINGS_CSV = "{embeddings_csv_path}"', code_new)

    with open(EMBEDDING_UTILS_FILE, "w") as f:
        f.write(code_new)

    print(f"\n🔧 Updated paths in {EMBEDDING_UTILS_FILE}")
else:
    print(f"\n⚠️ embedding_utils.py not found at {EMBEDDING_UTILS_FILE}")
