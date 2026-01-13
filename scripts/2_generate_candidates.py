import pandas as pd
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors
import os
import argparse

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
INPUT_FILE = '/project/home/p200253/hyperbolic/projections/files/combined_secondary_tertiary_skills.csv'
OUTPUT_DIR = '/project/home/p200253/hyperbolic/projections/files/candidates'
DEFAULT_K = 15  # Number of neighbors to look at (local neighborhood)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def parse_vector(vec_str):
    """
    Parses a string representation of a vector into a numpy array.
    Handles JSON strings or plain string representations.
    """
    try:
        # If it's a list inside a string, json.loads usually works
        return np.array(json.loads(vec_str), dtype=np.float32)
    except (TypeError, json.JSONDecodeError):
        # Fallback if data is already a list or other format
        if isinstance(vec_str, (list, np.ndarray)):
            return np.array(vec_str, dtype=np.float32)
        return np.zeros(1) # Fail safe (should not happen with clean data)

def get_embeddings(df, mode='name'):
    """
    Extracts and stacks embeddings based on the selected mode.
    Modes: 'name', 'description', 'mean' (average of both)
    """
    print(f"--> Processing embeddings. Mode: {mode}")
    
    # Parse vectors if they are strings
    # We assume the CSV reading might have loaded them as strings
    if isinstance(df['name_vector'].iloc[0], str):
        print("    Parsing 'name_vector' from strings...")
        name_vecs = np.stack(df['name_vector'].apply(parse_vector).values)
    else:
        name_vecs = np.stack(df['name_vector'].values)
        
    if mode == 'name':
        return name_vecs

    if isinstance(df['description_vector'].iloc[0], str):
        print("    Parsing 'description_vector' from strings...")
        desc_vecs = np.stack(df['description_vector'].apply(parse_vector).values)
    else:
        desc_vecs = np.stack(df['description_vector'].values)

    if mode == 'description':
        return desc_vecs
    
    if mode == 'mean':
        print("    Averaging name and description vectors...")
        return (name_vecs + desc_vecs) / 2.0
    
    raise ValueError("Invalid embedding mode. Choose 'name', 'description', or 'mean'.")

def is_valid_dependency(source_level, target_level):
    """
    Hard Constraint Logic:
    A Secondary skill CANNOT depend on a Tertiary skill.
    
    Source -> Target (Source is Prerequisite of Target)
    
    Allowed:
    - Secondary -> Secondary
    - Secondary -> Tertiary
    - Tertiary -> Tertiary
    
    Forbidden:
    - Tertiary -> Secondary
    """
    if source_level == 'tertiary' and target_level == 'secondary':
        return False
    return True

# ==========================================
# MAIN EXECUTION
# ==========================================

def main(vector_mode='name', k_neighbors=DEFAULT_K):
    # 1. Load Data
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} skills.")

    # 2. Prepare Embeddings
    embeddings = get_embeddings(df, mode=vector_mode)
    print(f"Embedding shape: {embeddings.shape}")

    # 3. Build k-NN Graph (The 'Similarity Structure')
    print(f"Building k-NN graph with k={k_neighbors}...")
    knn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine', n_jobs=-1)
    knn.fit(embeddings)
    
    # distances: cosine distance (lower is closer), indices: row indices in df
    distances, indices = knn.kneighbors(embeddings)

    # 4. Generate Candidates & Apply Constraints
    print("Generating candidate pairs and applying hard constraints...")
    
    candidates = []
    
    # Optimizing lookup for speed
    levels = df['education_level'].values
    uids = df['skill_uid'].values
    
    count_total_pairs = 0
    count_dropped_constraints = 0

    for i in range(len(df)):
        source_idx = i
        source_uid = uids[source_idx]
        source_level = levels[source_idx]
        
        # Iterate through neighbors of i
        # Note: j starts at 1 usually to skip self-loop (distance 0), 
        # but sometimes duplicates exist, so we check strict identity if needed.
        for rank, target_idx in enumerate(indices[i]):
            if source_idx == target_idx:
                continue # Skip self
            
            target_uid = uids[target_idx]
            target_level = levels[target_idx]
            dist = distances[i][rank]
            similarity = 1 - dist # Convert cosine distance to similarity approximation
            
            # Constraint Check: Can Source be a prerequisite for Target?
            if is_valid_dependency(source_level, target_level):
                candidates.append({
                    'source_uid': source_uid,
                    'target_uid': target_uid,
                    'source_level': source_level,
                    'target_level': target_level,
                    'similarity': similarity,
                    'rank': rank
                })
            else:
                count_dropped_constraints += 1
            
            count_total_pairs += 1

    # 5. Save Results
    candidates_df = pd.DataFrame(candidates)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filename = f"{OUTPUT_DIR}/candidates_k{k_neighbors}_mode_{vector_mode}.parquet"
    
    print("-" * 30)
    print(f"Total neighbor pairs examined: {count_total_pairs}")
    print(f"Pairs dropped due to Education Level constraint (Ter->Sec): {count_dropped_constraints}")
    print(f"Valid candidates retained: {len(candidates_df)}")
    print("-" * 30)
    
    # Save as Parquet for efficiency (maintains types better than CSV)
    candidates_df.to_parquet(output_filename, index=False)
    print(f"Saved candidate pairs to: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prerequisite candidates from embeddings.")
    parser.add_argument('--mode', type=str, default='mean', choices=['name', 'description', 'mean'], 
                        help="Which vector to use for similarity.")
    parser.add_argument('--k', type=int, default=15, 
                        help="Number of nearest neighbors to retrieve.")
    
    args = parser.parse_args()
    
    main(vector_mode=args.mode, k_neighbors=args.k)