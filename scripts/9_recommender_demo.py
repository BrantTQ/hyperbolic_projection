import pandas as pd
import numpy as np
import torch
import os

# ==========================================
# CONFIGURATION
# ==========================================
EMBEDDINGS_FILE = '/project/home/p200253/hyperbolic/projections/files/embeddings/hyperbolic_embeddings_2d.csv'

# We will test the recommender on these specific skills
# You can change these to any skill names that exist in your dataset
TEST_QUERIES = [
    "mathematics",              # Very General (Secondary)
    "computer programming",     # Bridge Skill
    "machine learning",         # Specialized (Tertiary)
    "cyber security"            # Specialized (Tertiary)
]

# ==========================================
# MATH UTILS (Hyperbolic Distance)
# ==========================================
def poincare_dist_numpy(u, v):
    """
    Numpy version of the Poincare distance formula for fast inference.
    u, v: arrays of shape (N, 2)
    """
    # Euclidean norms squared
    sq_u = np.sum(u**2, axis=1)
    sq_v = np.sum(v**2, axis=1)
    
    # Euclidean distance squared
    sq_dist = np.sum((u - v)**2, axis=1)
    
    # Mobius addition / Poincare metric
    # boundary protection (epsilon)
    epsilon = 1e-7
    val = 1 + 2 * sq_dist / ((1 - sq_u) * (1 - sq_v) + epsilon)
    
    # arccosh
    return np.arccosh(np.clip(val, 1.0 + epsilon, None))

# ==========================================
# RECOMMENDER ENGINE
# ==========================================
def get_recommendations(target_name, df, top_k=5):
    # 1. Find the target skill vector
    # We use string matching (case insensitive)
    row = df[df['name'].str.lower() == target_name.lower()]
    
    if len(row) == 0:
        print(f"Skill '{target_name}' not found in dataset.")
        return

    target_vec = row[['h_x', 'h_y']].values
    target_norm = row['h_norm'].values[0]
    target_level = row['education_level'].values[0]
    
    print(f"\n{'='*60}")
    print(f"QUERY: {row['name'].values[0]} ({target_level.capitalize()})")
    print(f"Complexity Score (Radius): {target_norm:.4f}")
    print(f"{'='*60}")
    
    # 2. Calculate Distance to ALL other skills
    all_vecs = df[['h_x', 'h_y']].values
    
    # Identify target index to exclude self
    dists = poincare_dist_numpy(all_vecs, target_vec)
    
    df['dist'] = dists
    
    # 3. Filter into Categories based on Radius (Hierarchy)
    
    # A) Prerequisites (Parents): 
    # Must be significantly closer to center (smaller radius) AND close in space
    # Logic: radius < target_radius - margin
    parents = df[
        (df['h_norm'] < target_norm - 0.05) & 
        (df['dist'] < 2.0) # Within reasonable semantic distance
    ].sort_values('dist').head(top_k)
    
    # B) Next Steps (Children):
    # Must be significantly further out (larger radius) AND close in space
    children = df[
        (df['h_norm'] > target_norm + 0.05) & 
        (df['dist'] < 2.5) # Allow slightly wider search for future steps
    ].sort_values('dist').head(top_k)
    
    # C) Related Topics (Siblings):
    # Roughly same radius (+/- 0.05)
    siblings = df[
        (np.abs(df['h_norm'] - target_norm) <= 0.05) & 
        (df['name'] != row['name'].values[0])
    ].sort_values('dist').head(top_k)
    
    # 4. Print Results
    print(f"--- 🔙 PREREQUISITES (Foundational) ---")
    if len(parents) > 0:
        for _, r in parents.iterrows():
            print(f"  • {r['name']} ({r['education_level'][0].upper()}) [Dist: {r['dist']:.2f}]")
    else:
        print("  (None found - This might be a root skill)")

    print(f"\n--- ↔️ RELATED SKILLS (Siblings) ---")
    if len(siblings) > 0:
        for _, r in siblings.iterrows():
            print(f"  • {r['name']} ({r['education_level'][0].upper()}) [Dist: {r['dist']:.2f}]")
    else:
        print("  (None found)")

    print(f"\n--- 🔜 NEXT STEPS (Advanced) ---")
    if len(children) > 0:
        for _, r in children.iterrows():
            print(f"  • {r['name']} ({r['education_level'][0].upper()}) [Dist: {r['dist']:.2f}]")
    else:
        print("  (None found - This might be a leaf skill)")

# ==========================================
# MAIN
# ==========================================
def main():
    print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    df = pd.read_csv(EMBEDDINGS_FILE)
    
    print(f"Loaded {len(df)} skills.")
    
    for query in TEST_QUERIES:
        get_recommendations(query, df)

if __name__ == "__main__":
    main()