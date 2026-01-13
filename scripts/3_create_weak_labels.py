import pandas as pd
import numpy as np
import os
import argparse
from sklearn.neighbors import NearestNeighbors

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_CSV = '/project/home/p200253/hyperbolic/projections/files/combined_secondary_tertiary_skills.csv'
INPUT_CANDIDATES = '/project/home/p200253/hyperbolic/projections/files/candidates/candidates_k15_mode_mean.parquet'
OUTPUT_DIR = '/project/home/p200253/hyperbolic/projections/files/training_data'

# Suppress OpenBLAS warning
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_generality(df, candidates_df):
    """
    Approximates 'Generality' using In-Degree within the candidate graph.
    Hypothesis: Foundational skills appear as neighbors (targets) for many other skills 
    OR act as hubs.
    
    Here, we assume 'Broad/General' skills are those that many others relate to.
    """
    print("Calculating generality heuristics...")
    # Count how many times a skill appears as a neighbor in the raw similarity graph
    # (Using the target_uid column from candidates is a good proxy for 'centrality')
    centrality = candidates_df['target_uid'].value_counts().to_dict()
    
    # Map to dataframe
    df['centrality_score'] = df['skill_uid'].map(centrality).fillna(0)
    return df

def parse_vector(vec_str):
    import json
    try:
        return np.array(json.loads(vec_str), dtype=np.float32)
    except:
        if isinstance(vec_str, (list, np.ndarray)):
            return np.array(vec_str, dtype=np.float32)
        return np.zeros(1)

def get_embeddings(df):
    # Quick parser assuming mean mode (consistent with your previous run)
    print("Parsing vectors for Hard Negative regeneration...")
    if isinstance(df['name_vector'].iloc[0], str):
        n = np.stack(df['name_vector'].apply(parse_vector).values)
        d = np.stack(df['description_vector'].apply(parse_vector).values)
        return (n + d) / 2.0
    else:
        return (np.stack(df['name_vector'].values) + np.stack(df['description_vector'].values)) / 2.0

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading Metadata: {INPUT_CSV}")
    skills_df = pd.read_csv(INPUT_CSV)
    
    print(f"Loading Candidates: {INPUT_CANDIDATES}")
    positives_df = pd.read_parquet(INPUT_CANDIDATES)
    
    # 2. Refine Positives (Add Directional Heuristics)
    # Strategy: For Cross-Level (Sec->Ter), direction is fixed.
    # For Same-Level, use Centrality (General -> Specific).
    
    skills_df = calculate_generality(skills_df, positives_df)
    
    # Map centrality scores to the pairs
    uid_to_score = dict(zip(skills_df['skill_uid'], skills_df['centrality_score']))
    
    positives_df['src_score'] = positives_df['source_uid'].map(uid_to_score)
    positives_df['tgt_score'] = positives_df['target_uid'].map(uid_to_score)
    
    # Label types
    conditions = [
        (positives_df['source_level'] == 'secondary') & (positives_df['target_level'] == 'tertiary'),
        (positives_df['source_level'] == positives_df['target_level'])
    ]
    choices = ['cross_level_strong', 'same_level_weak']
    positives_df['type'] = np.select(conditions, choices, default='unknown')
    positives_df['label'] = 1  # POSITIVE CLASS
    
    print(f"Positives prepared: {len(positives_df)} pairs")

    # 3. Regenerate Hard Negatives (Ter -> Sec)
    # We re-run the specific check to capture the pairs we dropped in Step 2.
    # (Ideally, we could have saved them, but re-computing kNN for 1435 items is <1 sec)
    print("Regenerating Hard Negatives (Ter -> Sec)...")
    
    # Need embeddings again to find the exact same neighbors
    embeddings = get_embeddings(skills_df)
    k = 15 # Must match Step 2
    knn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1).fit(embeddings)
    _, indices = knn.kneighbors(embeddings)
    
    hard_negatives = []
    uids = skills_df['skill_uid'].values
    levels = skills_df['education_level'].values
    
    for i in range(len(skills_df)):
        src_level = levels[i]
        if src_level != 'tertiary': continue # Only look for Ter -> Sec violations
        
        src_uid = uids[i]
        
        for neighbor_idx in indices[i]:
            if i == neighbor_idx: continue
            
            tgt_level = levels[neighbor_idx]
            if tgt_level == 'secondary':
                # This is a violation (Ter -> Sec) -> Hard Negative
                hard_negatives.append({
                    'source_uid': src_uid,
                    'target_uid': uids[neighbor_idx],
                    'source_level': src_level,
                    'target_level': tgt_level,
                    'type': 'hard_negative_violation',
                    'label': 0
                })

    hard_neg_df = pd.DataFrame(hard_negatives)
    print(f"Hard negatives regenerated: {len(hard_neg_df)}")

    # 4. Generate Random Negatives (Noise)
    # Select random pairs that are NOT in the positives or hard negatives
    print("Generating Random Negatives...")
    n_random = len(positives_df) # Match size of positives for balance
    
    # Simple sampling: random source, random target
    # In a larger graph, checking existence is slow, but here set lookups are fast.
    existing_pairs = set(zip(positives_df['source_uid'], positives_df['target_uid']))
    existing_pairs.update(set(zip(hard_neg_df['source_uid'], hard_neg_df['target_uid'])))
    
    random_pairs = []
    all_uids = skills_df['skill_uid'].values
    
    while len(random_pairs) < n_random:
        s = np.random.choice(all_uids)
        t = np.random.choice(all_uids)
        
        if s == t: continue
        if (s, t) not in existing_pairs:
            random_pairs.append({
                'source_uid': s,
                'target_uid': t,
                'source_level': 'unknown', # optimization: skip lookup
                'target_level': 'unknown',
                'type': 'random_negative',
                'label': 0
            })
            # Add to set to prevent duplicates
            existing_pairs.add((s,t))
            
    rand_neg_df = pd.DataFrame(random_pairs)
    print(f"Random negatives generated: {len(rand_neg_df)}")

    # 5. Merge and Save
    # We only keep essential columns for training
    cols = ['source_uid', 'target_uid', 'label', 'type']
    
    final_df = pd.concat([
        positives_df[cols],
        hard_neg_df[cols],
        rand_neg_df[cols]
    ], ignore_index=True)
    
    # Shuffle
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_file = f"{OUTPUT_DIR}/weak_supervision_data.parquet"
    final_df.to_parquet(output_file)
    
    print("-" * 30)
    print(f"Total training pairs: {len(final_df)}")
    print(f"Breakdown:\n{final_df['type'].value_counts()}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()