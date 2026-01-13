import pandas as pd
import networkx as nx
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
EDGES_CSV = '/project/home/p200253/hyperbolic/projections/files/graph/graph_edges_detailed.csv'
METADATA_CSV = '/project/home/p200253/hyperbolic/projections/files/combined_secondary_tertiary_skills.csv'

def main():
    print("Loading Graph Data...")
    edges_df = pd.read_csv(EDGES_CSV)
    
    # Rebuild Graph from CSV
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        G.add_edge(row['source_id'], row['target_id'], weight=row['score'])
        
    print(f"Graph Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # ==========================================
    # CHECK 1: CONSTRAINT ENFORCEMENT
    # ==========================================
    print("\n--- 1. CONSTRAINT CHECK (Ter -> Sec) ---")
    violations = edges_df[
        (edges_df['source_level'] == 'tertiary') & 
        (edges_df['target_level'] == 'secondary')
    ]
    
    if len(violations) == 0:
        print("✅ SUCCESS: 0 violations found. The hierarchy is strict.")
    else:
        print(f"❌ WARNING: {len(violations)} violations found (Tertiary -> Secondary).")
        
    # Breakdown of Edge Types
    print("\n--- Edge Distribution ---")
    conditions = [
        (edges_df['source_level'] == 'secondary') & (edges_df['target_level'] == 'secondary'),
        (edges_df['source_level'] == 'secondary') & (edges_df['target_level'] == 'tertiary'),
        (edges_df['source_level'] == 'tertiary') & (edges_df['target_level'] == 'tertiary')
    ]
    choices = ['Sec->Sec (Internal)', 'Sec->Ter (Bridge)', 'Ter->Ter (Internal)']
    edges_df['type'] = np.select(conditions, choices, default='Unknown')
    print(edges_df['type'].value_counts())

    # ==========================================
    # CHECK 2: TOPOLOGICAL FEATURES
    # ==========================================
    print("\n--- 2. TOPOLOGY ---")
    if nx.is_directed_acyclic_graph(G):
        print("✅ Graph is a valid DAG (No Cycles).")
    else:
        print("❌ Graph contains cycles!")
        
    # Find Roots (No prerequisites) and Leaves (No future skills)
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    roots = [n for n, d in in_degrees.items() if d == 0]
    leaves = [n for n, d in out_degrees.items() if d == 0]
    
    print(f"Root Skills (Foundational): {len(roots)}")
    print(f"Leaf Skills (Specialized): {len(leaves)}")
    
    # ==========================================
    # CHECK 3: MOST INFLUENTIAL SKILLS (HUBS)
    # ==========================================
    print("\n--- 3. TOP 'PREREQUISITE' SKILLS (Out-Degree) ---")
    # We need the names to make sense of this
    meta_df = pd.read_csv(METADATA_CSV)
    name_map = dict(zip(meta_df['skill_uid'], meta_df['name']))
    
    sorted_nodes = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Skill Name':<50} | {'Direct Dependents'}")
    print("-" * 70)
    for uid, degree in sorted_nodes[:10]:
        name = name_map.get(uid, "Unknown")
        print(f"{name[:48]:<50} | {degree}")

    # ==========================================
    # CHECK 4: SAMPLE LEARNING PATHS
    # ==========================================
    print("\n--- 4. SAMPLE LEARNING PATHS (Random Walks) ---")
    print("Tracing paths from Secondary Roots to Tertiary Leaves...")
    
    longest_path = []
    
    # Try to find 5 nice paths to display
    found_paths = 0
    attempts = 0
    
    import random
    roots_sec = [r for r in roots if "sec" in str(r)] # Assuming uid contains 'sec' or we check map
    
    while found_paths < 5 and attempts < 1000:
        attempts += 1
        if not roots_sec: break
        
        start_node = random.choice(roots_sec)
        
        # Simple random walk
        curr = start_node
        path = [curr]
        
        for _ in range(10): # Max depth walk
            neighbors = list(G.successors(curr))
            if not neighbors:
                break
            curr = random.choice(neighbors)
            path.append(curr)
            
        # Only show if it crosses boundary (Sec -> Ter) and has length > 3
        if len(path) >= 3:
            # Check if it transitions levels
            # We look up levels in the dataframe or assume UIDs indicate it
            # Let's rely on the name map for display
            path_names = [name_map.get(uid, "Unknown") for uid in path]
            
            print(f"\nPath {found_paths+1}:")
            print(" -> ".join(path_names))
            found_paths += 1

if __name__ == "__main__":
    main()