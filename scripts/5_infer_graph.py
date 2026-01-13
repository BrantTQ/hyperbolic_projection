import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import json
import os
import sys

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_CANDIDATES = '/project/home/p200253/hyperbolic/projections/files/candidates/candidates_k15_mode_mean.parquet'
INPUT_METADATA = '/project/home/p200253/hyperbolic/projections/files/combined_secondary_tertiary_skills.csv'
MODEL_PATH = '/project/home/p200253/hyperbolic/projections/models/dependency_mlp.pt'
OUTPUT_DIR = '/project/home/p200253/hyperbolic/projections/files/graph'

EMBEDDING_DIM = 3072
BATCH_SIZE = 1024
SCORE_THRESHOLD = 0.55  # Minimum confidence to consider an edge

# Fix OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# MODEL CLASS (Must Match Step 4)
# ==========================================
class DependencyScorer(nn.Module):
    def __init__(self, input_dim):
        super(DependencyScorer, self).__init__()
        combined_dim = input_dim * 3 
        self.net = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, u, v):
        diff = v - u
        x = torch.cat([u, v, diff], dim=1)
        return self.net(x)

# ==========================================
# DATA LOADING
# ==========================================
def parse_vector(vec_str):
    try:
        return np.array(json.loads(vec_str), dtype=np.float32)
    except:
        if isinstance(vec_str, (list, np.ndarray)):
            return np.array(vec_str, dtype=np.float32)
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

def load_data(csv_path):
    print("Loading metadata and embeddings...")
    df = pd.read_csv(csv_path)
    
    # 1. Parse Vectors
    if isinstance(df['name_vector'].iloc[0], str):
        n = np.stack(df['name_vector'].apply(parse_vector).values)
        d = np.stack(df['description_vector'].apply(parse_vector).values)
        vecs = (n + d) / 2.0
    else:
        vecs = (np.stack(df['name_vector'].values) + np.stack(df['description_vector'].values)) / 2.0
    
    emb_map = dict(zip(df['skill_uid'], vecs))
    
    # 2. Create Name & Level Maps (for Plotting/CSV)
    name_map = dict(zip(df['skill_uid'], df['name']))
    level_map = dict(zip(df['skill_uid'], df['education_level']))
    
    return emb_map, name_map, level_map

class InferenceDataset(Dataset):
    def __init__(self, df, emb_map):
        self.uids_src = df['source_uid'].values
        self.uids_tgt = df['target_uid'].values
        self.emb_map = emb_map

    def __len__(self):
        return len(self.uids_src)

    def __getitem__(self, idx):
        u = self.emb_map[self.uids_src[idx]]
        v = self.emb_map[self.uids_tgt[idx]]
        return torch.tensor(u), torch.tensor(v)

# ==========================================
# VISUALIZATION FUNCTION
# ==========================================
def plot_graph(G, output_path):
    print("Generating Graph Plot...")
    plt.figure(figsize=(20, 12))
    
    # 1. Create Layout
    # We enforce a "Left -> Right" layout based on education level
    pos = {}
    
    # Separate nodes by level
    sec_nodes = [n for n, attr in G.nodes(data=True) if attr.get('level') == 'secondary']
    ter_nodes = [n for n, attr in G.nodes(data=True) if attr.get('level') == 'tertiary']
    
    # Initialize positions: Sec on Left (x < 0), Ter on Right (x > 0)
    # We use random y to start, then let spring_layout smooth it out
    for node in sec_nodes:
        pos[node] = np.array([-1 + np.random.rand()*0.5, np.random.rand()])
    for node in ter_nodes:
        pos[node] = np.array([1 + np.random.rand()*0.5, np.random.rand()])
        
    # Apply Spring Layout (iterations=50 is usually enough to untangle)
    # We fix the x-coordinates somewhat by using a small 'k' (repulsion)
    pos = nx.spring_layout(G, pos=pos, k=0.15, iterations=50)
    
    # 2. Draw
    # Draw Edges (Transparent grey)
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='gray', arrows=False)
    
    # Draw Nodes (Color by level)
    nx.draw_networkx_nodes(G, pos, nodelist=sec_nodes, node_color='#1f77b4', node_size=30, label='Secondary')
    nx.draw_networkx_nodes(G, pos, nodelist=ter_nodes, node_color='#ff7f0e', node_size=30, label='Tertiary')
    
    # 3. Finalize
    plt.title("Inferred Prerequisite Graph (Secondary -> Tertiary)", fontsize=20)
    plt.legend(markerscale=3)
    plt.axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")

# ==========================================
# MAIN
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    emb_map, name_map, level_map = load_data(INPUT_METADATA)
    
    print(f"Loading candidates from {INPUT_CANDIDATES}")
    candidates_df = pd.read_parquet(INPUT_CANDIDATES)
    
    # 2. Load Model & Score
    print("Loading model...")
    model = DependencyScorer(EMBEDDING_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    print("Running inference...")
    dataset = InferenceDataset(candidates_df, emb_map)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    all_scores = []
    with torch.no_grad():
        for batch_u, batch_v in loader:
            batch_u, batch_v = batch_u.to(device), batch_v.to(device)
            probs = torch.sigmoid(model(batch_u, batch_v).squeeze())
            all_scores.extend(probs.cpu().numpy())
            
    candidates_df['dependency_score'] = all_scores
    
    # 3. Filter & Greedy DAG Construction
    strong_edges = candidates_df[candidates_df['dependency_score'] > SCORE_THRESHOLD].copy()
    strong_edges = strong_edges.sort_values(by='dependency_score', ascending=False)
    
    print(f"Building DAG from {len(strong_edges)} candidate edges...")
    G = nx.DiGraph()
    
    # Add nodes with attributes (important for plotting and CSV)
    for uid in emb_map.keys():
        G.add_node(uid, label=name_map.get(uid, "Unknown"), level=level_map.get(uid, "unknown"))
    
    edges_added = 0
    skipped = 0
    final_edges_list = []
    
    for _, row in strong_edges.iterrows():
        src, tgt, score = row['source_uid'], row['target_uid'], row['dependency_score']
        
        # Cycle Check
        if not nx.has_path(G, tgt, src):
            G.add_edge(src, tgt, weight=score)
            edges_added += 1
            
            # Prepare data for CSV
            final_edges_list.append({
                'source_id': src,
                'source_name': name_map.get(src),
                'source_level': level_map.get(src),
                'target_id': tgt,
                'target_name': name_map.get(tgt),
                'target_level': level_map.get(tgt),
                'score': score
            })
        else:
            skipped += 1
    
    print(f"Graph Built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print(f"Cycles prevented: {skipped}")

    # 4. Save Detailed CSV
    csv_path = f"{OUTPUT_DIR}/graph_edges_detailed.csv"
    edges_df = pd.DataFrame(final_edges_list)
    edges_df.to_csv(csv_path, index=False)
    print(f"Detailed CSV saved to: {csv_path}")
    
    # 5. Save Gephi File (GEXF)
    # GEXF supports attributes, so we can color by 'level' in Gephi easily
    nx.write_gexf(G, f"{OUTPUT_DIR}/skill_graph.gexf")
    print("Gephi file saved.")

    # 6. Save Plot (PNG)
    plot_path = f"{OUTPUT_DIR}/skill_graph_plot.png"
    plot_graph(G, plot_path)

if __name__ == "__main__":
    main()