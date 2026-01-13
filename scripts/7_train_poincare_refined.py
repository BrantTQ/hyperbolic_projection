import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import time

# ==========================================
# CONFIGURATION
# ==========================================
EDGES_CSV = '/project/home/p200253/hyperbolic/projections/files/graph/graph_edges_detailed.csv'
OUTPUT_DIR = '/project/home/p200253/hyperbolic/projections/files/embeddings'
MODEL_PATH = '/project/home/p200253/hyperbolic/projections/models/poincare_model_refined.pt'

EMBEDDING_DIM = 2
EPOCHS = 40  # Increased slightly to allow settling
BATCH_SIZE = 512
LEARNING_RATE = 0.05 # Lowered slightly for stability
NEGATIVE_SAMPLES = 20 # Increased to push unconnected nodes apart harder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# MATH UTILS
# ==========================================
def poincare_distance(u, v, epsilon=1e-5):
    sq_u = torch.sum(u * u, dim=-1, keepdim=True)
    sq_v = torch.sum(v * v, dim=-1, keepdim=True)
    sq_dist = torch.sum(torch.pow(u - v, 2), dim=-1, keepdim=True)
    val = 1 + 2 * sq_dist / ((1 - sq_u) * (1 - sq_v) + epsilon)
    return torch.acosh(torch.clamp(val, min=1.0 + epsilon))

class PoincareEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(PoincareEmbedding, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        # We will manually initialize weights later, so standard init here doesn't matter much
        nn.init.uniform_(self.embeddings.weight, -0.001, 0.001)
        
    def forward(self, inputs):
        return self.embeddings(inputs)
    
    def project(self):
        with torch.no_grad():
            norms = torch.norm(self.embeddings.weight, dim=1, keepdim=True)
            max_norm = 0.9999
            cond = norms >= max_norm
            scale = torch.where(cond, max_norm / (norms + 1e-6), torch.ones_like(norms))
            self.embeddings.weight.mul_(scale)

# ==========================================
# DATA LOADING
# ==========================================
class GraphDataset(Dataset):
    def __init__(self, edges_df, node_to_idx, num_negatives=10):
        self.sources = edges_df['source_id'].map(node_to_idx).values
        self.targets = edges_df['target_id'].map(node_to_idx).values
        self.num_nodes = len(node_to_idx)
        self.num_negatives = num_negatives
        
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx):
        src = self.sources[idx]
        tgt = self.targets[idx]
        negs = torch.randint(0, self.num_nodes, (self.num_negatives,), dtype=torch.long)
        return torch.tensor(src), torch.tensor(tgt), negs

# ==========================================
# MAIN
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading Edges: {EDGES_CSV}")
    edges_df = pd.read_csv(EDGES_CSV)
    
    # Vocabulary Maps
    all_nodes = pd.concat([edges_df['source_id'], edges_df['target_id']]).unique()
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    # 2. Build Initialization Masks
    # We need to know which index belongs to which level to apply the "Nudge"
    print("Applying Stratified Initialization (The 'Nudge')...")
    
    # We can infer level from the edges_df since it has 'source_level' and 'target_level'
    # Create a mapping: uid -> level
    level_map = {}
    for _, row in edges_df.iterrows():
        level_map[row['source_id']] = row['source_level']
        level_map[row['target_id']] = row['target_level']
        
    # Create lists of indices
    sec_indices = [i for i in range(len(all_nodes)) if level_map.get(idx_to_node[i]) == 'secondary']
    ter_indices = [i for i in range(len(all_nodes)) if level_map.get(idx_to_node[i]) == 'tertiary']
    
    # 3. Init Model & Apply Weights
    model = PoincareEmbedding(len(all_nodes), EMBEDDING_DIM).to(device)
    
    # --- THE FIX IS HERE ---
    with torch.no_grad():
        # Secondary: Very close to center (Radius ~ 0.05)
        # We use uniform small noise around 0
        if sec_indices:
            model.embeddings.weight[sec_indices] = torch.rand(len(sec_indices), EMBEDDING_DIM).to(device) * 0.1 - 0.05
            
        # Tertiary: Further out (Radius ~ 0.5)
        # We generate random angles and fixed radius range
        if ter_indices:
            # Random angles
            theta = torch.rand(len(ter_indices)).to(device) * 2 * np.pi
            # Random radius between 0.4 and 0.6
            r = torch.rand(len(ter_indices)).to(device) * 0.2 + 0.4
            
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            
            # Stack and assign
            weights = torch.stack([x, y], dim=1)
            model.embeddings.weight[ter_indices] = weights
            
    print(f"Initialized {len(sec_indices)} Secondary nodes near center.")
    print(f"Initialized {len(ter_indices)} Tertiary nodes at mid-range.")

    # 4. Train
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dataset = GraphDataset(edges_df, node_to_idx, num_negatives=NEGATIVE_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Starting training for {EPOCHS} epochs...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for u_idx, v_idx, neg_idxs in dataloader:
            u_idx, v_idx, neg_idxs = u_idx.to(device), v_idx.to(device), neg_idxs.to(device)
            optimizer.zero_grad()
            
            u = model(u_idx)
            v = model(v_idx)
            negs = model(neg_idxs)
            
            pos_dist = poincare_distance(u, v)
            u_exp = u.unsqueeze(1)
            neg_dist = poincare_distance(u_exp, negs)
            
            numerator = torch.exp(-pos_dist)
            denominator = numerator + torch.sum(torch.exp(-neg_dist), dim=1)
            loss = -torch.log(numerator / denominator).mean()
            
            loss.backward()
            optimizer.step()
            model.project()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

    # 5. Save Results
    print("Saving refined embeddings...")
    embeddings_np = model.embeddings.weight.detach().cpu().numpy()
    norms = np.linalg.norm(embeddings_np, axis=1)
    
    output_df = pd.DataFrame({
        'skill_uid': [idx_to_node[i] for i in range(len(all_nodes))],
        'h_x': embeddings_np[:, 0],
        'h_y': embeddings_np[:, 1],
        'h_norm': norms
    })
    
    # Merge metadata
    meta_df = pd.read_csv('/project/home/p200253/hyperbolic/projections/files/combined_secondary_tertiary_skills.csv')
    final_df = output_df.merge(meta_df[['skill_uid', 'name', 'education_level']], on='skill_uid', how='left')
    
    out_file = f"{OUTPUT_DIR}/hyperbolic_embeddings_2d.csv"
    final_df.to_csv(out_file, index=False)
    print(f"Saved to: {out_file}")
    
    # Quick Check
    sec_mean = final_df[final_df['education_level']=='secondary']['h_norm'].mean()
    ter_mean = final_df[final_df['education_level']=='tertiary']['h_norm'].mean()
    print("-" * 30)
    print(f"New Avg Radius (Sec): {sec_mean:.4f}")
    print(f"New Avg Radius (Ter): {ter_mean:.4f}")
    if sec_mean < ter_mean:
        print("✅ SUCCESS: Hierarchy is correctly oriented.")

if __name__ == "__main__":
    main()