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
MODEL_PATH = '/project/home/p200253/hyperbolic/projections/models/poincare_model.pt'

EMBEDDING_DIM = 2  # 2D is best for visualization; use 5 or 10 for downstream tasks
EPOCHS = 30
BATCH_SIZE = 512
LEARNING_RATE = 0.1
NEGATIVE_SAMPLES = 10  # Number of noise examples per positive edge

# Check Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# MATH UTILS (POINCARE BALL)
# ==========================================
# Standard Poincare distance and projection formulas

def poincare_distance(u, v, epsilon=1e-5):
    """
    Calculates hyperbolic distance between u and v in the Poincaré ball.
    d(u, v) = arccosh(1 + 2 * ||u-v||^2 / ((1 - ||u||^2)(1 - ||v||^2)))
    """
    sq_u = torch.sum(u * u, dim=-1, keepdim=True)
    sq_v = torch.sum(v * v, dim=-1, keepdim=True)
    
    sq_dist = torch.sum(torch.pow(u - v, 2), dim=-1, keepdim=True)
    
    val = 1 + 2 * sq_dist / ((1 - sq_u) * (1 - sq_v) + epsilon)
    
    # Numerical stability safeguard for arccosh
    return torch.acosh(torch.clamp(val, min=1.0 + epsilon))

class PoincareEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(PoincareEmbedding, self).__init__()
        # Initialize uniformly near the origin (1e-4)
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -0.001, 0.001)
        
    def forward(self, inputs):
        return self.embeddings(inputs)
    
    def project(self):
        """
        Constraint enforcement: If ||x|| >= 1, project back into the ball.
        We scale to max_norm = 1 - epsilon
        """
        with torch.no_grad():
            norms = torch.norm(self.embeddings.weight, dim=1, keepdim=True)
            max_norm = 0.9999
            cond = norms >= max_norm
            # If norm > max, scale it down. Otherwise leave it.
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
        
        # Negative Sampling: Random nodes that are NOT the target
        # (Simplified: we just pick random nodes, collision probability is low)
        negs = torch.randint(0, self.num_nodes, (self.num_negatives,), dtype=torch.long)
        
        return torch.tensor(src), torch.tensor(tgt), negs

# ==========================================
# TRAINING LOOP
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # 1. Load Graph
    print(f"Loading Edges: {EDGES_CSV}")
    edges_df = pd.read_csv(EDGES_CSV)
    
    # Create vocabulary
    all_nodes = pd.concat([edges_df['source_id'], edges_df['target_id']]).unique()
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    print(f"Vocabulary Size: {len(all_nodes)}")
    
    dataset = GraphDataset(edges_df, node_to_idx, num_negatives=NEGATIVE_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Init Model
    model = PoincareEmbedding(len(all_nodes), EMBEDDING_DIM).to(device)
    
    # We use Riemannian SGD approximation (standard SGD usually works fine for Poincare if learning rate is small)
    # Ideally use 'rsgd', but Adam with projection works for "good enough" embeddings
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting training for {EPOCHS} epochs on {device}...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for u_idx, v_idx, neg_idxs in dataloader:
            u_idx, v_idx, neg_idxs = u_idx.to(device), v_idx.to(device), neg_idxs.to(device)
            
            optimizer.zero_grad()
            
            u = model(u_idx)       # (Batch, Dim)
            v = model(v_idx)       # (Batch, Dim)
            negs = model(neg_idxs) # (Batch, Negs, Dim)
            
            # --- LOSS CALCULATION ---
            # 1. Positive distance (minimize)
            pos_dist = poincare_distance(u, v) # (Batch, 1)
            
            # 2. Negative distance (maximize)
            # Expand u to match negs shape: (Batch, 1, Dim)
            u_expanded = u.unsqueeze(1) 
            neg_dist = poincare_distance(u_expanded, negs) # (Batch, Negs, 1)
            
            # Log-Likelihood Loss (Nickel & Kiela)
            # Loss = -log( exp(-pos) / sum(exp(-neg)) )
            # Simplified proxy: Minimize pos_dist, Maximize neg_dist
            # We use a margin ranking loss proxy or direct log-sigmoid
            
            # Softmax style loss:
            # We want exp(-d_pos) to be large relative to exp(-d_neg)
            numerator = torch.exp(-pos_dist)
            denominator = numerator + torch.sum(torch.exp(-neg_dist), dim=1)
            loss = -torch.log(numerator / denominator).mean()
            
            loss.backward()
            optimizer.step()
            
            # --- PROJECTION STEP ---
            # Crucial for Hyperbolic space: Ensure points stay inside the ball
            model.project()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")
        
        # Decay LR
        if (epoch+1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

    print(f"Training finished in {time.time() - start_time:.1f}s")
    
    # 3. Save Embeddings
    print("Saving embeddings...")
    embeddings_np = model.embeddings.weight.detach().cpu().numpy()
    
    # Calculate norms (distance from center = generality)
    norms = np.linalg.norm(embeddings_np, axis=1)
    
    output_df = pd.DataFrame({
        'skill_uid': [idx_to_node[i] for i in range(len(all_nodes))],
        'h_x': embeddings_np[:, 0],
        'h_y': embeddings_np[:, 1],
        'h_norm': norms
    })
    
    # Merge with original metadata for easier analysis
    meta_df = pd.read_csv('/project/home/p200253/hyperbolic/projections/files/combined_secondary_tertiary_skills.csv')
    final_df = output_df.merge(meta_df[['skill_uid', 'name', 'education_level']], on='skill_uid', how='left')
    
    out_file = f"{OUTPUT_DIR}/hyperbolic_embeddings_2d.csv"
    final_df.to_csv(out_file, index=False)
    print(f"Saved hyperbolic coordinates to: {out_file}")
    
    # Save Model
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved.")

if __name__ == "__main__":
    main()