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
MODEL_PATH = '/project/home/p200253/hyperbolic/projections/models/poincare_anchored.pt'

EMBEDDING_DIM = 2
EPOCHS = 50
BATCH_SIZE = 1024       # Larger batch for stable gradients
LEARNING_RATE = 0.01    # Slower, more careful learning
ANCHOR_STRENGTH = 2.0   # Gravity strength for Secondary nodes (High!)

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
        # Init small random near zero
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
    def __init__(self, edges_df, node_to_idx, num_negatives=20):
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
    
    # 2. Identify Anchors (Secondary Skills)
    print("Identifying Anchor Nodes (Secondary)...")
    level_map = {}
    for _, row in edges_df.iterrows():
        level_map[row['source_id']] = row['source_level']
        level_map[row['target_id']] = row['target_level']
        
    sec_indices = [i for i in range(len(all_nodes)) if level_map.get(idx_to_node[i]) == 'secondary']
    ter_indices = [i for i in range(len(all_nodes)) if level_map.get(idx_to_node[i]) == 'tertiary']
    
    # Convert to Tensor for fast lookup in loss loop
    sec_tensor = torch.tensor(sec_indices, dtype=torch.long).to(device)
    
    print(f"Anchoring {len(sec_indices)} Secondary nodes. {len(ter_indices)} Tertiary nodes are free.")

    # 3. Init Model with "Nudge" (Still useful as a starting point)
    model = PoincareEmbedding(len(all_nodes), EMBEDDING_DIM).to(device)
    
    with torch.no_grad():
        if sec_indices:
            # Start very close to 0
            model.embeddings.weight[sec_indices] = torch.rand(len(sec_indices), EMBEDDING_DIM).to(device) * 0.01
        if ter_indices:
            # Start further out
            theta = torch.rand(len(ter_indices)).to(device) * 2 * np.pi
            r = torch.rand(len(ter_indices)).to(device) * 0.2 + 0.5
            weights = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
            model.embeddings.weight[ter_indices] = weights

    # 4. Train with Custom Anchored Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dataset = GraphDataset(edges_df, node_to_idx, num_negatives=20)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Starting Anchored Training ({EPOCHS} epochs)...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_reg = 0
        
        for u_idx, v_idx, neg_idxs in dataloader:
            u_idx, v_idx, neg_idxs = u_idx.to(device), v_idx.to(device), neg_idxs.to(device)
            optimizer.zero_grad()
            
            u = model(u_idx)
            v = model(v_idx)
            negs = model(neg_idxs)
            
            # --- Standard Poincare Loss ---
            pos_dist = poincare_distance(u, v)
            u_exp = u.unsqueeze(1)
            neg_dist = poincare_distance(u_exp, negs)
            
            numerator = torch.exp(-pos_dist)
            denominator = numerator + torch.sum(torch.exp(-neg_dist), dim=1)
            task_loss = -torch.log(numerator / denominator).mean()
            
            # --- Anchor Regularization ---
            # We penalize the Norm of Secondary nodes involved in this batch
            # We check which of u_idx are in sec_tensor
            
            # Optimization: Instead of checking membership every batch, 
            # we just apply a global penalty to ALL secondary nodes once per epoch 
            # or simply apply it to the whole batch if they happen to be secondary.
            # Let's do the batch-wise approach for gradients.
            
            # Mask for source nodes in batch
            is_sec_u = torch.isin(u_idx, sec_tensor)
            is_sec_v = torch.isin(v_idx, sec_tensor)
            
            reg_term = torch.tensor(0.0).to(device)
            
            if is_sec_u.any():
                reg_term += torch.sum(u[is_sec_u].norm(dim=1)**2)
            if is_sec_v.any():
                reg_term += torch.sum(v[is_sec_v].norm(dim=1)**2)
                
            # Normalize by batch size to keep scale consistent
            reg_term = reg_term / BATCH_SIZE
            
            loss = task_loss + (ANCHOR_STRENGTH * reg_term)
            
            loss.backward()
            optimizer.step()
            model.project()
            
            total_loss += task_loss.item()
            total_reg += reg_term.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Task Loss: {total_loss/len(dataloader):.4f} | Anchor Reg: {total_reg/len(dataloader):.4f}")

    # 5. Save and Check
    embeddings_np = model.embeddings.weight.detach().cpu().numpy()
    norms = np.linalg.norm(embeddings_np, axis=1)
    
    output_df = pd.DataFrame({
        'skill_uid': [idx_to_node[i] for i in range(len(all_nodes))],
        'h_x': embeddings_np[:, 0],
        'h_y': embeddings_np[:, 1],
        'h_norm': norms
    })
    
    meta_df = pd.read_csv('/project/home/p200253/hyperbolic/projections/files/combined_secondary_tertiary_skills.csv')
    final_df = output_df.merge(meta_df[['skill_uid', 'name', 'education_level']], on='skill_uid', how='left')
    
    final_df.to_csv(f"{OUTPUT_DIR}/hyperbolic_embeddings_2d.csv", index=False)
    
    sec_mean = final_df[final_df['education_level']=='secondary']['h_norm'].mean()
    ter_mean = final_df[final_df['education_level']=='tertiary']['h_norm'].mean()
    
    print("-" * 30)
    print(f"Final Radius (Sec): {sec_mean:.4f}")
    print(f"Final Radius (Ter): {ter_mean:.4f}")
    
    if sec_mean < ter_mean:
        print("✅ SUCCESS: Secondary is significantly closer to center.")
    else:
        print("⚠️ FAILURE: Try increasing ANCHOR_STRENGTH.")

if __name__ == "__main__":
    main()