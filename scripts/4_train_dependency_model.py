import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import json
import os
import argparse

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_CSV = '/project/home/p200253/hyperbolic/projections/files/combined_secondary_tertiary_skills.csv'
TRAIN_DATA = '/project/home/p200253/hyperbolic/projections/files/training_data/weak_supervision_data.parquet'
MODEL_SAVE_PATH = '/project/home/p200253/hyperbolic/projections/models/dependency_mlp.pt'

# Fix OpenBLAS warning
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 3072  # As per your description

# Check Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# DATA UTILS
# ==========================================

def parse_vector(vec_str):
    """Robust parser for vectors stored as strings/lists"""
    try:
        return np.array(json.loads(vec_str), dtype=np.float32)
    except:
        if isinstance(vec_str, (list, np.ndarray)):
            return np.array(vec_str, dtype=np.float32)
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

def load_embeddings_map(csv_path):
    print("Loading skill embeddings into memory...")
    df = pd.read_csv(csv_path)
    
    # We use the mean of name+description as the canonical representation
    # (Matching your previous steps)
    
    # Check if we need to parse strings
    if isinstance(df['name_vector'].iloc[0], str):
        print("Parsing vectors...")
        n = np.stack(df['name_vector'].apply(parse_vector).values)
        d = np.stack(df['description_vector'].apply(parse_vector).values)
        vectors = (n + d) / 2.0
    else:
        vectors = (np.stack(df['name_vector'].values) + np.stack(df['description_vector'].values)) / 2.0
        
    # Map UID -> Vector
    return dict(zip(df['skill_uid'], vectors))

class PrerequisiteDataset(Dataset):
    def __init__(self, pairs_df, embedding_map):
        self.pairs = pairs_df
        self.embedding_map = embedding_map
        self.uids_src = pairs_df['source_uid'].values
        self.uids_tgt = pairs_df['target_uid'].values
        self.labels = pairs_df['label'].values.astype(np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Fetch vectors
        u = self.embedding_map.get(self.uids_src[idx])
        v = self.embedding_map.get(self.uids_tgt[idx])
        
        # Safety check for missing keys (shouldn't happen with clean data)
        if u is None: u = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        if v is None: v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
        # Input features: We feed specific interaction features to the NN
        # [u, v, v - u]
        # v - u is crucial: it represents the "vector traversal" from source to target
        
        return torch.tensor(u), torch.tensor(v), torch.tensor(self.labels[idx])

# ==========================================
# MODEL DEFINITION
# ==========================================

class DependencyScorer(nn.Module):
    def __init__(self, input_dim):
        super(DependencyScorer, self).__init__()
        
        # Input size: 
        # Source (3072) + Target (3072) + Difference (3072) = 9216
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
            
            nn.Linear(256, 1) # Output raw logit
        )

    def forward(self, u, v):
        # Feature engineering inside the forward pass
        diff = v - u
        # Concatenate: [Source, Target, Direction]
        x = torch.cat([u, v, diff], dim=1)
        return self.net(x)

# ==========================================
# TRAINING LOOP
# ==========================================

def main():
    # 1. Prepare Data
    emb_map = load_embeddings_map(INPUT_CSV)
    
    print(f"Loading Training Data: {TRAIN_DATA}")
    df = pd.read_parquet(TRAIN_DATA)
    
    # Split Train/Val
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['type'])
    
    print(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)}")
    
    train_ds = PrerequisiteDataset(train_df, emb_map)
    val_ds = PrerequisiteDataset(val_df, emb_map)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Setup Model
    model = DependencyScorer(EMBEDDING_DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss() # More stable than BCELoss + Sigmoid
    
    # 3. Train
    print("\nStarting training...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    best_val_auc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch_u, batch_v, batch_y in train_loader:
            batch_u, batch_v, batch_y = batch_u.to(device), batch_v.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_u, batch_v).squeeze()
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_u, batch_v, batch_y in val_loader:
                batch_u, batch_v, batch_y = batch_u.to(device), batch_v.to(device), batch_y.to(device)
                logits = model(batch_u, batch_v).squeeze()
                probs = torch.sigmoid(logits)
                
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_auc = roc_auc_score(val_targets, val_preds)
        val_acc = accuracy_score(val_targets, np.round(val_preds))
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print("-" * 30)
    print(f"Training complete. Best model saved to: {MODEL_SAVE_PATH}")
    print(f"Best Validation AUC: {best_val_auc:.4f}")

if __name__ == "__main__":
    main()