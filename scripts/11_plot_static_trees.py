import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import math

# ==========================================
# CONFIGURATION
# ==========================================
EDGES_FILE = '/project/home/p200253/hyperbolic/projections/files/graph/graph_edges_detailed.csv'
METADATA_FILE = '/project/home/p200253/hyperbolic/projections/files/combined_secondary_tertiary_skills.csv'
OUTPUT_DIR = '/project/home/p200253/hyperbolic/projections/files/static_trees'

MAX_DEPTH = 3        # How many hops to trace
MAX_NODES = 15       # Strict limit to keep trees readable

# ==========================================
# LAYOUT ENGINE
# ==========================================
def get_layered_pos(G, root_node, mode='prerequisites'):
    """
    Calculates X,Y positions based on distance from root.
    Includes fallback for disconnected nodes.
    """
    pos = {}
    
    # 1. Determine Search Direction
    if mode == 'prerequisites':
        # Graph is Ancestor -> Root. We want distance Root -> Ancestor.
        G_search = G.reverse()
    else:
        # Graph is Root -> Descendant.
        G_search = G
            
    # 2. Calculate Layers (Distance)
    try:
        layers = dict(nx.shortest_path_length(G_search, source=root_node))
    except:
        layers = {root_node: 0}

    # 3. Handle Disconnected Nodes (The Fix)
    # Any node in the subgraph that didn't get a layer (because it was pruned/isolated)
    # gets assigned to the "furthest" layer + 1 so it appears at the edge.
    max_dist = max(layers.values()) if layers else 0
    fallback_layer = max_dist + 1
    
    for n in G.nodes():
        if n not in layers:
            layers[n] = fallback_layer

    # 4. Group by Layer
    layer_map = {}
    final_max_layer = 0
    for node, layer in layers.items():
        if layer not in layer_map: layer_map[layer] = []
        layer_map[layer].append(node)
        final_max_layer = max(final_max_layer, layer)
        
    # 5. Assign X, Y
    for layer_idx, nodes_in_layer in layer_map.items():
        nodes_in_layer.sort(key=lambda x: str(x)) # Sort for deterministic order
        count = len(nodes_in_layer)
        
        for i, node in enumerate(nodes_in_layer):
            y = (i + 1) / (count + 1)
            
            if mode == 'prerequisites':
                # Root Right (Max X), Ancestors Left
                x = final_max_layer - layer_idx
            else:
                # Root Left (0), Descendants Right
                x = layer_idx
            
            pos[node] = (x, y)
            
    return pos, final_max_layer

# ==========================================
# PLOTTING LOGIC
# ==========================================
def plot_tree(G_full, root_id, root_name, mode='prerequisites', id_map=None, level_map=None):
    
    # 1. Extract Subgraph (BFS)
    relevant_nodes = set([root_id])
    
    if mode == 'prerequisites':
        G_search = G_full.reverse()
        title_prefix = "Prerequisites for"
    else:
        G_search = G_full
        title_prefix = "Outcomes of"
        
    # Expand layers
    curr = {root_id}
    for _ in range(MAX_DEPTH):
        next_gen = set()
        for n in curr:
            neighbors = list(G_search.neighbors(n))
            next_gen.update(neighbors)
        relevant_nodes.update(next_gen)
        curr = next_gen
        
    subgraph = G_full.subgraph(relevant_nodes).copy()

    # 2. Smart Pruning (The Fix)
    # If too big, keep only the MAX_NODES closest to root
    if subgraph.number_of_nodes() > MAX_NODES:
        # Calculate distance from root for all nodes
        try:
            dists = nx.shortest_path_length(G_search.subgraph(relevant_nodes), source=root_id)
            # Sort by distance (asc) then take top N
            sorted_nodes = sorted(dists.keys(), key=lambda x: dists[x])
            keep_nodes = sorted_nodes[:MAX_NODES]
            subgraph = subgraph.subgraph(keep_nodes).copy()
        except:
            # Fallback if calculation fails
            pass

    if subgraph.number_of_nodes() < 2:
        return # Skip single dots

    # 3. Layout
    pos, num_layers = get_layered_pos(subgraph, root_id, mode)
    
    # 4. Sizing
    layer_counts = {}
    for n in subgraph.nodes():
        x = pos[n][0]
        layer_counts[x] = layer_counts.get(x, 0) + 1
    max_vertical = max(layer_counts.values()) if layer_counts else 1
    
    # Adaptive size: wider if deep, taller if dense
    fig_width = max(12, num_layers * 3.5)
    fig_height = max(8, max_vertical * 1.2)
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # 5. Draw
    # Colors
    node_colors = []
    for n in subgraph.nodes():
        if n == root_id:
            node_colors.append('#d62728') # Red for Root
        else:
            lvl = level_map.get(n, 'unknown')
            node_colors.append('#1f77b4' if lvl == 'secondary' else '#ff7f0e')

    # Edges
    nx.draw_networkx_edges(subgraph, pos, 
                           edge_color='#AAAAAA', 
                           alpha=0.5, 
                           arrows=True, 
                           arrowstyle='-|>', 
                           arrowsize=25,
                           connectionstyle="arc3,rad=0.1")
    
    # Nodes
    nx.draw_networkx_nodes(subgraph, pos, 
                           node_color=node_colors, 
                           node_size=3500, 
                           edgecolors='white',
                           linewidths=2)
    
    # Labels
    labels = {}
    for n in subgraph.nodes():
        raw_name = id_map.get(n, str(n))
        # Word Wrap
        words = raw_name.split()
        wrapped = ""
        line = ""
        for w in words:
            if len(line) + len(w) > 14:
                wrapped += line + "\n"
                line = w + " "
            else:
                line += w + " "
        wrapped += line
        labels[n] = wrapped.strip()

    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, font_weight='bold', font_color='white')
    
    # Title & Legend
    plt.title(f"{title_prefix}:\n{root_name}", fontsize=22, pad=20)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=15, label='Selected Skill'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=15, label='Secondary'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=15, label='Tertiary')
    ]
    plt.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=12)
    
    plt.axis('off')
    
    # Save
    safe_name = "".join([c for c in root_name if c.isalnum() or c in (' ','-')]).strip().replace(' ','_')
    fname = f"{OUTPUT_DIR}/{mode}_{safe_name}.png"
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

# ==========================================
# MAIN
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading Graph...")
    edges_df = pd.read_csv(EDGES_FILE)
    meta_df = pd.read_csv(METADATA_FILE)
    
    id_map = dict(zip(meta_df['skill_uid'], meta_df['name']))
    level_map = dict(zip(meta_df['skill_uid'], meta_df['education_level']))
    
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        G.add_edge(row['source_id'], row['target_id'])

    # Select interesting nodes
    print("Selecting targets...")
    out_deg = dict(G.out_degree())
    in_deg = dict(G.in_degree())
    
    tertiary_nodes = [n for n in G.nodes() if level_map.get(n)=='tertiary']
    secondary_nodes = [n for n in G.nodes() if level_map.get(n)=='secondary']
    
    # Prioritize highly connected nodes
    tertiary_nodes.sort(key=lambda x: in_deg.get(x,0), reverse=True)
    secondary_nodes.sort(key=lambda x: out_deg.get(x,0), reverse=True)
    
    targets = tertiary_nodes[:10]
    roots = secondary_nodes[:10]
    
    # Add manual keywords
    keywords = ["machine learning", "linear algebra", "statistics", "python", "management"]
    for kw in keywords:
        for uid, name in id_map.items():
            if kw in str(name).lower():
                if level_map.get(uid) == 'tertiary':
                    if uid not in targets: targets.append(uid)
                else:
                    if uid not in roots: roots.append(uid)
                break
                
    print(f"Generating {len(targets)} Prerequisite Trees...")
    for uid in targets:
        plot_tree(G, uid, id_map.get(uid, "Unknown"), 'prerequisites', id_map, level_map)
        
    print(f"Generating {len(roots)} Outcome Trees...")
    for uid in roots:
        plot_tree(G, uid, id_map.get(uid, "Unknown"), 'outcomes', id_map, level_map)

    print("Done.")

if __name__ == "__main__":
    main()