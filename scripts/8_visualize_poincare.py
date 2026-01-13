import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_CSV = '/project/home/p200253/hyperbolic/projections/files/embeddings/hyperbolic_embeddings_2d.csv'
OUTPUT_PLOT = '/project/home/p200253/hyperbolic/projections/files/embeddings/poincare_disk.png'

def main():
    print(f"Loading embeddings from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Create the Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 1. Draw the Poincaré Disk Boundary (Unit Circle)
    circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linestyle='--', linewidth=1, alpha=0.5)
    ax.add_artist(circle)
    
    # 2. Plot Points (Color by Education Level)
    # Secondary = Blue, Tertiary = Orange
    colors = {'secondary': '#1f77b4', 'tertiary': '#ff7f0e'}
    
    for level, group in df.groupby('education_level'):
        ax.scatter(
            group['h_x'], 
            group['h_y'], 
            c=colors.get(level, 'gray'), 
            label=level.capitalize(),
            alpha=0.6, 
            s=20,          # Marker size
            edgecolors='none'
        )
        
    # 3. Annotate "Landmark" Skills
    # We want to see:
    # A) The most central skills (Lowest Norm)
    # B) Some specialized leaf skills (Highest Norm)
    
    df['norm'] = np.sqrt(df['h_x']**2 + df['h_y']**2)
    
    # Pick Top 5 Central (Foundational)
    central_skills = df.nsmallest(5, 'norm')
    
    # Pick Top 5 Peripheral (Specialized)
    peripheral_skills = df.nlargest(5, 'norm')
    
    # Pick 5 Random Tertiary Skills (for variety)
    random_tertiary = df[df['education_level'] == 'tertiary'].sample(5, random_state=42)

    annotate_list = pd.concat([central_skills, peripheral_skills, random_tertiary])
    
    texts = []
    for _, row in annotate_list.iterrows():
        # Shorten very long names for the plot
        name = row['name']
        if len(name) > 25:
            name = name[:23] + "..."
            
        t = ax.text(
            row['h_x'], 
            row['h_y'], 
            name, 
            fontsize=8, 
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )
        texts.append(t)
        
    # 4. Final Formatting
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off') # Hide square axes
    
    plt.title("Hyperbolic Embedding of Skill Curriculum\n(Center = Foundational, Edge = Specialized)", fontsize=16)
    plt.legend(loc='upper right', frameon=True)
    
    # Use adjust_text if installed to prevent overlap, otherwise just save
    try:
        from adjustText import adjust_text
        print("Optimizing text placement (this takes a moment)...")
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
    except ImportError:
        print("Note: 'adjustText' library not found. Labels might overlap.")
    
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {OUTPUT_PLOT}")
    
    # --- Print Stats ---
    avg_sec_norm = df[df['education_level'] == 'secondary']['norm'].mean()
    avg_ter_norm = df[df['education_level'] == 'tertiary']['norm'].mean()
    
    print("-" * 30)
    print(f"Average Radius (Secondary): {avg_sec_norm:.4f}")
    print(f"Average Radius (Tertiary):  {avg_ter_norm:.4f}")
    if avg_sec_norm < avg_ter_norm:
        print("✅ SUCCESS: Secondary skills are closer to the center.")
    else:
        print("⚠️ WARNING: Hierarchy separation is weak.")

if __name__ == "__main__":
    main()