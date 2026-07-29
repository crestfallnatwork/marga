import os
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory containing the generated CSV files
CSV_DIR = "csvs"
OUTPUT_DIR = "plots"

# The targets and algorithms to generate plots for
TARGETS = ["4090", "a6000"]
ALGOS = ["pr", "tc", "bfs"]

# 1. Libraries to include in the plots
LIBRARIES_TO_RENDER = [
    "cugraph",
    "gunrock",
    "graphrtx_optix",
    "graphrtx_hybrid",
    "pictor",
]

# 2. Display names for libraries
LIBRARY_NAMES = {
    "cugraph": "cuGraph",
    "gunrock": "Gunrock",
    "pictor": "MARGA",
    "graphrtx_optix": "GraphRTX (OptiX)",
    "graphrtx_hybrid": "GraphRTX (Hybrid)",
}

# 3. Consistent colors for each library
LIBRARY_COLORS = {
    "cugraph": "#1f77b4",       # Blue
    "gunrock": "#ff7f0e",       # Orange
    "pictor": "#9467bd",        # Purple
    "graphrtx_optix": "#d62728", # Red
    "graphrtx_hybrid": "#2ca02c" # Green
}

# 4. Datasets to include in the plots (change order or comment out to modify)
DATASETS_TO_RENDER = [
    "GAP-road", 
    # "europe_osm", 
    # "soc-LiveJournal1", 
    # "hollywood-2009", 
    "com-Orkut", 
    "soc-twitter-2010", 
    "GAP-twitter", 
    # "com-Friendster", 
    "indochina-2004", 
    # "uk-2005", 
    # "webbase-2001", 
    "it-2004", 
    # "GAP-web", 
    # "delaunay_n24", 
    "kron_g500-logn21", 
    # "GAP-kron", 
    # "GAP-urand"
]

# 5. Display names for graphs (datasets)
DATASET_NAMES = {
    "GAP-road": "GAP-road",
    "europe_osm": "europe_osm",
    "soc-LiveJournal1": "soc-LJ1",
    "hollywood-2009": "hollywood",
    "com-Orkut": "com-Orkut",
    "soc-twitter-2010": "soc-twitter",
    "GAP-twitter": "GAP-twitter",
    "com-Friendster": "Friendster",
    "indochina-2004": "indochina",
    "uk-2005": "uk-2005",
    "webbase-2001": "webbase",
    "it-2004": "it-2004",
    "GAP-web": "GAP-web",
    "delaunay_n24": "delaunay",
    "kron_g500-logn21": "kron_g500",
    "GAP-kron": "GAP-kron",
    "GAP-urand": "GAP-urand"
}

# =============================================================================
# SCRIPT LOGIC
# =============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_data(target, algo):
    data = {}
    for lib in LIBRARIES_TO_RENDER:
        csv_path = os.path.join(CSV_DIR, f"{target}-{lib}-{algo}.csv")
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path, header=None, names=["Graph", "Time"])
        
        # Filter to only the requested datasets
        df = df[df["Graph"].isin(DATASETS_TO_RENDER)].copy()
        
        # Convert Time to numeric; non-numeric values (OOM, Crash, TLE) become NaN
        # NaN values will simply not be rendered in the matplotlib bar chart.
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        
        # Map dataset names
        df["Graph"] = df["Graph"].map(lambda x: DATASET_NAMES.get(x, x))
        
        df = df.set_index("Graph")
        data[lib] = df["Time"]
        
    if not data:
        return pd.DataFrame()
        
    combined = pd.DataFrame(data)
    
    # Reorder rows based on DATASETS_TO_RENDER order to maintain consistent X-axis
    rendered_names = [DATASET_NAMES.get(d, d) for d in DATASETS_TO_RENDER]
    valid_names = [n for n in rendered_names if n in combined.index]
    combined = combined.loc[valid_names]
    
    return combined

for target in TARGETS:
    for algo in ALGOS:
        df = read_data(target, algo)
        if df.empty:
            print(f"Skipping {target}_{algo} (no data)")
            continue
            
        # Get colors mapping correctly aligned with columns
        col_colors = [LIBRARY_COLORS.get(col, "#333333") for col in df.columns]
        
        # Rename columns for the legend
        df_plot = df.rename(columns=LIBRARY_NAMES)
        
        ax = df_plot.plot(
            kind='bar',
            figsize=(15, 7),
            width=0.8,
            color=col_colors,
            logy=True
        )
        
        ax.set_ylabel("Time (ms)")
        ax.set_xlabel("Dataset")
        ax.set_title(f"Algorithm: {algo.upper()} | Target: {target.upper()}")
        
        # Set Y-axis to start from 1
        ax.set_ylim(bottom=1)
        
        # Add a subtle grid on Y-axis
        ax.grid(True, which="major", ls="-", alpha=0.2, axis='y')
        ax.grid(True, which="minor", ls="--", alpha=0.1, axis='y')
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        out_path = os.path.join(OUTPUT_DIR, f"{target}_{algo}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Generated {out_path}")

print("All plots generated successfully.")
