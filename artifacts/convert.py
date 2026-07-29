import os
import re
import csv

indir = "/home/karl/Code/mtech-proj/pictor_graphs/artifacts"
outdir = os.path.join(indir, "csvs")
os.makedirs(outdir, exist_ok=True)

graphs_19 = [
    "GAP-road", "europe_osm", "soc-LiveJournal1", "hollywood-2009", "com-Orkut",
    "soc-twitter-2010", "GAP-twitter", "com-Friendster", "indochina-2004",
    "uk-2005", "webbase-2001", "it-2004", "GAP-web", "delaunay_n24",
    "kron_g500-logn21", "GAP-kron", "GAP-urand", "kmer_V1r", "nlpkkt240"
]

# Exclude kmer_V1r and nlpkkt240 as requested
graphs = graphs_19[:17]

def write_csv(target, library, algo, data):
    filename = f"{target}-{library}-{algo}.csv"
    filepath = os.path.join(outdir, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        for g in graphs:
            # GAP-kron and GAP-urand should be crash for now if missing
            val = data.get(g, "Crash")
            
            if val == "" or val is None:
                val = "OOM"
            elif isinstance(val, str):
                val = val.strip()
                if val == "":
                    val = "OOM"
                    
            writer.writerow([g, val])

def parse_unstructured(filepath):
    # Returns {algo: {graph: val}}
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    
    data = {'pr': {}, 'tc': {}, 'bfs': {}}
    
    # find indices
    pr_idx = -1
    tc_idx = -1
    bfs_idx = -1
    for i, line in enumerate(lines):
        if line.strip().lower() == "pagerank":
            pr_idx = i
        elif line.strip().lower() == "tc":
            tc_idx = i
        elif line.strip().lower() == "bfs":
            bfs_idx = i
            
    if pr_idx != -1:
        for i in range(19):
            if pr_idx + 1 + i < len(lines):
                data['pr'][graphs_19[i]] = lines[pr_idx + 1 + i]
    if tc_idx != -1:
        for i in range(19):
            if tc_idx + 1 + i < len(lines):
                data['tc'][graphs_19[i]] = lines[tc_idx + 1 + i]
    if bfs_idx != -1:
        for i in range(19):
            if bfs_idx + 1 + i < len(lines):
                data['bfs'][graphs_19[i]] = lines[bfs_idx + 1 + i]
                
    return data

def parse_markdown_table_simple(filepath):
    # For a6000-cugraph, a6000-pictor
    # | Graph | PageRank (ms) | TC (ms) | BFS (ms) |
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    data = {'pr': {}, 'tc': {}, 'bfs': {}}
    for line in lines:
        if line.startswith('|') and '---' not in line and 'PageRank' not in line:
            parts = [p.strip() for p in line.strip().strip('|').split('|')]
            if len(parts) >= 4 and parts[0] in graphs:
                g = parts[0]
                data['pr'][g] = parts[1]
                data['tc'][g] = parts[2]
                data['bfs'][g] = parts[3]
    return data

def parse_markdown_table_graphrtx(filepath):
    # | Graph | TC OptiX (ms) | TC Hybrid (ms) | PR OptiX (ms) | PR Hybrid (ms) | BFS OptiX (ms) | BFS Hybrid (ms) |
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    data_optix = {'pr': {}, 'tc': {}, 'bfs': {}}
    data_hybrid = {'pr': {}, 'tc': {}, 'bfs': {}}
    for line in lines:
        if line.startswith('|') and '---' not in line and 'TC OptiX' not in line:
            parts = [p.strip() for p in line.strip().strip('|').split('|')]
            if len(parts) >= 7 and parts[0] in graphs:
                g = parts[0]
                data_optix['tc'][g] = parts[1]
                data_hybrid['tc'][g] = parts[2]
                data_optix['pr'][g] = parts[3]
                data_hybrid['pr'][g] = parts[4]
                data_optix['bfs'][g] = parts[5]
                data_hybrid['bfs'][g] = parts[6]
    return data_optix, data_hybrid

def parse_a6000_gunrock(filepath):
    data = {'pr': {}, 'tc': {}, 'bfs': {}}
    with open(filepath, 'r') as f:
        content = f.read()
    matches = re.finditer(r"Results for ([\w-]+): PR=([^,]+)ms, TC=([^,]+)ms, BFS=([^,\n]+)ms", content)
    for m in matches:
        g, pr, tc, bfs = m.groups()
        data['pr'][g] = pr
        data['tc'][g] = tc
        data['bfs'][g] = bfs
    
    return data

# Process files
files = [
    "4090-cugraph", "4090-graphrtx", "4090-gunrock", "4090-pictor",
    "a6000-cugraph", "a6000-graphrtx", "a6000-gunrock", "a6000-pictor"
]

for fname in files:
    filepath = os.path.join(indir, fname)
    target = fname.split('-')[0]
    library = fname.split('-')[1]
    
    if library in ["cugraph", "pictor", "gunrock"]:
        if target == "4090":
            data = parse_unstructured(filepath)
        elif fname == "a6000-gunrock":
            data = parse_a6000_gunrock(filepath)
        else: # a6000-cugraph, a6000-pictor
            data = parse_markdown_table_simple(filepath)
            
        write_csv(target, library, "pr", data['pr'])
        write_csv(target, library, "tc", data['tc'])
        write_csv(target, library, "bfs", data['bfs'])
        
    elif library == "graphrtx":
        data_optix, data_hybrid = parse_markdown_table_graphrtx(filepath)
        write_csv(target, "graphrtx_optix", "pr", data_optix['pr'])
        write_csv(target, "graphrtx_optix", "tc", data_optix['tc'])
        write_csv(target, "graphrtx_optix", "bfs", data_optix['bfs'])
        
        write_csv(target, "graphrtx_hybrid", "pr", data_hybrid['pr'])
        write_csv(target, "graphrtx_hybrid", "tc", data_hybrid['tc'])
        write_csv(target, "graphrtx_hybrid", "bfs", data_hybrid['bfs'])

print("Done generating CSVs.")
