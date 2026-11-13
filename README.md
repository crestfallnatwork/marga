# MARGA (Mesh-shader ARchitecture for Graph Analytics)

## Building

### Dependencies
* CMake
* Modern C++ compiler
* Vulkan SDK
* Python (for running benchmarking scripts)

### Compilation
Configure the build
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```
Compile
```
cmake --build build --parallel
```

## Benchmarking

### Datasets
After compiling, fetch the datasets using the script in the `datasets` folder
```
python datasets/download_datasets.py
```
Individual datasets can also be fetched
```
python datasets/download_datasets.py com-Orkut
```
Each dataset is stored as `datasets/<graph>/<graph>.mtx`, together with the
`vertices.txt` source list used by BFS and BC.

### Running benchmarks
All runners live in `benchmarks/` and share `benchmarks/config.py`, which reads
its paths from the environment:

| Variable | Meaning | Default |
|---|---|---|
| `MARGA_DATA` | directory holding `<graph>/<graph>.mtx` | `./datasets` |
| `MARGA_OUT` | output directory for the result CSVs | `./artifacts/csvs` |
| `MARGA_BUILD` | MARGA build directory | `./build` |
| `CUGRAPH_BUILD` | cuGraph benchmark build | `./baselines/cugraph/build` |
| `GUNROCK_BUILD` | GunRock build | `./baselines/gunrock/build` |
| `GRAPHTX_BIN` | GraphRTX benchmark binary | unset |

Run one system, or every system, with the same selection flags:
```
python benchmarks/run_marga.py
python benchmarks/run_all.py --systems marga,cugraph,gunrock,graphrtx
python benchmarks/run_all.py --graphs com-Orkut,soc-twitter-2010 --algos tc,bfs
```
Each run writes one CSV per (platform, system, algorithm) to `MARGA_OUT`, with
one `graph,ms` line per graph and `OOM` for a cell that did not complete.

Peak device memory is measured separately, one process per cell:
```
python benchmarks/measure_memory.py
```

### Figures
The result CSVs are plotted with the scripts in `benchmarks/`:
```
python benchmarks/plot_graphs.py 4090 tc
python benchmarks/plot_memory.py
python benchmarks/plot_tc_ablation.py
```
Figures are written to `artifacts/figs/`.

## Notes
For the GraphRTX baseline, a fixed version of GraphRTX was used. The exact source code can be found at [this](https://github.com/crestfallnatwork/marga-graphrtx-fixes) repo.
