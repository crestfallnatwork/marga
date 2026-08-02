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
cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release 
```
Compile
```
cmake --build build --parallel
```

## Benchmarking
### Datasets
After compiling, fetch the datasets using the provided script in the `datasets` folder
```
python download_datasets.py
```
Individual datasets can also be fetched
```
python download_datasets.py com-Orkut
```

### Running benchmarks
Once all the datasets have been fetched, the benchmarks can be run using the `run_benchmarks.py` script provided in the root directory
```
python run_benchmarks.py
```
The exact datasets to run can be configured inside the script.

## Notes 
For the GraphRTX baseline, a fixed version of GraphRTX was used. The exact source code can be found at [this](https://github.com/crestfallnatwork/marga-graphrtx-fixes) repo.
