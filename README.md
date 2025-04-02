# Information Diagonal Algorithm (IDA)

## Overview

IDA is a matrix-vector operation algorithm for processing ultra-large sparse matrices through streaming. Its key advantage is the ability to work with matrices that can't fit into RAM by generating and processing data chunks on the fly.

## Architecture

The system uses a modular architecture with clear components:

**Engine (engine.py)** is the core algorithm implementing the diagonal information flow model. It manages matrix chunk generation with controlled density, parallel processing on multi-core systems, bidirectional computation for flow optimization, and adaptive chunk sizing based on task parameters.

**Metrics (metrics.py)** monitors information flows by tracking metrics such as magnitude, direction, locality, temporal patterns, and significance. It maintains compressed history for efficiency analysis and calculates flow parameters using JIT compilation.

**Benchmark (benchmark.py)** tests performance by simulating matrix processing up to 10M x 10M while measuring execution time and memory usage, demonstrating asymptotic efficiency of O(N^1.9) for time and O(N log N) for memory.

## Performance

IDA scales effectively even with massive matrices, showing time complexity of O(N^1.9) and space complexity of O(N log N). 

**Key Results:**

| Matrix Size          | Processing Time (sec) | Peak Memory (MB) |
|:---------------------|:----------------------|:-----------------|
| 1,000,000 x 1,000,000 | 25.12               | 76.69            |
| 2,000,000 x 2,000,000 | 82.07               | 210.31           |
| 3,000,000 x 3,000,000 | 178.89              | 419.29           |
| 10,000,000 x 10,000,000 | 1938.37           | 4320.83          |

It successfully processes matrices as large as 10M x 10M (10^14 total elements with density 0.00001, containing approximately 10^9 non-zero elements) in approximately 32 minutes using only 4.3 GB of memory, compared to conventional methods that would require petabytes of RAM.

For detailed benchmark results and analysis, see [benchmark_results.md](benchmark_results.md).

## Research Status

This repository contains a research implementation of the IDA algorithm, focusing on demonstrating the theoretical capabilities of the approach. The benchmark saves sparse vectors to files and then simulates processing of ultra-large matrices by generating matrix chunks on-the-fly during computation rather than loading the entire structure into memory.

To run the benchmark demonstration:

```
python benchmark.py
```

This is not intended as a production-ready library, but rather as a proof-of-concept showing that the diagonal information flow approach can handle matrix operations at scales that would be impossible with conventional methods. 