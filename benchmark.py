import numpy as np
import scipy.sparse
import time
import tracemalloc
from pathlib import Path
import gc  # Garbage collector control
from typing import Tuple

# Import our engine
from engine import IDAEngine

# --- Constants and Parameters ---
MATRIX_ROWS = 1_000_000    
MATRIX_COLS = 1_000_000    
DENSITY = 0.00001
VECTOR_DENSITY = 0.001
DATA_FILE = Path("1M_benchmark_data.npz")

IDA_CHUNK_SIZE = 50_000

# --- Functions ---

def generate_and_save_data(rows: int, cols: int, density: float, vector_density: float, filename: Path):
    """Generates parameters for a large task and saves them to an .npz file for reference."""
    print(f"Generating task parameters: Matrix {rows}x{cols} (density {density}), Vector (density {vector_density})...")
    start_time = time.time()

    nnz_matrix = int(rows * cols * density)
    print("  - Generating indices and data for the matrix...")
    row_indices = np.random.randint(0, rows, size=nnz_matrix, dtype=np.int32)
    col_indices = np.random.randint(0, cols, size=nnz_matrix, dtype=np.int32)
    data = np.random.randn(nnz_matrix).astype(np.float32)
    print("  - Creating CSR matrix for saving (requires additional memory)...")
    try:
        matrix_csr = scipy.sparse.csr_matrix((data, (row_indices, col_indices)), shape=(rows, cols))
        print("  - CSR matrix created, generating vector...")
    except MemoryError:
        print("  - Error: Not enough memory to create full CSR matrix for saving.")
        print("  - Matrix file saving will be skipped, but vector will be saved.")
        matrix_csr = None

    nnz_vector = int(cols * vector_density)
    vector_indices = np.random.choice(cols, nnz_vector, replace=False)
    vector = np.zeros(cols, dtype=np.float32)
    vector[vector_indices] = np.random.randn(nnz_vector).astype(np.float32)
    print("  - Vector generated, saving data...")

    vector_filename = Path(str(filename).replace(".npz", "_vec.npz"))
    if matrix_csr is not None:
        scipy.sparse.save_npz(filename, matrix_csr)
        np.savez(vector_filename, vector=vector)
        print(f"Task parameters saved to '{filename}' and '{vector_filename}'")
        del matrix_csr
    else:
        np.savez(vector_filename, vector=vector)
        print(f"Vector parameters saved to '{vector_filename}'. Matrix not saved due to insufficient memory.")

    end_time = time.time()
    print(f"Parameter generation time: {end_time - start_time:.2f} sec.")

    del vector, row_indices, col_indices, data
    gc.collect()

def benchmark_ida(rows: int, cols: int, density: float, vector_file: Path, chunk_size: int) -> Tuple[float, float]:
    """Measures time and peak memory for IDA, simulating stream processing."""
    print("\nStarting IDA benchmark (stream processing simulation)...")

    print("  - IDA: Loading vector from file...")
    try:
        vector_data = np.load(vector_file)
        vector = vector_data['vector']
        print("  - IDA: Vector loaded.")
    except FileNotFoundError:
        print(f"  - Error: Vector file '{vector_file}' not found. Cannot start IDA.")
        return -1.0, -1.0

    print("  - IDA: Initializing engine...")
    engine = IDAEngine()
    print("  - IDA: Engine initialized, starting matrix processing simulation...")

    tracemalloc.start()
    start_time = time.time()

    result_ida = engine.process_matrix(rows, cols, density, vector, chunk_size)
    print("  - IDA: Matrix processing simulation completed.")

    end_time = time.time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    exec_time = end_time - start_time
    peak_mem_mb = peak_mem / (1024 * 1024)

    print(f"IDA (simulation) completed.")
    print(f"  - Execution time: {exec_time:.2f} sec.")
    print(f"  - Peak memory usage (during processing): {peak_mem_mb:.2f} MB")

    del vector, vector_data, engine, result_ida
    gc.collect()

    return exec_time, peak_mem_mb

# --- Main Block ---

if __name__ == "__main__":
    print("="*50)
    print("  IDA Benchmark: Large-Scale Matrix Processing Simulation") 
    print(f"  Matrix Parameters: {MATRIX_ROWS:,}x{MATRIX_COLS:,}, Density: {DENSITY}") 
    print("  Goal: Demonstrate IDA's ability to process parameters")
    print("        of a task that wouldn't fit in RAM, with low memory")
    print("        consumption DURING PROCESSING (generating chunks on the fly).")
    print("="*50)

    vector_file = Path(str(DATA_FILE).replace(".npz", "_vec.npz"))
    if not vector_file.exists():
        generate_and_save_data(MATRIX_ROWS, MATRIX_COLS, DENSITY, VECTOR_DENSITY, DATA_FILE)
    else:
        print(f"Found existing vector file '{vector_file}'. Data generation skipped.")
        if not DATA_FILE.exists():
             print(f"Warning: Matrix file '{DATA_FILE}' not found, although vector file exists.")
             print("IDA benchmark will use only parameters for chunk generation.")

    ida_time, ida_mem = benchmark_ida(MATRIX_ROWS, MATRIX_COLS, DENSITY, vector_file, IDA_CHUNK_SIZE)

    if ida_time >= 0:
        print("\n" + "="*50)
        print("            FINAL RESULTS (IDA)")
        print("            (Stream Processing Simulation)") 
        print("="*50)
        print(f"| Matrix        | Time (sec)  | Peak Memory (MB)     |") 
        print(f"| Parameters    | (Processing)| (During Processing)  |") 
        print(f"|----------------|-------------|---------------------|")
        print(f"| {MATRIX_ROWS:,}x{MATRIX_COLS:,} | {ida_time:<11.2f} | {ida_mem:<19.2f} |")
        print("="*50)
    else:
        print("\nIDA benchmark was not successfully completed due to an error.")

    # Optional: delete data files after benchmark
    # print("Cleaning up data files...")
    # DATA_FILE.unlink(missing_ok=True)
    # vector_file.unlink(missing_ok=True)
    # print("Data files deleted.")
