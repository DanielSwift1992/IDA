"""Core engine module for IDA algorithm"""

import numpy as np
import time
import psutil
from numba import jit
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from metrics import MetricsTracker, FlowMetrics, calculate_flow

@dataclass
class Chunk:
    """Compact chunk representation"""
    start_row: int
    end_row: int
    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    direction: int  # 1 for forward, -1 for backward

@jit(nopython=True)
def ida_matrix_vector_multiply(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, 
                             vector: np.ndarray, chunk_size: int, history: np.ndarray,
                             direction: int = 1) -> Tuple[np.ndarray, Tuple[float, int, float, float, float]]:
    """Optimized IDA implementation with flow tracking and symmetric processing"""
    result = np.zeros(len(indptr) - 1)
    
    # Calculate flow metrics
    flow_metrics = calculate_flow(data, indices, history, chunk_size)
    
    # Process matrix-vector multiplication with direction
    if direction > 0:
        # Forward processing
        for i in range(len(indptr) - 1):
            start = indptr[i]
            end = indptr[i + 1]
            for j in range(start, end):
                if indices[j] < len(vector) and vector[indices[j]] != 0:
                    result[i] += data[j] * vector[indices[j]]
    else:
        # Backward processing
        for i in range(len(indptr) - 2, -1, -1):
            start = indptr[i]
            end = indptr[i + 1]
            for j in range(end - 1, start - 1, -1):
                if indices[j] < len(vector) and vector[indices[j]] != 0:
                    result[i] += data[j] * vector[indices[j]]
    
    return result, flow_metrics

def generate_chunk(rows: int, cols: int, density: float, start_row: int, direction: int = 1) -> Chunk:
    """Generate chunk with CSR format and direction"""
    total_elements = rows * cols
    nnz = int(total_elements * density)
    
    # Generate random positions
    positions = np.random.randint(0, total_elements, size=nnz)
    positions = np.unique(positions)
    while len(positions) < nnz:
        new_positions = np.random.randint(0, total_elements, size=nnz - len(positions))
        positions = np.unique(np.concatenate([positions, new_positions]))
    
    row_indices = positions // cols
    col_indices = positions % cols
    row_indices += start_row
    
    # Sort for CSR format
    sort_idx = np.argsort(row_indices)
    row_indices = row_indices[sort_idx]
    col_indices = col_indices[sort_idx]
    
    # Generate values
    data = np.random.randn(len(positions))
    
    # Create indptr
    indptr = np.zeros(rows + 1, dtype=np.int32)
    for i in range(len(positions)):
        indptr[row_indices[i] - start_row + 1] += 1
    indptr = np.cumsum(indptr)
    
    return Chunk(
        start_row=start_row,
        end_row=start_row + rows,
        data=data,
        indices=col_indices,
        indptr=indptr,
        direction=direction
    )

def process_chunk(args: Tuple) -> Tuple[int, int, np.ndarray, FlowMetrics]:
    """Process a single chunk with direction"""
    chunk, vector, history_array = args
    chunk_start_time = time.time()
    
    # Process chunk with direction
    result, flow_metrics = ida_matrix_vector_multiply(
        chunk.data, chunk.indices, chunk.indptr, vector,
        chunk.end_row - chunk.start_row, history_array,
        chunk.direction
    )
    
    # Record metrics
    processing_time = time.time() - chunk_start_time
    memory_usage = psutil.Process().memory_info().rss / (1024**3)  # GB
    
    flow = FlowMetrics(
        magnitude=flow_metrics[0],
        direction=flow_metrics[1],
        locality=flow_metrics[2],
        temporal=flow_metrics[3],
        significance=flow_metrics[4]
    )
    
    return chunk.start_row, chunk.end_row, result, flow

class IDAEngine:
    """Main engine for IDA algorithm with parallel processing and symmetry"""
    def __init__(self, config=None):
        self.metrics_tracker = MetricsTracker()
        self.num_cores = multiprocessing.cpu_count()
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.metrics_history = []
    
    def process_matrix(self, rows: int, cols: int, density: float, 
                      vector: np.ndarray, initial_chunk_size: int = 10_000) -> np.ndarray:
        """Process matrix with parallel adaptive chunking and symmetric processing"""
        result = np.zeros(rows)
        # Determine actual cores to use (ensure all CPUs are utilized)
        cores_to_use = self.num_cores
        
        print(f"\nStarting parallel adaptive processing with {cores_to_use} cores...")
        print("=" * 50)
        
        # Calculate actual number of chunks based on matrix size and CPU cores
        current_chunk_size = initial_chunk_size
        # Aim to have at least 2 chunks per core for efficient parallelization
        chunks_per_core = 2
        total_chunks_needed = cores_to_use * chunks_per_core
        
        # Adjust chunk size to ensure we can utilize all cores efficiently
        adjusted_chunk_size = min(current_chunk_size, (rows // 2) // total_chunks_needed)
        if adjusted_chunk_size < 1000:  # Ensure minimum chunk size
            adjusted_chunk_size = current_chunk_size
        
        current_chunk_size = adjusted_chunk_size
        chunk_batch = []
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=cores_to_use) as executor:
            # Prepare all chunks in advance to fully utilize all cores
            chunks_to_process = []
            for i in range(0, rows // 2, current_chunk_size):
                # Generate forward and backward chunks
                forward_chunk = generate_chunk(
                    min(current_chunk_size, rows // 2 - i),
                    cols, density, i, direction=1
                )
                
                backward_chunk = generate_chunk(
                    min(current_chunk_size, rows - (i + current_chunk_size)),
                    cols, density, rows - (i + current_chunk_size), direction=-1
                )
                
                # Add both chunks to the processing list
                chunks_to_process.append(forward_chunk)
                chunks_to_process.append(backward_chunk)
                
                # If we have enough chunks to utilize all cores, process them
                if len(chunks_to_process) >= cores_to_use:
                    break
            
            # Process all chunks in parallel
            history_array = np.array([f.magnitude for f in self.metrics_tracker.flow_history]) \
                if self.metrics_tracker.flow_history else np.array([])
            
            print(f"Processing {len(chunks_to_process)} chunks in parallel across {cores_to_use} cores...")
            
            # Submit all chunks for parallel processing
            for chunk in chunks_to_process:
                chunk_args = (chunk, vector, history_array)
                chunk_batch.append(executor.submit(process_chunk, chunk_args))
            
            # Process completed chunks
            for future in as_completed(chunk_batch):
                start_row, end_row, chunk_result, flow = future.result()
                
                # Store results
                result[start_row:end_row] = chunk_result
                
                # Update metrics
                self.metrics_tracker.add_metrics(
                    flow, current_chunk_size,
                    time.time() - start_time,
                    psutil.Process().memory_info().rss / (1024**3)
                )
                
                # Save metrics for visualization
                self.metrics_history.append({
                    'flow_magnitude': flow.magnitude,
                    'direction': flow.direction,
                    'locality': flow.locality,
                    'temporal_pattern': flow.temporal,
                    'significance': flow.significance,
                    'chunk_size': current_chunk_size,
                    'processing_time': time.time() - start_time,
                    'memory_usage': psutil.Process().memory_info().rss / (1024**3)
                })
                
                # Print progress
                self._print_progress(start_row, end_row, rows, flow, {})
        
        # Save final metrics
        self._save_metrics()
        return result
    
    def _print_progress(self, start_row: int, end_row: int, total_rows: int,
                       flow: FlowMetrics, predictions: Dict):
        """Print progress with metrics"""
        chunk_idx = start_row // 10_000 + 1
        print(f"\nCompleted chunk {chunk_idx}")
        print(f"Rows {start_row:,} to {end_row:,} of {total_rows:,}")
        print(f"Flow magnitude: {flow.magnitude:.2f}")
        print(f"Locality: {flow.locality:.3f}")
        print(f"Temporal pattern: {flow.temporal:.3f}")
        print(f"Significance: {flow.significance:.3f}")
        
        if predictions:
            print("\nPredicted metrics:")
            for metric, value in predictions.items():
                print(f"{metric}: {value:.3f}")
        print("-" * 50)
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        metrics_file = self.results_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2) 