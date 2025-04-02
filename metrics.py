"""Metrics module for IDA algorithm"""

import numpy as np
from numba import jit
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

@dataclass
class FlowMetrics:
    """Compact representation of flow metrics"""
    magnitude: float
    direction: int
    locality: float
    temporal: float
    significance: float

class MetricsTracker:
    """Efficient metrics tracking with fixed-size history"""
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.flow_history = deque(maxlen=history_size)
        self.chunk_sizes = deque(maxlen=history_size)
        self.processing_times = deque(maxlen=history_size)
        self.memory_usages = deque(maxlen=history_size)
    
    def add_metrics(self, flow: FlowMetrics, chunk_size: int,
                   processing_time: float, memory_usage: float):
        """Add new metrics to history"""
        self.flow_history.append(flow)
        self.chunk_sizes.append(chunk_size)
        self.processing_times.append(processing_time)
        self.memory_usages.append(memory_usage)
    
    def get_recent_metrics(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get recent metrics for prediction"""
        if len(self.flow_history) < 2:
            return None, None
        
        # Features: flow metrics
        X = np.array([
            [f.magnitude, f.direction, f.locality, f.temporal, f.significance]
            for f in self.flow_history
        ])
        
        # Targets: performance metrics
        y = np.array([
            [s, t, m]
            for s, t, m in zip(
                self.chunk_sizes,
                self.processing_times,
                self.memory_usages
            )
        ])
        
        return X, y

@jit(nopython=True)
def calculate_flow(data: np.ndarray, indices: np.ndarray,
                  history: np.ndarray, chunk_size: int) -> Tuple[float, int, float, float, float]:
    """Calculate flow metrics for a chunk"""
    # Calculate flow magnitude
    magnitude = np.sum(np.abs(data))
    if len(data) == 0:
        magnitude = 0.0
    
    # Calculate flow direction (1 for forward, -1 for backward)
    direction = 1 if np.mean(indices) > chunk_size / 2 else -1
    
    # Calculate locality (spatial concentration of non-zero elements)
    if len(indices) > 0:
        locality = np.std(indices) / max(chunk_size, 1)
    else:
        locality = 0.0
    
    # Calculate temporal pattern (correlation with history)
    if len(history) > 0:
        diffs = np.diff(history)
        if len(diffs) > 0:
            temporal = np.mean(diffs)
        else:
            temporal = 0.0
    else:
        temporal = 0.0
    
    # Calculate significance
    mean_val = np.mean(np.abs(data)) if len(data) > 0 else 0.0
    if mean_val < 1e-10:
        significance = 0.0
    else:
        significance = np.sum(np.abs(data) > mean_val) / max(len(data), 1)
    
    return magnitude, direction, locality, temporal, significance 