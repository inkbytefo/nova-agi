## Developer: inkbytefo
## Modified: 2025-12-11

import jax.numpy as jnp
import numpy as np
from typing import Tuple

def build_causal_H(
    n_nodes: int, 
    max_edges: int = 4096,
    window_sizes: list = [2, 3, 5],
    add_long_range: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds Causal Hypergraph Incidence Matrices (H_in, H_out) for the entire sequence.
    Ensures that for any edge e:
      - It gathers from nodes {t-k, ..., t} (via H_in)
      - It scatters to node t (via H_out)
    This guarantees no future leakage.
    
    Args:
        n_nodes: Sequence length.
        max_edges: Fixed size for second dimension (padding).
        window_sizes: List of n-gram sizes (e.g., [2, 3] for bigrams/trigrams).
        add_long_range: Whether to add power-of-2 skip connections (1, 2, 4, 8...).
        
    Returns:
        H_in: Gather matrix (n_nodes, max_edges).
        H_out: Scatter matrix (n_nodes, max_edges).
    """
    H_in = np.zeros((n_nodes, max_edges), dtype=np.float32)
    H_out = np.zeros((n_nodes, max_edges), dtype=np.float32)
    
    edge_idx = 0
    
    # 1. Self-loops (Unigram context) - Important for preserving identity
    # Often implicitly handled by residual, but explicit edges help.
    # We skip explicit self-loops if window_sizes doesn't include 1, 
    # as residuals handle it.
    
    # 2. Sliding Window Edges (N-grams)
    for w in window_sizes:
        # For a window of size w, the first valid edge ends at index w-1
        for t in range(w - 1, n_nodes):
            if edge_idx >= max_edges:
                break
                
            # Define edge span: [t - w + 1, ..., t]
            # Gather from these nodes
            for k in range(w):
                H_in[t - k, edge_idx] = 1.0
                
            # Scatter to the last node (t)
            H_out[t, edge_idx] = 1.0
            
            edge_idx += 1

    # 3. Long-Range Edges (Power of 2 Dilations)
    # Connects t to t - 2^k
    if add_long_range:
        for t in range(n_nodes):
            k = 1
            while True:
                prev_t = t - k
                if prev_t < 0:
                    break
                
                if edge_idx >= max_edges:
                    break

                # Create an edge that gathers from prev_t and scatters to t
                # This allows direct information flow from past
                H_in[prev_t, edge_idx] = 1.0
                H_out[t, edge_idx] = 1.0
                
                edge_idx += 1
                k *= 2
            
            if edge_idx >= max_edges:
                break
            
    # Note: Global edges are handled dynamically in ops.py via cumsum,
    # so we don't add them here to save memory/compute.
    
    return H_in, H_out

# Legacy alias for compatibility if needed, but discouraged
def build_incremental_H(n_nodes, seq_fixed=8, ctx_fixed=4):
    # This was the old leaky implementation. 
    # We redirect to the safe one, but it returns 2 matrices now.
    # Raising error to force update.
    raise DeprecationWarning("build_incremental_H is deprecated. Use build_causal_H.")
