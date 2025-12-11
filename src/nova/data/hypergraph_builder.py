## Developer: inkbytefo
## Modified: 2025-12-11

import jax.numpy as jnp
import numpy as np

def build_incremental_H(
    n_nodes: int, 
    seq_fixed: int = 8,
    ctx_fixed: int = 4
) -> np.ndarray:
    """
    Builds a hypergraph incidence matrix H with incremental/stable structure.
    
    Structure:
    1. Sequential edges: [i, i+1] for the last `seq_fixed` tokens.
    2. Context edges: [i, i+1, i+2] for the last `ctx_fixed` tokens.
    3. Global edge: One edge connecting all nodes (if n_nodes > 1).
    
    Args:
        n_nodes: Number of nodes (tokens).
        seq_fixed: Number of recent sequential edges to keep.
        ctx_fixed: Number of recent context edges to keep.
        
    Returns:
        H: Incidence matrix of shape (n_nodes, m_edges).
    """
    edges = []
    
    # 1. Sequential Edges (Window 2)
    start_seq = max(0, n_nodes - seq_fixed - 1)
    for i in range(start_seq, n_nodes - 1):
        edges.append([i, i+1])
        
    # 2. Context Edges (Window 3)
    start_ctx = max(0, n_nodes - ctx_fixed - 2)
    for i in range(start_ctx, n_nodes - 2):
        edges.append([i, i+1, i+2])
        
    # 3. Global Edge
    if n_nodes > 1:
        edges.append(list(range(n_nodes)))
        
    # Build H matrix
    m_edges = len(edges)
    if m_edges == 0:
        return np.zeros((n_nodes, 0), dtype=np.float32)
        
    H = np.zeros((n_nodes, m_edges), dtype=np.float32)
    for j, edge_nodes in enumerate(edges):
        H[edge_nodes, j] = 1.0
        
    return H
