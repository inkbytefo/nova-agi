## Developer: inkbytefo
## Modified: 2025-12-11

import jax.numpy as jnp
import numpy as np

def build_incremental_H(n_nodes: int, fixed_edges_per_type: int = 8) -> np.ndarray:
    """
    Builds a hypergraph incidence matrix H with incremental/stable structure.
    
    Structure:
    1. Sequential edges: [i, i+1] for the last `fixed_edges_per_type` tokens.
    2. Context edges: [i, i+1, i+2] for the last `fixed_edges_per_type` tokens.
    3. Global edge: One edge connecting all nodes (if n_nodes > 1).
    
    Args:
        n_nodes: Number of nodes (tokens).
        fixed_edges_per_type: Number of recent edges to keep for seq/context types.
        
    Returns:
        H: Incidence matrix of shape (n_nodes, m_edges).
    """
    # Use lists for construction, convert to array at the end
    edges = []
    
    # 1. Sequential Edges (Window 2)
    # Range: last N edges, so we start from max(0, n_nodes - fixed_edges_per_type - 1)
    # But wait, user code: range(max(0, n_nodes-fixed_edges_per_type), n_nodes-1)
    # If n=10, fixed=8. range(2, 9). i=2..8. 
    # Edges: [2,3], [3,4] ... [8,9]. Total 7 edges.
    start_seq = max(0, n_nodes - fixed_edges_per_type)
    for i in range(start_seq, n_nodes - 1):
        edges.append([i, i+1])
        
    # 2. Context Edges (Window 3)
    # User code: range(max(0, n_nodes-fixed_edges_per_type), n_nodes-2)
    start_ctx = max(0, n_nodes - fixed_edges_per_type)
    for i in range(start_ctx, n_nodes - 2):
        edges.append([i, i+1, i+2])
        
    # 3. Global Edge
    if n_nodes > 1:
        edges.append(list(range(n_nodes)))
        
    # Build H matrix
    m_edges = len(edges)
    if m_edges == 0:
        # Edge case: 0 or 1 node might have 0 edges if n_nodes < 2 for global and < 2 for seq
        # If n=1, edges=[], returns (1,0) matrix?
        return np.zeros((n_nodes, 0), dtype=np.float32)
        
    H = np.zeros((n_nodes, m_edges), dtype=np.float32)
    for j, edge_nodes in enumerate(edges):
        H[edge_nodes, j] = 1.0
        
    return H
