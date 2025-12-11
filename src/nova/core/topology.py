## Developer: inkbytefo
## Modified: 2025-12-11

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Int
import chex

def update_topology(
    H: Float[Array, "n m"],
    embeddings: Float[Array, "n d"],
    k: int = 5, # Unused in simplified version
    threshold: float = 0.6
) -> Float[Array, "n m"]:
    """
    Refines the hypergraph topology.
    
    Simplified Logic:
    - Only refines the Global Edge (assumed to be the last column).
    - Connects nodes to the global edge if they are similar to the current (last) node.
    - O(n^2) complexity due to similarity calculation, but much faster than full rewiring.
    
    Args:
        H: Current incidence matrix (n, m).
        embeddings: Node embeddings (n, d).
        k: Unused (kept for API compatibility).
        threshold: Cosine similarity threshold for connection.
        
    Returns:
        H_new: Updated incidence matrix.
    """
    chex.assert_rank(H, 2)
    chex.assert_rank(embeddings, 2)
    
    # Normalize embeddings
    norm_embeddings = embeddings / (jnp.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-7)
    
    # Compute similarity with the LAST node only (current context)
    # We don't need full n*n matrix if we only care about the last column.
    # sim[i] = dot(emb[i], emb[-1])
    # Shape: (n,)
    last_emb = norm_embeddings[-1]
    sim_to_last = jnp.dot(norm_embeddings, last_emb)
    
    # Thresholding
    # 1.0 if sim > threshold, else 0.0
    new_global_col = (sim_to_last > threshold).astype(H.dtype)
    
    # Update only the last column of H
    # H is (n, m). Last column is H[:, -1]
    H_new = H.at[:, -1].set(new_global_col)
    
    return H_new
