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
    # Relax rank check to allow batches (Rank 3)
    # chex.assert_rank(H, 2)
    # chex.assert_rank(embeddings, 2)
    
    # Normalize embeddings
    # embeddings shape: (..., n, d)
    # Norm along the last dimension (d)
    norm_embeddings = embeddings / (jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-7)
    
    # Compute similarity with the AVERAGE embedding (Global Context)
    # Average over nodes (axis=-2)
    # avg_emb shape: (..., d)
    avg_emb = jnp.mean(norm_embeddings, axis=-2)
    avg_emb = avg_emb / (jnp.linalg.norm(avg_emb, axis=-1, keepdims=True) + 1e-7)
    
    # Compute similarity
    # norm_embeddings: (..., n, d)
    # avg_emb: (..., d)
    # We want dot product along d, resulting in (..., n)
    # Expand avg_emb to (..., 1, d) for broadcasting
    sim_to_global = jnp.sum(norm_embeddings * jnp.expand_dims(avg_emb, axis=-2), axis=-1)
    
    # Thresholding
    new_global_col = (sim_to_global > threshold).astype(H.dtype)
    
    # Update only the last column of H
    # H is (..., n, m). Last column is H[..., -1]
    H_new = H.at[..., -1].set(new_global_col)
    
    return H_new
