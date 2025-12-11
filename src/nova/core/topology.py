## Developer: inkbytefo
## Modified: 2025-12-11

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Int
import chex

def update_topology(
    H: Float[Array, "n m"],
    embeddings: Float[Array, "n d"],
    k: int = 5,
    threshold: float = 0.5
) -> Float[Array, "n m"]:
    """
    Updates hypergraph topology by rewiring edges based on node similarity.
    
    Logic:
    1. Compute Edge Centroids: Mean embedding of nodes currently in each edge.
    2. Compute Similarity: Cosine similarity between Edge Centroids and All Nodes.
    3. Top-k Rewiring: For each edge, connect the k nodes with highest similarity 
       (if similarity > threshold).
       
    This approach allows edges to drift towards clusters of similar nodes (communities).
    
    Args:
        H: Current incidence matrix (n, m).
        embeddings: Node embeddings (n, d).
        k: Number of nodes to connect per edge (target size).
        threshold: Minimum similarity threshold to create a connection.
        
    Returns:
        H_new: Updated incidence matrix (n, m).
    """
    chex.assert_rank(H, 2)
    chex.assert_rank(embeddings, 2)
    
    n, m = H.shape
    
    # 1. Normalize Embeddings for Cosine Similarity
    # (n, d)
    norm_embeddings = embeddings / (jnp.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-7)
    
    # 2. Compute Edge Centroids
    # Weighted sum of embeddings for each edge? Or just binary mean?
    # H is binary-ish.
    # edge_sum: (m, d) = H.T @ norm_embeddings
    edge_sum = jnp.matmul(H.T, norm_embeddings)
    
    # edge_counts: (m, 1) = H.T @ ones
    edge_counts = jnp.sum(H, axis=0, keepdims=True).T
    
    # centroids: (m, d)
    centroids = edge_sum / (edge_counts + 1e-7)
    
    # Normalize centroids for cosine sim
    norm_centroids = centroids / (jnp.linalg.norm(centroids, axis=1, keepdims=True) + 1e-7)
    
    # 3. Compute Similarity (Edge-to-Node)
    # Sim: (m, n) = Centroids @ Nodes.T
    sim_matrix = jnp.matmul(norm_centroids, norm_embeddings.T)
    
    # 4. Top-k Selection
    # For each row (edge), find indices of top k nodes.
    # We want a binary matrix H_new where top-k entries are 1.
    
    # Use jax.lax.top_k
    top_k_vals, top_k_indices = jax.lax.top_k(sim_matrix, k)
    
    # Create mask for threshold
    # mask: (m, k)
    threshold_mask = top_k_vals > threshold
    
    # Scatter back to (m, n) matrix
    # Initialize zeros
    H_new_T = jnp.zeros((m, n))
    
    # Create row indices: [[0,0,0...], [1,1,1...], ...]
    row_indices = jnp.arange(m)[:, None] # (m, 1)
    row_indices = jnp.broadcast_to(row_indices, (m, k))
    
    # Set values to 1 (where mask is True)
    # We need to handle the threshold. 
    # Values to set: 1.0 * threshold_mask
    values = 1.0 * threshold_mask
    
    # Update H_new_T at (row_indices, top_k_indices)
    H_new_T = H_new_T.at[row_indices, top_k_indices].set(values)
    
    # Transpose back to (n, m)
    H_new = H_new_T.T
    
    return H_new
