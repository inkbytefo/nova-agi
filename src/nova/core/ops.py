## Developer: inkbytefo
## Modified: 2025-12-11

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
import chex
from typing import Optional, Callable

def compute_degrees(H: Float[Array, "... n m"]) -> tuple[Float[Array, "... n"], Float[Array, "... m"]]:
    """
    Computes node and edge degrees from the incidence matrix.
    
    Args:
        H: Incidence matrix of shape (..., num_nodes, num_edges).
        
    Returns:
        d_v: Node degrees (..., num_nodes).
        d_e: Edge degrees (..., num_edges).
    """
    d_v = jnp.sum(H, axis=-1)
    d_e = jnp.sum(H, axis=-2)
    return d_v, d_e

def hypergraph_conv(
    x: Float[Array, "... n d"],
    H: Float[Array, "... n m"],
    activation: Callable[[Array], Array] = jax.nn.relu,
    use_attention: bool = False,
    attention_weights: Optional[Float[Array, "... n m"]] = None,
    eps: float = 1e-7
) -> Float[Array, "... n d"]:
    """
    Performs a Hypergraph Convolution operation.
    
    Implements the core message passing logic:
    1. Node-to-Edge: Gather node features to hyperedges.
    2. Edge-Update: Apply non-linearity and optional attention.
    3. Edge-to-Node: Scatter updated edge features back to nodes.
    
    Includes symmetric degree normalization: D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2} logic,
    decomposed into steps for message passing.

    Args:
        x: Node features of shape (..., num_nodes, feature_dim).
        H: Incidence matrix of shape (..., num_nodes, num_edges). 
           Entries H[i, j] = 1 if node i is in edge j, else 0.
        activation: Non-linear activation function applied to edge features.
        use_attention: If True, uses attention_weights to modulate message passing.
        attention_weights: Optional attention matrix of shape (..., num_nodes, num_edges).
                           If provided, replaces binary H for aggregation.
        eps: Small epsilon for numerical stability in division.

    Returns:
        Updated node features of shape (..., num_nodes, feature_dim).
    """
    # Relaxed rank checks to allow batching
    # chex.assert_rank(x, 2)
    # chex.assert_rank(H, 2)
    
    # Get shapes dynamically
    n = H.shape[-2]
    m = H.shape[-1]
    
    # 1. Degree Normalization
    # Compute degrees based on the structural H (binary/weighted)
    d_v, d_e = compute_degrees(H)
    
    # Inverse square root for Node degrees: D_v^{-1/2}
    d_v_inv_sqrt = jnp.power(jnp.maximum(d_v, eps), -0.5)
    # Inverse for Edge degrees: D_e^{-1}
    d_e_inv = jnp.power(jnp.maximum(d_e, eps), -1.0)
    
    # Create diagonal matrices (as vectors for broadcasting)
    # In JAX, we can just multiply columns/rows.
    
    # 2. Node-to-Edge (Gather)
    # Step 2a: Apply D_v^{-1/2} to X
    # x_norm = D_v^{-1/2} * X
    # Handle broadcasting for batch dimensions
    x_norm = x * d_v_inv_sqrt[..., None]
    
    # Step 2b: Aggregate to edges
    # If using attention, we might use a different H here, but usually 
    # attention implies learning the weights in H. 
    # If attention_weights is provided, use it.
    W = attention_weights if (use_attention and attention_weights is not None) else H
    
    # edge_features = W.T @ x_norm
    # Use swapaxes for correct transpose with batches
    W_T = jnp.swapaxes(W, -1, -2)
    edge_features = jnp.matmul(W_T, x_norm)
    
    # Step 2c: Apply D_e^{-1} to edge features (Average aggregation equivalent)
    # edge_features = D_e^{-1} * edge_features
    edge_features = edge_features * d_e_inv[..., None]

    # 3. Edge-Update
    # Apply non-linearity
    edge_features_updated = activation(edge_features)
    
    # 4. Edge-to-Node (Scatter)
    # Step 4a: Aggregate back to nodes using H (or W)
    # out = W @ edge_features_updated
    # Shape: (n, m) @ (m, d) -> (n, d)
    out = jnp.matmul(W, edge_features_updated)
    
    # Step 4b: Apply D_v^{-1/2} again (Symmetric norm)
    out = out * d_v_inv_sqrt[..., None]
    
    return out
