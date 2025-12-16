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

def causal_hypergraph_conv(
    x: Float[Array, "... n d"],
    H_in: Float[Array, "... n m"],
    H_out: Float[Array, "... n m"],
    activation: Callable[[Array], Array] = jax.nn.relu,
    use_global: bool = True,
    eps: float = 1e-7
) -> Float[Array, "... n d"]:
    """
    Performs a Causal Hypergraph Convolution.
    Ensures that node i only receives information from nodes j <= i.
    
    Mechanism:
    1. Local Edges: Defined by (H_in, H_out).
       - H_in (Gather): Nodes -> Edge.
       - H_out (Scatter): Edge -> Nodes.
       - Causality is enforced by construction: If Edge e scatters to Node i (H_out[i,e]=1),
         then Edge e must only gather from Nodes j <= i (H_in[j,e]=1).
         
    2. Global Context: Implemented via Cumulative Sum (Scan), avoiding explicit O(N^2) edges.
    
    Args:
        x: Node features (..., n, d).
        H_in: Gather incidence matrix (..., n, m).
        H_out: Scatter incidence matrix (..., n, m).
        use_global: Whether to add causal global context (cumsum).
        
    Returns:
        Updated node features.
    """
    # 1. Node-to-Edge (Gather)
    # E = H_in^T * X
    # Normalize? Standard GCN uses degree normalization.
    # D_e = sum(H_in, axis=0) -> Edge degree (number of nodes in edge)
    edge_features = jnp.matmul(jnp.swapaxes(H_in, -1, -2), x) # (..., m, d)
    
    # Edge Normalization (Optional but recommended)
    # edge_deg = jnp.sum(H_in, axis=-2, keepdims=True) # (..., 1, m)
    # edge_features = edge_features / (jnp.swapaxes(edge_deg, -1, -2) + eps)
    
    # Activation
    edge_features = activation(edge_features)
    
    # 2. Edge-to-Node (Scatter)
    # X_local = H_out * E
    x_local = jnp.matmul(H_out, edge_features) # (..., n, d)
    
    # Node Normalization?
    # node_deg = jnp.sum(H_out, axis=-1, keepdims=True) # (..., n, 1)
    # x_local = x_local / (node_deg + eps)
    
    # 3. Global Causal Context
    if use_global:
        # Cumulative Sum along sequence dimension (n)
        # Assumes x is (batch, n, d) or (n, d)
        # axis -2 is n.
        global_ctx = jnp.cumsum(x, axis=-2)
        
        # Cumulative Mean
        # counts = 1, 2, ..., n
        n = x.shape[-2]
        counts = jnp.arange(1, n + 1, dtype=x.dtype)
        # Reshape counts to broadcast: (n, 1)
        shape = [1] * (x.ndim - 2) + [n, 1]
        counts = counts.reshape(shape)
        
        global_ctx = global_ctx / counts
        
        # Combine
        # Using a simple addition or gated mechanism?
        # Simple addition for now.
        x_out = x_local + global_ctx
    else:
        x_out = x_local
        
    return x_out
