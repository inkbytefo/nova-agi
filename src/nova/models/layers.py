## Developer: inkbytefo
## Modified: 2025-12-11

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Optional
from jaxtyping import Float, Array
import chex

from nova.core.ops import hypergraph_conv

class HypergraphLayer(nn.Module):
    """
    Hypergraph Convolutional Layer with learnable weights and optional attention.
    
    Attributes:
        features: Output dimensionality of node features.
        use_attention: Whether to use attention mechanism.
        dropout_rate: Dropout rate (not currently used in basic conv, but good practice).
    """
    features: int
    use_attention: bool = False
    
    @nn.compact
    def __call__(
        self, 
        x: Float[Array, "n d_in"], 
        H: Float[Array, "n m"],
        train: bool = True
    ) -> Float[Array, "n features"]:
        """
        Forward pass of HypergraphLayer.
        
        Args:
            x: Node features (n, d_in).
            H: Incidence matrix (n, m).
            train: Boolean flag for training mode (unused currently but standard).
            
        Returns:
            Updated node features (n, features).
        """
        # 1. Node Transformation (W_node)
        # Transform inputs to hidden dimension
        x_trans = nn.Dense(self.features, name='W_node')(x)
        
        # 2. Edge Transformation Logic preparation
        # We define a callable that applies W_edge + Activation
        # This will be passed to hypergraph_conv's activation argument
        dense_edge = nn.Dense(self.features, name='W_edge')
        
        def edge_update_fn(edge_feats):
            # Apply linear transformation
            e = dense_edge(edge_feats)
            # Apply non-linearity (ReLU)
            return nn.relu(e)

        # 3. Attention Mechanism (Optional)
        attn_weights = None
        if self.use_attention:
            # To compute attention, we need a preliminary aggregation
            # This is a design choice: compute attention based on "raw" aggregation of x_trans
            # Simple approach: e_raw = H^T @ x_trans
            # Then score = W_attn(e_raw)
            # Then softmax over incidence structure?
            # For simplicity and standard HGNN attention:
            # We often learn dynamic edge weights.
            
            # Let's aggregate first to get rough edge profile
            # We use a simple sum or mean for the attention signal
            H_T = jnp.swapaxes(H, -1, -2)
            e_raw = jnp.matmul(H_T, x_trans) # (..., m, features)
            
            # W_attn: Project to scalar score
            # Shape: (..., m, 1)
            attn_scores = nn.Dense(1, name='W_attn')(e_raw)
            
            # Apply Sigmoid to get a gating factor between 0 and 1
            # Or we can do something more complex. 
            # The prompt says "project edge features to a scalar and apply softmax (or sigmoid)"
            # Since these are weights for the incidence matrix, they should probably be per-edge scalars.
            # We broadcast these scores back to the H structure?
            # Or is this attention *within* the aggregation (node-to-edge)?
            # "Attention Mechanism for hyperedges" usually means weighting hyperedges.
            # We will assume it re-weights the columns of H.
            
            edge_gates = nn.sigmoid(attn_scores) # (..., m, 1)
            
            # We modulate H by these gates. 
            # H_attn = H * gates^T (broadcasting)
            # H is (n, m), gates is (m, 1). We need (1, m)
            # Use swapaxes for batch support
            edge_gates_T = jnp.swapaxes(edge_gates, -1, -2) # (..., 1, m)
            attn_weights = H * edge_gates_T
            
        # 4. Core Convolution
        # Pass the transformed x and the edge update function
        out = hypergraph_conv(
            x=x_trans,
            H=H,
            activation=edge_update_fn,
            use_attention=self.use_attention,
            attention_weights=attn_weights
        )
        
        # 5. Layer Norm & Residual (Residual handled in NovaNet usually, but prompt mentions it here?)
        # "Apply nn.LayerNorm at the end for stability."
        out = nn.LayerNorm()(out)
        
        return out
