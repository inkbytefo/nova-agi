## Developer: inkbytefo
## Modified: 2025-12-11

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Optional
from jaxtyping import Float, Array
import chex

from nova.core.ops import causal_hypergraph_conv

class HypergraphLayer(nn.Module):
    """
    Hypergraph Convolutional Layer with Causal Masking.
    
    Attributes:
        features: Output dimensionality of node features.
        use_global: Whether to use global causal context.
    """
    features: int
    use_global: bool = True

    @nn.compact
    def __call__(
        self, 
        x: Float[Array, "n d_in"], 
        H_in: Float[Array, "n m"],
        H_out: Float[Array, "n m"],
        train: bool = True
    ) -> Float[Array, "n features"]:
        """
        Forward pass of HypergraphLayer.
        
        Args:
            x: Node features (n, d_in).
            H_in: Gather incidence matrix (n, m).
            H_out: Scatter incidence matrix (n, m).
            train: Boolean flag for training mode.
            
        Returns:
            Updated node features (n, features).
        """
        # 1. Node Transformation (W_node)
        # Transform inputs to hidden dimension
        x_trans = nn.Dense(self.features, name='W_node')(x)
        
        # 2. Edge Transformation Logic preparation
        dense_edge = nn.Dense(self.features, name='W_edge')

        def edge_update_fn(edge_feats):
            # Apply linear transformation
            e = dense_edge(edge_feats)
            # Apply non-linearity (ReLU)
            return nn.relu(e)

        # 3. Core Causal Convolution
        out = causal_hypergraph_conv(
            x=x_trans,
            H_in=H_in,
            H_out=H_out,
            activation=edge_update_fn,
            use_global=self.use_global
        )
        
        # 4. Layer Norm
        out = nn.LayerNorm()(out)
        
        return out
