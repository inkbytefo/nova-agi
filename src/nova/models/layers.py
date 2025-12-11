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
            H_T = jnp.swapaxes(H, -1, -2)
            e_raw = jnp.matmul(H_T, x_trans)
            q = nn.Dense(self.features, name='attn_q')(x_trans)
            k = nn.Dense(self.features, name='attn_k')(e_raw)
            scale = jnp.sqrt(jnp.array(self.features, dtype=x_trans.dtype))
            attn_logits = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / (scale + 1e-7)
            
            mask = (H > 0).astype(x_trans.dtype)
            neg_inf = -1e9 # Safe negative infinity equivalent for logits
            
            # Fix: Use jnp.where to avoid NaN from 0 * -inf
            masked_logits = jnp.where(mask > 0, attn_logits, neg_inf)
            
            # Softmax
            exp_logits = jnp.exp(masked_logits - jnp.max(masked_logits, axis=-2, keepdims=True)) * mask
            denom = jnp.sum(exp_logits, axis=-2, keepdims=True) + 1e-7
            attn_weights = exp_logits / denom
            
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
