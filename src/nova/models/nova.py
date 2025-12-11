## Developer: inkbytefo
## Modified: 2025-12-11

from typing import Sequence, Union
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Float, Array, Int
import chex

from nova.models.layers import HypergraphLayer

class NovaNet(nn.Module):
    """
    NovaNet: Deep Hypergraph Neural Network with Residual Connections.
    
    Structure:
      - Input Projection (Embedding or Dense)
      - Stack of Hypergraph Layers (Residual)
      - Output Projection
    """
    hidden_dim: int
    num_layers: int
    out_dim: int
    use_attention: bool = False
    dropout_rate: float = 0.0
    vocab_size: int = 32000 # Default for BERT-like tokenizers
    embedding_dim: int = 512 # Dimension of token embeddings
    
    @nn.compact
    def __call__(
        self, 
        x: Union[Float[Array, "n d_in"], Int[Array, "n"]], 
        H: Float[Array, "n m"],
        train: bool = True
    ) -> tuple[Float[Array, "n out_dim"], Float[Array, "n hidden_dim"]]:
        """
        Forward pass.
        
        Args:
            x: Input node features (float) or token IDs (int).
            H: Incidence matrix.
            train: Training mode flag.
            
        Returns:
            logits: Output predictions (n, out_dim).
            embeddings: Final node embeddings before classifier (n, hidden_dim).
        """
        # 1. Input Projection
        if jnp.issubdtype(x.dtype, jnp.integer):
            # If input is token IDs, use Embedding layer
            x = nn.Embed(num_embeddings=self.vocab_size, features=self.embedding_dim, name='input_embed')(x)
        else:
            # Project raw features to hidden dimension
            x = nn.Dense(self.hidden_dim, name='input_proj')(x)
            x = nn.relu(x)
        
        # Ensure dimension matches hidden_dim for residual connections
        if x.shape[-1] != self.hidden_dim:
            x = nn.Dense(self.hidden_dim, name='dim_proj')(x)
        
        # 2. Stack of Hypergraph Layers with Residuals
        for i in range(self.num_layers):
            residual = x
            
            # Layer
            x = HypergraphLayer(
                features=self.hidden_dim,
                use_attention=self.use_attention,
                name=f'layer_{i}'
            )(x, H, train=train)
            
            # Dropout
            if train and self.dropout_rate > 0:
                x = nn.Dropout(self.dropout_rate, deterministic=False)(x)
            
            # Residual connection
            x = x + residual
            
        # Store embeddings for loss calculation
        embeddings = x
        
        # 3. Output Projection
        logits = nn.Dense(self.out_dim, name='output_proj')(embeddings)
        
        return logits, embeddings
