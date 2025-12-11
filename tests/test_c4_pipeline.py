
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from nova.data.text_stream import TurkishTextStream, text_to_hypergraph
from nova.models.nova import NovaNet

def test_text_to_hypergraph():
    # Test with dummy data
    token_ids = [101, 200, 300, 400, 102] # 5 tokens
    max_seq_len = 5
    
    x, H, y = text_to_hypergraph(token_ids, max_seq_len)
    
    # Expected shapes
    # x: input [101, 200, 300, 400] (len 4)
    # y: target [200, 300, 400, 102] (len 4)
    # n_nodes = 4
    
    assert x.shape == (4,)
    assert y.shape == (4,)
    assert x[0] == 101
    assert y[0] == 200
    
    # Edges:
    # 1. Sequential: (0,1), (1,2), (2,3) -> 3 edges
    # 2. Context: (0,1,2), (1,2,3) -> 2 edges
    # 3. Global: (0,1,2,3) -> 1 edge
    # Total edges = 3 + 2 + 1 = 6
    
    assert H.shape == (4, 6)
    
    # Check Sequential edge 0: (0, 1)
    assert H[0, 0] == 1.0
    assert H[1, 0] == 1.0
    assert H[2, 0] == 0.0
    
    # Check Global edge (last one)
    assert np.all(H[:, -1] == 1.0)

def test_model_forward_with_tokens():
    # Create model
    model = NovaNet(
        hidden_dim=64,
        num_layers=2,
        out_dim=1000, # small vocab for test
        vocab_size=1000,
        embedding_dim=32
    )
    
    # Create dummy input
    # Batch size 1, sequence length 10
    x = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
    # H matrix for 5 nodes
    # Let's just make a dummy H
    H = jnp.ones((5, 3), dtype=jnp.float32)
    
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, x, H)
    
    logits, embeddings = model.apply(params, x, H)
    
    assert logits.shape == (5, 1000)
    assert embeddings.shape == (5, 64)

if __name__ == "__main__":
    test_text_to_hypergraph()
    test_model_forward_with_tokens()
    print("Tests passed!")
