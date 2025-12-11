import pytest
import numpy as np
import jax
import jax.numpy as jnp
from nova.data.text_stream import text_to_hypergraph, TurkishTextStream
from nova.models.nova import NovaNet
from transformers import AutoTokenizer

def test_text_to_hypergraph():
    # Test simple sequence
    token_ids = [101, 2000, 2001, 2002, 102]
    window_size = 3
    
    x, H, y = text_to_hypergraph(token_ids, window_size)
    
    # Check shapes
    assert x.shape == (4,) # L-1
    assert y.shape == (4,) # L-1
    assert x[0] == 101
    assert y[0] == 2000
    
    # Check H
    # Nodes: 4
    # Edges: 
    # 1. Sequential: (0,1), (1,2), (2,3) -> 3 edges
    # 2. N-Gram: [0,1,2], [1,2,3] -> 2 edges (window size 3, 4 nodes: 4-3+1 = 2)
    # 3. Self-loops: 4 edges
    # Total edges: 3 + 2 + 4 = 9
    
    n_nodes = 4
    n_edges = 3 + 2 + 4
    assert H.shape == (n_nodes, n_edges)
    
    # Check content
    # First sequential edge: (0, 1)
    assert H[0, 0] == 1
    assert H[1, 0] == 1
    assert H[2, 0] == 0

def test_novanet_embedding():
    model = NovaNet(
        hidden_dim=16,
        num_layers=1,
        out_dim=10,
        vocab_size=100
    )
    
    # Integer input (batch of tokens)
    # NovaNet __call__ expects (n, d) or (n,).
    # If 1D int array, it treats as tokens.
    
    key = jax.random.PRNGKey(0)
    x = jnp.array([1, 5, 20, 3], dtype=jnp.int32)
    H = jnp.ones((4, 5)) # Dummy H
    
    # Init
    params = model.init(key, x, H)
    
    # Forward
    logits, embeddings = model.apply(params, x, H)
    
    assert logits.shape == (4, 10)
    assert embeddings.shape == (4, 16)

# We can't easily test TurkishTextStream without internet/downloading model, 
# but we can mock tokenizer if needed. 
# However, AutoTokenizer will try to download. 
# We'll skip TurkishTextStream test if not online or allow it to fail if model not found.
