## Developer: inkbytefo
## Modified: 2025-12-11

import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from nova.core.ops import hypergraph_conv

def test_hypergraph_conv_shape():
    """Tests that hypergraph_conv returns the correct shape."""
    key = jax.random.PRNGKey(0)
    
    # 5 nodes, 3 edges, 4 features
    n, m, d = 5, 3, 4
    
    x = jax.random.normal(key, (n, d))
    # Random binary incidence matrix
    H = jax.random.bernoulli(key, 0.5, (n, m)).astype(jnp.float32)
    
    out = hypergraph_conv(x, H)
    
    assert out.shape == (n, d)

def test_hypergraph_conv_values():
    """Tests basic value properties (e.g. no NaNs)."""
    key = jax.random.PRNGKey(1)
    n, m, d = 5, 3, 4
    x = jax.random.normal(key, (n, d))
    H = jnp.ones((n, m)) # Fully connected
    
    out = hypergraph_conv(x, H)
    
    assert not jnp.isnan(out).any()
    assert not jnp.isinf(out).any()

def test_hypergraph_conv_attention():
    """Tests hypergraph_conv with attention weights."""
    key = jax.random.PRNGKey(2)
    n, m, d = 5, 3, 4
    x = jax.random.normal(key, (n, d))
    H = jnp.ones((n, m))
    att = jax.random.uniform(key, (n, m))
    
    out = hypergraph_conv(x, H, use_attention=True, attention_weights=att)
    
    assert out.shape == (n, d)
    assert not jnp.isnan(out).any()

if __name__ == "__main__":
    # Manually run if executed directly
    test_hypergraph_conv_shape()
    test_hypergraph_conv_values()
    test_hypergraph_conv_attention()
    print("All tests passed!")
