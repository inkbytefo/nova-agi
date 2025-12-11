## Developer: inkbytefo
## Modified: 2025-12-11

import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from nova.models.nova import NovaNet
from nova.core.loss import thermodynamic_loss

def test_novanet_forward():
    """Tests the forward pass of NovaNet."""
    key = jax.random.PRNGKey(0)
    key1, key2, key3 = jax.random.split(key, 3)
    
    n_nodes = 10
    n_edges = 5
    d_in = 8
    hidden_dim = 16
    out_dim = 2
    
    x = jax.random.normal(key1, (n_nodes, d_in))
    H = jax.random.bernoulli(key2, 0.5, (n_nodes, n_edges)).astype(jnp.float32)
    
    model = NovaNet(hidden_dim=hidden_dim, num_layers=2, out_dim=out_dim)
    params = model.init(key3, x, H)
    
    logits, embeddings = model.apply(params, x, H)
    
    assert logits.shape == (n_nodes, out_dim)
    assert embeddings.shape == (n_nodes, hidden_dim)
    print("NovaNet forward pass: OK")

def test_thermodynamic_loss():
    """Tests the thermodynamic loss calculation."""
    key = jax.random.PRNGKey(1)
    n_nodes = 10
    out_dim = 2
    hidden_dim = 16
    
    # Dummy data
    logits = jax.random.normal(key, (n_nodes, out_dim))
    targets = jax.random.normal(key, (n_nodes, out_dim))
    embeddings = jax.random.normal(key, (n_nodes, hidden_dim))
    
    # Dummy params (just a dict of arrays)
    params = {'w': jnp.ones((5, 5))}
    
    loss, metrics = thermodynamic_loss(params, logits, targets, embeddings)
    
    assert isinstance(loss, jnp.ndarray) or isinstance(loss, float)
    assert not jnp.isnan(loss)
    assert "loss_task" in metrics
    assert "loss_energy" in metrics
    assert "loss_diversity" in metrics
    print("Thermodynamic loss: OK")

if __name__ == "__main__":
    test_novanet_forward()
    test_thermodynamic_loss()
