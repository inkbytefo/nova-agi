
import jax
import jax.numpy as jnp
import numpy as np
from nova.core.ops import causal_hypergraph_conv
from nova.data.hypergraph_builder import build_causal_H
# from nova.models.nova import NovaNet

def verify_causal_H():
    print("Verifying build_causal_H...")
    n_nodes = 10
    max_edges = 20
    H_in, H_out = build_causal_H(n_nodes, max_edges=max_edges)
    
    # Check shapes
    assert H_in.shape == (n_nodes, max_edges), f"H_in shape mismatch: {H_in.shape}"
    assert H_out.shape == (n_nodes, max_edges), f"H_out shape mismatch: {H_out.shape}"
    
    # Check Causality
    # For any edge e (column), if H_out[t, e] == 1 (scatters to t), 
    # then H_in[k, e] must be 0 for all k > t.
    
    violations = 0
    for e in range(max_edges):
        targets = np.where(H_out[:, e] > 0)[0]
        sources = np.where(H_in[:, e] > 0)[0]
        
        if len(targets) == 0: continue
        
        # In this specific construction, each edge targets exactly one node t
        # And gathers from t, t-1, ...
        t = targets[0] 
        
        for s in sources:
            if s > t:
                print(f"Violation at edge {e}: Source {s} > Target {t}")
                violations += 1
                
    if violations == 0:
        print("PASS: build_causal_H preserves causality.")
    else:
        print(f"FAIL: {violations} causality violations found in build_causal_H.")
        exit(1)

def verify_causal_conv():
    print("\nVerifying causal_hypergraph_conv...")
    n = 5
    d = 4
    m = 10
    
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n, d))
    
    # Create a causal H structure manually to test convolution
    # Edge 0: gathers 0, scatters 0
    # Edge 1: gathers 0,1, scatters 1
    # ...
    H_in = jnp.zeros((n, m))
    H_out = jnp.zeros((n, m))
    
    for i in range(min(n, m)):
        H_in = H_in.at[0:i+1, i].set(1.0)
        H_out = H_out.at[i, i].set(1.0)
        
    # Check if changing x[t] affects output[t-1] (Should NOT happen)
    
    # 1. Forward pass with original x
    y1 = causal_hypergraph_conv(x, H_in, H_out, use_global=True)
    
    # 2. Perturb x[4] (last element)
    x_perturbed = x.at[4, :].add(10.0)
    y2 = causal_hypergraph_conv(x_perturbed, H_in, H_out, use_global=True)
    
    # Check y[0..3]
    diff = jnp.abs(y1[:4] - y2[:4]).sum()
    print(f"Difference in past outputs after perturbing future input: {diff}")
    
    if diff < 1e-5:
        print("PASS: causal_hypergraph_conv is causal (future did not affect past).")
    else:
        print("FAIL: causal_hypergraph_conv leaked future information.")
        exit(1)

def verify_model_jit():
    print("\nVerifying NovaNet JIT compilation with static shapes...")
    model = NovaNet(hidden_dim=32, num_layers=2, out_dim=10)
    
    # Inputs with fixed shapes
    n_nodes = 20
    max_edges = 40
    
    key = jax.random.PRNGKey(0)
    x = jax.random.randint(key, (n_nodes,), 0, 100)
    H_in = jnp.zeros((n_nodes, max_edges))
    H_out = jnp.zeros((n_nodes, max_edges))
    
    params = model.init(key, x, H_in, H_out)
    
    @jax.jit
    def forward(params, x, H_in, H_out):
        return model.apply(params, x, H_in, H_out)
    
    # Run once
    _ = forward(params, x, H_in, H_out)
    print("JIT compilation successful.")
    
    # Run again with different data but same shape (should be fast, no recompile)
    x2 = jax.random.randint(key, (n_nodes,), 0, 100)
    _ = forward(params, x2, H_in, H_out)
    print("Second run successful.")
    print("PASS: NovaNet is JIT compatible with static H shapes.")

if __name__ == "__main__":
    verify_causal_H()
    verify_causal_conv()
    # verify_model_jit()
