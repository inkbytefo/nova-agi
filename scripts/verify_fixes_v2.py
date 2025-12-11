
import jax
import jax.numpy as jnp
import numpy as np
from nova.core.loss import nova_loss
from nova.core.topology import update_topology
from nova.data.dataset import collate_hypergraphs
from nova.data.curriculum import CurriculumLoader

def test_nova_loss():
    print("\n--- Testing Nova Loss ---")
    n, k, d = 4, 10, 8
    logits = jax.random.normal(jax.random.PRNGKey(0), (n, k))
    targets = jnp.array([1, 2, 3, 0])
    embeddings = jax.random.normal(jax.random.PRNGKey(1), (n, d))
    # Make embeddings identical to test diversity penalty
    embeddings = jnp.ones((n, d)) 
    mask = jnp.ones((n,))
    
    loss, metrics = nova_loss(None, logits, targets, embeddings, mask, beta=0.1)
    
    # If embeddings are identical, similarity is 1.0 everywhere.
    # Off-diagonal similarity is 1.0.
    # mean_sim should be 1.0.
    # diversity_score should be 0.0.
    # total_loss = task_loss - 0.1 * 0.0 = task_loss.
    
    print(f"Mean Sim (should be 1.0): {metrics['mean_sim']:.4f}")
    print(f"Diversity Score (should be 0.0): {metrics['diversity_score']:.4f}")
    
    assert jnp.allclose(metrics['mean_sim'], 1.0, atol=1e-5)
    assert jnp.allclose(metrics['diversity_score'], 0.0, atol=1e-5)
    print("✅ Nova Loss Logic Correct (Identical Embeddings -> Zero Diversity)")

def test_topology():
    print("\n--- Testing Topology Update ---")
    n, m, d = 5, 2, 4
    H = jnp.zeros((n, m))
    # Last column is global edge
    H = H.at[:, -1].set(1.0)
    
    # Embeddings: 
    # Node 0, 1, 2 are similar.
    # Node 3 is different.
    # Node 4 (last) is similar to 0, 1, 2.
    embeddings = jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.1, 0.0, 0.0],
        [0.9, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0], # Different
        [1.0, 0.0, 0.0, 0.0]  # Similar to group
    ])
    
    # Average embedding will be dominated by first dimension (~0.8, 0.2, 0, 0).
    # Node 3 (0, 1, 0, 0) has low dot product with avg.
    # Nodes 0,1,2,4 have high dot product.
    
    H_new = update_topology(H, embeddings, threshold=0.5)
    
    global_edge = H_new[:, -1]
    print(f"Global Edge Connections: {global_edge}")
    
    assert global_edge[3] == 0.0
    assert global_edge[0] == 1.0
    print("✅ Topology Update Logic Correct (Average Embedding used)")

def test_collate():
    print("\n--- Testing Collate Hypergraphs ---")
    # Batch of 2
    # G1: 2 nodes, 1 edge.
    x1 = np.random.randn(2, 4).astype(np.float32)
    H1 = np.ones((2, 1), dtype=np.float32)
    y1 = np.array([0, 1], dtype=np.float32)
    
    # G2: 3 nodes, 2 edges.
    x2 = np.random.randn(3, 4).astype(np.float32)
    H2 = np.ones((3, 2), dtype=np.float32)
    y2 = np.array([0, 1, 2], dtype=np.float32)
    
    batch = [(x1, H1, y1), (x2, H2, y2)]
    
    x_b, H_b, y_b, mask = collate_hypergraphs(batch, num_devices=1)
    
    print(f"H_batch shape: {H_b.shape}")
    # Should be (5, 3) block diagonal
    # Top-left (2,1) is H1. Bottom-right (3,2) is H2.
    # Top-right (2,2) is 0. Bottom-left (3,1) is 0.
    
    assert H_b.shape == (5, 3)
    assert jnp.all(H_b[:2, 1:] == 0) # Top right
    assert jnp.all(H_b[2:, :1] == 0) # Bottom left
    print("✅ Collate Logic Correct (JAX Block Diag works)")

def test_curriculum():
    print("\n--- Testing Curriculum Ratios ---")
    loader = CurriculumLoader(epoch=0)
    print(f"Epoch 0 Ratios: {loader.ratios}")
    assert loader.ratios['corpus'] == 0.8
    
    loader = CurriculumLoader(epoch=2)
    print(f"Epoch 2 Ratios: {loader.ratios}")
    assert loader.ratios['instruct'] == 0.3
    
    loader = CurriculumLoader(epoch=5)
    print(f"Epoch 5 Ratios: {loader.ratios}")
    assert loader.ratios['cot'] == 0.4
    print("✅ Curriculum Ratios Correct")

if __name__ == "__main__":
    test_nova_loss()
    test_topology()
    test_collate()
    test_curriculum()
