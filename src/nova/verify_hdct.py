
import numpy as np
import jax
import jax.numpy as jnp
from nova.data.tokenizer import HypergraphTokenizer
from nova.data.text_stream import text_to_hypergraph
from nova.models.nova import NovaNet
from nova.data.hypergraph_builder import build_incremental_H

def verify_hdct():
    print("=== Verifying HDCT ===")
    
    # 1. Test Tokenizer
    tokenizer = HypergraphTokenizer()
    text = "Devrimsel"
    encoded = tokenizer(text)
    ids = encoded['input_ids'][0]
    
    print(f"Text: '{text}'")
    print(f"IDs: {ids}")
    decoded = tokenizer.decode(ids)
    print(f"Decoded: '{decoded}'")
    
    # Expected: "Devrimsel" -> len 9 + BOS + EOS = 11 IDs (if no padding)
    # Our tokenizer adds BOS/EOS.
    assert "Devrimsel" in decoded, "Decoding failed to recover text"
    
    # 2. Test Hypergraph Construction
    print("\n--- Hypergraph Construction ---")
    x, H, y = text_to_hypergraph(ids.tolist(), max_seq_len=20)
    
    print(f"x shape: {x.shape}")
    print(f"H shape: {H.shape}")
    print(f"y shape: {y.shape}")
    
    # x should be length 10 (11 IDs - 1 for target shift)
    assert x.shape[0] == len(ids) - 1, f"x shape mismatch: {x.shape[0]} vs {len(ids)-1}"
    
    # Check H edges
    # We expect sequential (2-gram), context (3-gram), 4-gram, 5-gram + Global
    # Let's count expected edges roughly
    n = len(x) # 10
    # 2-gram: n-1 = 9
    # 3-gram: n-2 = 8
    # 4-gram: n-3 = 7
    # 5-gram: n-4 = 6
    # Global: 1
    # Total ~ 31 edges
    
    print(f"Number of edges: {H.shape[1]}")
    # Verify incidence values
    print(f"H sum (total incidence): {H.sum()}")
    
    # Check specific edge types
    # First edge should be [0, 1] (BOS, 'D')
    first_edge = H[:, 0]
    indices = np.where(first_edge == 1)[0]
    print(f"First edge nodes: {indices}")
    assert len(indices) == 2, "First edge should be 2-gram"
    
    # 3. Test NovaNet Forward Pass
    print("\n--- NovaNet Inference ---")
    model = NovaNet(
        hidden_dim=32,
        num_layers=2,
        out_dim=tokenizer.vocab_size,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=16
    )
    
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, x, H, train=True)
    
    logits, embeddings = model.apply(params, x, H, train=True)
    
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (n, tokenizer.vocab_size), "Logits shape mismatch"
    print("NovaNet forward pass successful!")

if __name__ == "__main__":
    verify_hdct()
