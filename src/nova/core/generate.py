## Developer: inkbytefo
## Modified: 2025-12-11

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer
from nova.data.text_stream import text_to_hypergraph

@jax.jit
def append_token(H, x, new_token_id):
    """
    Appends a new token to the hypergraph and updates edges incrementally.
    
    Args:
        H: Incidence matrix (n, m).
        x: Node features (n,).
        new_token_id: ID of the new token.
        
    Returns:
        new_H: Updated incidence matrix (n+1, m+3).
        new_x: Updated node features (n+1,).
    """
    n = H.shape[0]
    
    # Update x
    new_x = jnp.concatenate([x, jnp.array([new_token_id], dtype=x.dtype)])
    
    # Create new columns for Sequential, Context, and Global edges
    # Shape (n+1, 3)
    new_cols = jnp.zeros((n + 1, 3), dtype=H.dtype)
    
    # 1. Sequential Edge (n-1, n)
    if n > 0:
        new_cols = new_cols.at[n-1, 0].set(1.0)
    new_cols = new_cols.at[n, 0].set(1.0)
    
    # 2. Context Edge (n-2, n-1, n)
    if n >= 2:
        new_cols = new_cols.at[n-2:n+1, 1].set(1.0)
        
    # 3. Global Edge (0..n)
    # Always 1s column for the current scope
    new_cols = new_cols.at[:, 2].set(1.0)
    
    # Pad existing H to (n+1, m)
    # We pad the last row with 0s (new node not connected to old edges)
    if n == 0:
        # Initial state
        new_H = new_cols[:, 2:] # Only global? Or seq/ctx too? 
        # If n=0, we have 1 node (index 0). 
        # Seq: [0]. Ctx: [0]? 
        # User logic: "if n >= 2 ... ctx".
        # "new_cols.at[n, 0].set(1.0)". Seq edge has node 0.
        # But seq edge usually needs 2 nodes? [i, i+1].
        # If n=0 (1 node), seq edge is just self-loop?
        # User code `new_H = new_cols[:, 2:]` (only global) if n=0.
        # Let's follow user code for n=0 case.
        new_H = new_cols[:, 2:]
    else:
        H_padded = jnp.pad(H, ((0, 1), (0, 0)), mode='constant', constant_values=0)
        
        # Concatenate: [Old Edges | Seq | Ctx | Global]
        # User code: new_H = jnp.concatenate([H, new_cols[:, :2]], axis=1) ... then global
        # Note: We use H_padded instead of H
        new_H = jnp.concatenate([H_padded, new_cols[:, :2]], axis=1)
        new_H = jnp.concatenate([new_H, new_cols[:, 2:3]], axis=1)
        
    return new_H, new_x

def generate_text(
    model,
    params,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 0.7,
    seed: int = 42
) -> str:
    """
    Generates text using the trained NovaNet model with incremental hypergraph construction.
    
    Args:
        model: The NovaNet model instance.
        params: The trained parameters.
        tokenizer: The tokenizer (HuggingFace).
        prompt: Input text prompt.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 for greedy).
        seed: Random seed.
        
    Returns:
        Generated text string.
    """
    rng = jax.random.PRNGKey(seed)
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"][0].tolist()
    
    # Build initial hypergraph
    # We use text_to_hypergraph for the initial chunk
    x_np, H_np, _ = text_to_hypergraph(input_ids, max_seq_len=len(input_ids))
    
    # Convert to JAX arrays
    x = jnp.array(x_np, dtype=jnp.int32)
    H = jnp.array(H_np, dtype=jnp.float32)
    
    # Generation Loop
    for _ in range(max_new_tokens):
        # Forward pass
        # We need to ensure shapes are correct for the model
        # Model expects (n, d) and (n, m).
        # We might need to add batch dimension if model expects it?
        # NovaNet usually handles unbatched input if designed so, or we unsqueeze.
        # Checking NovaNet: "x: Float[Array, 'n d']".
        # But Flax models usually handle batching via vmap or explicit dim.
        # If trained with batch, it expects batch.
        # Let's assume we need batch dim (1, n, ...).
        # But x is (n,). Model embedding will turn it to (n, d).
        
        logits, _ = model.apply({'params': params}, x, H, train=False)
        
        # Logits shape: (n, vocab_size)
        next_token_logits = logits[-1]
        
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            rng, key = jax.random.split(rng)
            next_token = jax.random.categorical(key, next_token_logits)
        else:
            next_token = jnp.argmax(next_token_logits)
            
        # Append token and update H
        H, x = append_token(H, x, next_token)
        
        # Stop if EOS (optional, depending on tokenizer)
        if next_token == tokenizer.eos_token_id:
            break
            
    # Decode
    output_ids = x.tolist()
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return generated_text
