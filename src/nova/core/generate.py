## Developer: inkbytefo
## Modified: 2025-12-11

import jax
import jax.numpy as jnp
import numpy as np
from nova.data.tokenizer import HypergraphTokenizer
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
    
    # n=0 case (empty start -> 1 token)
    # If n=0, we are creating the first token (or rather, second token if x has 1?).
    # If x was empty (n=0), new_x has 1 token.
    # We return H for 1 token. Just a global self-loop?
    # User said: "n==0 için H = jnp.ones((1,1)) global olarak."
    
    # We can handle n=0 separately with jax.lax.cond or simple logic if not traced
    # But since it's jitted, we use conditional logic or masks.
    
    # Logic for n > 0
    
    # Create new columns for Sequential, Context, and Global edges
    # Shape (n+1, 3)
    new_cols = jnp.zeros((n + 1, 3), dtype=H.dtype)
    
    # 1. Sequential Edge (n-1, n)
    # If n=0 (1 node total), we don't have n-1. 
    # If n>0 (2+ nodes total), we connect previous to current.
    # Use where to avoid index -1
    idx_prev = jnp.maximum(0, n - 1)
    # Set 1.0 at idx_prev if n > 0
    # JAX way:
    new_cols = new_cols.at[idx_prev, 0].set(jnp.where(n > 0, 1.0, 0.0))
    new_cols = new_cols.at[n, 0].set(jnp.where(n > 0, 1.0, 0.0)) # Only create seq edge if n > 0
    
    # 2. Context Edge (n-2, n-1, n)
    # If n >= 2 (3+ nodes total)
    # Indices: n-2, n-1, n
    start_ctx = jnp.maximum(0, n - 2)
    # We can set range slice dynamically? No, slices must be static in JAX usually.
    # But we can set specific indices.
    # Since window is fixed size 3:
    # We set at n, n-1, n-2 IF n >= 2.
    # We use boolean mask for the column logic?
    
    # Or simpler:
    # Just set them, but if n < 2, the edge is invalid (has < 3 nodes)?
    # User wants "last 4 context". Here we add 1 context edge per step?
    # Yes, sliding window adds 1 edge.
    # If n < 2, we shouldn't add a context edge.
    
    # Let's use mask approach for columns
    # Col 0 (Seq): Valid if n >= 1 (so total nodes >= 2)
    # Col 1 (Ctx): Valid if n >= 2 (so total nodes >= 3)
    # Col 2 (Global): Always valid.
    
    # Construct columns
    # Seq
    new_cols = new_cols.at[n, 0].set(1.0)
    new_cols = new_cols.at[n-1, 0].set(1.0)
    
    # Ctx
    new_cols = new_cols.at[n, 1].set(1.0)
    new_cols = new_cols.at[n-1, 1].set(1.0)
    new_cols = new_cols.at[n-2, 1].set(1.0)
    
    # Global
    new_cols = new_cols.at[:, 2].set(1.0)
    
    # Mask invalid columns
    # If n < 1, Seq col (0) should be 0s? Or removed?
    # We can't change shape dynamically. We just zero it out effectively.
    seq_valid = n >= 1
    ctx_valid = n >= 2
    
    new_cols = new_cols.at[:, 0].multiply(seq_valid)
    new_cols = new_cols.at[:, 1].multiply(ctx_valid)
    
    # If n=0, new_cols shape (1, 3).
    # Seq col: 0. Ctx col: 0. Global: 1.
    # But user wants H = jnp.ones((1,1)) if n=0.
    # If we return (1, 3) with 2 zero columns, it's inefficient but works.
    # But `generate` loop expects consistent growth?
    # If we return (1, 1) now, next step n=1.
    # Next step H will be (1, 1). Pad to (2, 1).
    # Add new cols (2, 3). Result (2, 4).
    # This implies variable growth of columns.
    # `jnp.concatenate` handles it.
    
    def case_n0():
        # n=0 -> 1 node. Return (1, 1) global edge.
        # Return new_H and new_x
        return jnp.ones((1, 1), dtype=H.dtype), new_x

    def case_n_gt_0():
        # H is (n, m). Pad to (n+1, m).
        H_padded = jnp.pad(H, ((0, 1), (0, 0)), mode='constant', constant_values=0)
        
        # Append new columns
        # Filter columns to only append valid ones?
        # If we append zero-columns, we waste memory.
        # But JAX `cond` needs same return shape?
        # No, `cond` needs same pytree structure, but shapes can differ IF not compiled with fixed shapes?
        # Actually `jit` requires static return shapes usually unless dynamic shapes enabled.
        # But `append_token` is called inside a loop where shapes grow.
        # JAX recompiles when shapes change.
        # So we can return different shapes (e.g. 1 col added vs 3 cols added).
        
        # However, to be cleaner/stable:
        # If n < 1, we only append Global.
        # If n < 2, we append Global + Seq.
        # If n >= 2, we append Global + Seq + Ctx.
        
        # BUT, to avoid excessive recompilation or logic branching:
        # User said "append-only H".
        # If we always append 3 columns (some empty), it's uniform.
        # But user explicitly asked "n==0 için H = jnp.ones((1,1))".
        # So let's respect that specific override.
        
        # We can use python `if` because `n` is known at trace time if passed as static?
        # `n = H.shape[0]` is static in JAX JIT if H shape is static.
        # So python `if` works.
        
        # Actually, `append_token` changes shapes, so it triggers recompilation every step anyway (unless `max_new_tokens` loop is scanned).
        # In `generate_text`, it's a python loop. So recompilation happens.
        # So we can use python control flow.
        
        # Wait, if `n` comes from `H.shape`, it is a static integer during tracing.
        
        cols_to_add = []
        
        # Seq (if n >= 1) -> Connects n-1, n
        if n >= 1:
            c_seq = jnp.zeros((n+1, 1), dtype=H.dtype)
            c_seq = c_seq.at[n-1, 0].set(1.0)
            c_seq = c_seq.at[n, 0].set(1.0)
            cols_to_add.append(c_seq)
            
        # Ctx (if n >= 2) -> Connects n-2, n-1, n
        if n >= 2:
            c_ctx = jnp.zeros((n+1, 1), dtype=H.dtype)
            c_ctx = c_ctx.at[n-2, 0].set(1.0)
            c_ctx = c_ctx.at[n-1, 0].set(1.0)
            c_ctx = c_ctx.at[n, 0].set(1.0)
            cols_to_add.append(c_ctx)
            
        # Global -> Connects 0..n
        c_glob = jnp.zeros((n+1, 1), dtype=H.dtype)
        c_glob = c_glob.at[:, 0].set(1.0)
        cols_to_add.append(c_glob)
        
        new_cols_concat = jnp.concatenate(cols_to_add, axis=1)
        
        # Concatenate with padded old H
        # Return new_H and new_x
        return jnp.concatenate([H_padded, new_cols_concat], axis=1), new_x

    if n == 0:
        return case_n0()
    else:
        return case_n_gt_0()

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
    """
    rng = jax.random.PRNGKey(seed)
    
    # Tokenize prompt
    if hasattr(tokenizer, "encode_with_topology"):
        # Use HDCT logic
        input_ids, edges = tokenizer.encode_with_topology(prompt)
        x_np, H_np, _ = text_to_hypergraph(input_ids, max_seq_len=len(input_ids), topology_edges=edges)
    else:
        # Standard tokenizer
        inputs = tokenizer(prompt, return_tensors="np")
        if "input_ids" in inputs:
            input_ids = inputs["input_ids"][0].tolist()
        else:
            input_ids = inputs # Fallback if list returned
        
        x_np, H_np, _ = text_to_hypergraph(input_ids, max_seq_len=len(input_ids))
    
    # Convert to JAX arrays
    x = jnp.array(x_np, dtype=jnp.int32)
    H = jnp.array(H_np, dtype=jnp.float32)
    
    # Generation Loop
    for _ in range(max_new_tokens):
        # Forward pass
        logits, _ = model.apply({'params': params}, x, H, train=False)
        next_token_logits = logits[-1]
        
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            rng, key = jax.random.split(rng)
            next_token = jax.random.categorical(key, next_token_logits)
        else:
            next_token = jnp.argmax(next_token_logits)
            
        # Append token and update H
        H, x = append_token(H, x, next_token)
        
        if next_token == tokenizer.eos_token_id:
            break
            
    # Decode
    output_ids = x.tolist()
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return generated_text
