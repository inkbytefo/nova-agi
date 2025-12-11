## Developer: inkbytefo
## Modified: 2025-12-11

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer
from nova.data.text_stream import text_to_hypergraph

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
    Generates text using the trained NovaNet model.
    
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
    
    # Build initial hypergraph once (cache base)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    init_ids = input_ids + [pad_id]
    x_np, H_np, _ = text_to_hypergraph(init_ids, max_seq_len=len(init_ids))

    # Generation Loop with incremental hypergraph updates
    for _ in range(max_new_tokens):
        # Forward pass
        x_jax = jnp.array(x_np)
        H_jax = jnp.array(H_np)
        logits, _ = model.apply({'params': params}, x_jax, H_jax, train=False)

        # Next token selection
        next_token_logits = logits[-1]
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            rng, key = jax.random.split(rng)
            next_token = jax.random.categorical(key, next_token_logits)
        else:
            next_token = jnp.argmax(next_token_logits)
        next_token_id = int(next_token)

        # Append token
        input_ids.append(next_token_id)

        # Stop if EOS
        if next_token_id == tokenizer.eos_token_id:
            break

        # Incrementally update x and H without full recomputation
        # x grows by one previous token (the last token before the new one)
        prev_last_id = input_ids[-2]
        x_np = np.concatenate([x_np, np.array([prev_last_id], dtype=x_np.dtype)])

        # H update: maintain last column as global hyperedge
        n, m = H_np.shape
        H_base = H_np[:, :m-1]
        global_col = H_np[:, m-1:m]

        # Prepare new row of zeros for existing edges (excluding global)
        new_row_base = np.zeros((1, H_base.shape[1]), dtype=H_np.dtype)

        # New sequential edge column: connect nodes n-1 and n
        seq_col = np.zeros((n+1, 1), dtype=H_np.dtype)
        if n >= 1:
            seq_col[n-1, 0] = 1.0
            seq_col[n, 0] = 1.0

        # New context edge column: connect nodes n-2, n-1, n (if available)
        ctx_col = np.zeros((n+1, 1), dtype=H_np.dtype)
        if n >= 2:
            ctx_col[n-2, 0] = 1.0
        if n >= 1:
            ctx_col[n-1, 0] = 1.0
            ctx_col[n, 0] = 1.0

        # Update global column by appending 1 for the new node
        global_col_updated = np.concatenate([global_col, np.array([[1.0]], dtype=H_np.dtype)], axis=0)

        # Assemble new H: existing base edges with appended row, then new edges, then global
        H_base_row_appended = np.concatenate([H_base,], axis=0)
        H_base_row_appended = np.vstack([H_base_row_appended, new_row_base])
        H_np = np.concatenate([H_base_row_appended, seq_col, ctx_col, global_col_updated], axis=1)
            
    # Decode
    generated_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    return generated_text
