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
    input_ids = inputs["input_ids"][0].tolist() # List[int]
    
    # Generation Loop
    for _ in range(max_new_tokens):
        # 1. Prepare Input
        # We append a dummy token (0) so text_to_hypergraph treats the full input_ids as 'x'
        # x will be input_ids, y will be input_ids[1:] + [0] (ignored)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        current_ids = input_ids + [pad_id]
        
        # Construct Hypergraph
        # We pass len(current_ids) as max_seq_len to avoid truncation/padding issues inside the function
        # text_to_hypergraph returns x, H, y
        x, H, _ = text_to_hypergraph(current_ids, max_seq_len=len(current_ids))
        
        # Convert to JAX arrays
        x_jax = jnp.array(x)
        H_jax = jnp.array(H)
        
        # 2. Forward Pass
        # train=False to disable dropout
        # Wrap params in {'params': ...} because model.apply expects the variables dict
        logits, _ = model.apply({'params': params}, x_jax, H_jax, train=False)
        
        # 3. Next Token Selection
        # Get logits of the last token (corresponding to the last token in input_ids)
        next_token_logits = logits[-1]
        
        # Apply Temperature
        if temperature > 0:
            # Scale logits
            next_token_logits = next_token_logits / temperature
            # Sample
            rng, key = jax.random.split(rng)
            next_token = jax.random.categorical(key, next_token_logits)
        else:
            # Greedy
            next_token = jnp.argmax(next_token_logits)
            
        next_token_id = int(next_token)
        
        # 4. Append and Check EOS
        input_ids.append(next_token_id)
        
        if next_token_id == tokenizer.eos_token_id:
            break
            
    # Decode
    generated_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    return generated_text
