## Developer: inkbytefo
## Modified: 2025-12-11

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import jax
import jax.numpy as jnp
import pytest
from transformers import AutoTokenizer
from nova.models.nova import NovaNet
from nova.core.generate import generate_text

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 102
        self.vocab_size = 1000

    def __call__(self, text, return_tensors="np"):
        # simple mock encoding
        return {"input_ids": jnp.array([[2, 3, 4]])}

    def decode(self, token_ids, skip_special_tokens=True):
        return "mock generated text"

def test_generate_text_mock():
    # Setup
    tokenizer = MockTokenizer()
    model = NovaNet(
        hidden_dim=32,
        num_layers=1,
        out_dim=tokenizer.vocab_size,
        dropout_rate=0.0,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=16
    )
    
    # Init params
    rng = jax.random.PRNGKey(42)
    dummy_x = jnp.zeros((1, 5), dtype=jnp.int32)
    dummy_H = jnp.zeros((1, 5, 2))
    variables = model.init(rng, dummy_x, dummy_H)
    params = variables['params']
    
    # Test generation
    prompt = "Test prompt"
    generated = generate_text(
        model=model,
        params=params,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=5,
        temperature=0.0 # Greedy
    )
    
    assert isinstance(generated, str)
    assert generated == "mock generated text"

@pytest.mark.skip(reason="Requires internet connection and model download")
def test_generate_text_real():
    try:
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    except Exception:
        pytest.skip("Could not load tokenizer")
        
    model = NovaNet(
        hidden_dim=32,
        num_layers=1,
        out_dim=tokenizer.vocab_size,
        dropout_rate=0.0,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=16
    )
    
    rng = jax.random.PRNGKey(42)
    dummy_x = jnp.zeros((1, 5), dtype=jnp.int32)
    dummy_H = jnp.zeros((1, 5, 2))
    variables = model.init(rng, dummy_x, dummy_H)
    params = variables['params']
    
    prompt = "Merhaba"
    generated = generate_text(
        model=model,
        params=params,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=5,
        temperature=0.7
    )
    
    assert isinstance(generated, str)
    assert len(generated) > 0
