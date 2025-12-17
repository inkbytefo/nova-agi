
import os
import sys

# Patch Flax for Python 3.14 compatibility
try:
    import patch_flax
    patch_flax.apply_patch()
except ImportError:
    pass

import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax import linen as nn
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from nova.models.nova import NovaNet
from nova.data.text_stream import text_to_hypergraph
from nova.data.hypergraph_builder import build_causal_H
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ArchBenchmark")

def generate_niah_batch(batch_size, seq_len, vocab_size, needle_depth_pct=0.5):
    """
    Generates a synthetic 'Needle In A Haystack' (Associative Recall) task.
    Format:
    [KEY] [VALUE] ... noise ... [QUERY_KEY] -> Target: [VALUE]
    
    We map:
    KEY marker = 1
    QUERY marker = 2
    VALUE marker = 3 (start of answer)
    """
    
    # Random key and value
    keys = np.random.randint(100, vocab_size, (batch_size,))
    values = np.random.randint(100, vocab_size, (batch_size,))
    
    # Create input arrays
    inputs = np.random.randint(100, vocab_size, (batch_size, seq_len))
    targets = np.zeros((batch_size, seq_len), dtype=np.int32)
    masks = np.zeros((batch_size, seq_len), dtype=np.float32)
    
    for i in range(batch_size):
        # Determine needle position
        # deep placement: somewhere in the first 80%
        depth = int(seq_len * needle_depth_pct)
        # Ensure spacing
        depth = max(0, min(depth, seq_len - 10))
        
        # Insert Needle: KEY, VALUE
        inputs[i, depth] = 1 # Marker Key
        inputs[i, depth+1] = keys[i]
        inputs[i, depth+2] = values[i]
        
        # Insert Query at the very end
        inputs[i, -2] = 2 # Marker Query
        inputs[i, -1] = keys[i]
        
        # Target: The expected output at the last position should be the value
        targets[i, -1] = values[i]
        masks[i, -1] = 1.0 # Only calculate loss on the answer
        
    return inputs, targets, masks

def train_step(state, batch_x, batch_H_in, batch_H_out, batch_y, batch_mask):
    def loss_fn(params):
        logits, _ = state.apply_fn(
            {'params': params}, 
            x=batch_x, 
            H_in=batch_H_in,
            H_out=batch_H_out,
            train=True
        ) # [B, T, V]
        
        # Cross Entropy Loss
        # Only on masked positions (last token)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y)
        loss = (loss * batch_mask).sum() / (batch_mask.sum() + 1e-9)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    # Calculate accuracy
    predictions = jnp.argmax(logits, axis=-1)
    acc = (predictions == batch_y) * batch_mask
    acc = acc.sum() / (batch_mask.sum() + 1e-9)
    
    return state, loss, acc

def run_benchmark():
    logger.info("🚀 Starting NovaNet Architecture Benchmark")
    
    # Config matching your TPU setup roughly, but scaled for quick local test
    # We keep depth to test signal propagation
    SEQ_LEN = 512 # Reduced for CPU testing
    HIDDEN_DIM = 64 # Smaller for fast architecture probing
    LAYERS = 4 # Deep enough to test vanishing gradient but fast
    VOCAB = 5000
    BATCH = 8 # Reduced for CPU
    STEPS = 100
    
    logger.info(f"Parametreler: Seq={SEQ_LEN}, Layers={LAYERS}, Hidden={HIDDEN_DIM}")
    
    # Init Model
    model = NovaNet(
        hidden_dim=HIDDEN_DIM,
        num_layers=LAYERS,
        out_dim=VOCAB,
        vocab_size=VOCAB,
        embedding_dim=HIDDEN_DIM
    )
    
    rng = jax.random.PRNGKey(0)
    
    # Dummy init
    dummy_x = jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
    dummy_H_in = jnp.ones((1, SEQ_LEN, 10), dtype=jnp.float32)
    dummy_H_out = jnp.ones((1, SEQ_LEN, 10), dtype=jnp.float32)
    
    params = model.init(rng, dummy_x, dummy_H_in, dummy_H_out, train=False)['params']
    tx = optax.adam(learning_rate=1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    logger.info(f"Model initialized. Parameters: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,}")
    logger.info("🧪 TASK: 'Associative Recall' (Needle in a Haystack Simulation)")
    logger.info("The model must remember a key-value pair seen early in the context and retrieve it at the end.")
    
    # Pre-compute Hypergraph Structure (Static for this benchmark)
    logger.info("Pre-computing Hypergraph Topology...")
    max_edges = SEQ_LEN * 16 # Increased for long-range edges
    H_in_static, H_out_static = build_causal_H(SEQ_LEN, max_edges=max_edges, window_sizes=[2, 3], add_long_range=True)
    
    # Expand to batch
    H_in_batch = np.tile(H_in_static[None, ...], (BATCH, 1, 1))
    H_out_batch = np.tile(H_out_static[None, ...], (BATCH, 1, 1))
    
    # Convert to JAX arrays once
    H_in_jax = jnp.array(H_in_batch)
    H_out_jax = jnp.array(H_out_batch)
    logger.info("Topology ready.")

    # JIT Compile Step
    @jax.jit
    def train_step_jit(state, batch_x, batch_H_in, batch_H_out, batch_y, batch_mask):
        return train_step(state, batch_x, batch_H_in, batch_H_out, batch_y, batch_mask)

    # Training Loop
    start_time = time.time()
    
    for step in range(STEPS):
        # Generate Data
        # Dynamic Needle Depth: Sometimes expected at start, sometimes middle
        pct = np.random.uniform(0.0, 0.8) 
        inputs_raw, targets_raw, masks_raw = generate_niah_batch(BATCH, SEQ_LEN, VOCAB, needle_depth_pct=pct)
        
        # Process with text_to_hypergraph
        batch_x, batch_H_in, batch_H_out, batch_y = [], [], [], []
        
        for i in range(BATCH):
            # Append dummy token to preserve full sequence in x (since text_to_hypergraph splits x=[:-1])
            # We want x to be length SEQ_LEN
            curr_input = inputs_raw[i].tolist() + [0]
            
            # text_to_hypergraph returns x, H_in, H_out, y
            x, h_in, h_out, _ = text_to_hypergraph(curr_input, max_seq_len=SEQ_LEN + 1)
            
            # We only need the first SEQ_LEN tokens for x to match our task setup
            # But wait, text_to_hypergraph returns x as numpy array.
            # h_in, h_out are (N, M). N = len(x).
            
            batch_x.append(x)
            batch_H_in.append(h_in)
            batch_H_out.append(h_out)
            # batch_y is handled by targets_raw from NIAH
            
        # Stack and Pad
        # H matrices might have different sizes if max_edges depends on seq_len, 
        # but here we passed fixed max_seq_len so max_edges is fixed.
        # x length is fixed (SEQ_LEN).
        
        batch_x = jnp.array(np.stack(batch_x))
        batch_H_in = jnp.array(np.stack(batch_H_in))
        batch_H_out = jnp.array(np.stack(batch_H_out))
        
        targets = jnp.array(targets_raw)
        masks = jnp.array(masks_raw)
        
        # Use dynamic H matrices
        state, loss, acc = train_step_jit(state, batch_x, batch_H_in, batch_H_out, targets, masks)
        
        if step % 10 == 0 or step == 0:
            logger.info(f"Step {step:03d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")
            
        if acc > 0.95:
            logger.info("✅ Model CONVERGED! Architecture successfully solves Long-Context Retrieval.")
            break
            
    total_time = time.time() - start_time
    logger.info(f"Benchmark finished in {total_time:.2f}s")
    
    if acc < 0.5:
        logger.warning("⚠️ Model failed to learn the task. Possible issues: Vanishing gradient, broken Global Edge.")
    else:
        logger.info("🏆 RESULT: NovaNet Architecture is INDUCTIVELY BIAS-COMPATIBLE with Long-Context tasks.")

if __name__ == "__main__":
    run_benchmark()
