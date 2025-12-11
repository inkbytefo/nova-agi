## Developer: inkbytefo
## Modified: 2025-12-11

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Dict, Iterator
from jaxtyping import Float, Array
import chex

def generate_synthetic_data(
    num_samples: int,
    min_nodes: int = 10,
    max_nodes: int = 20,
    min_edges: int = 5,
    max_edges: int = 15,
    feature_dim: int = 8,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generates synthetic hypergraph data.
    
    Args:
        num_samples: Number of graphs to generate.
        min_nodes: Minimum number of nodes per graph.
        max_nodes: Maximum number of nodes per graph.
        min_edges: Minimum number of edges per graph.
        max_edges: Maximum number of edges per graph.
        feature_dim: Node feature dimension.
        seed: Random seed.
        
    Returns:
        List of tuples (x, H, y).
        x: (n, feature_dim)
        H: (n, m)
        y: (n, 1) - Target (e.g., node degree scaled)
    """
    rng = np.random.default_rng(seed)
    data = []
    
    for _ in range(num_samples):
        n = rng.integers(min_nodes, max_nodes + 1)
        m = rng.integers(min_edges, max_edges + 1)
        
        # Node features
        x = rng.standard_normal((n, feature_dim)).astype(np.float32)
        
        # Incidence matrix (Bernoulli)
        # Ensure at least one connection per edge/node usually, but random is fine for synthetic
        H = rng.binomial(1, 0.3, size=(n, m)).astype(np.float32)
        
        # Ensure no empty rows/cols to avoid NaNs in degree norm (though ops.py handles eps)
        # Let's just leave it random.
        
        # Target: Node degree (row sum of H)
        # Simple regression task: predict node degree
        # Normalize target roughly
        d_v = H.sum(axis=1, keepdims=True)
        y = (d_v / m).astype(np.float32)
        
        data.append((x, H, y))
        
    return data

import jax.scipy.linalg

def collate_hypergraphs(
    batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_devices: int = 1
) -> Tuple[Float[Array, "B N_max d"], Float[Array, "B N_max M_max"], Float[Array, "B N_max 1"], Float[Array, "B N_max"]]:
    """
    Collates a batch of hypergraphs. 
    If num_devices > 1, splits batch and pads to create a sharded batch for pmap.
    
    Args:
        batch: List of (x, H, y) tuples.
        num_devices: Number of devices to shard for.
        
    Returns:
        If num_devices == 1:
            x_batch: (N_total, d)
            H_batch: (N_total, M_total)
            y_batch: (N_total, 1)
            mask: (N_total,) - All ones
        If num_devices > 1:
            x_batch: (num_devices, N_max, d)
            H_batch: (num_devices, N_max, M_max)
            y_batch: (num_devices, N_max, 1)
            mask: (num_devices, N_max) - 1 for valid nodes, 0 for padded
    """
    # Note: We use jax.scipy.linalg.block_diag for efficiency as requested.
    # This assumes we are in the main process or JAX is safe to use.
    
    xs, Hs, ys = zip(*batch)
    
    if num_devices == 1:
        # Standard disjoint union for single device
        x_batch = np.concatenate(xs, axis=0)
        
        # Use jax.scipy.linalg.block_diag for H
        # Convert to JAX arrays first to ensure correct dispatch, or let JAX handle numpy inputs
        # Unpack list of H matrices
        H_batch = jax.scipy.linalg.block_diag(*Hs)
        
        y_batch = np.concatenate(ys, axis=0)
        if y_batch.ndim == 1:
            y_batch = y_batch[:, None]
        mask = np.ones((x_batch.shape[0],), dtype=np.float32)
        return x_batch, H_batch, y_batch, mask
        
    # Sharding logic
    batch_size = len(batch)
    assert batch_size % num_devices == 0, f"Batch size {batch_size} must be divisible by num_devices {num_devices}"
    sub_batch_size = batch_size // num_devices
    
    sub_xs, sub_Hs, sub_ys, sub_masks = [], [], [], []
    
    # First pass: Collate sub-batches to find max dimensions
    collated_subs = []
    max_n = 0
    max_m = 0
    
    for i in range(num_devices):
        start = i * sub_batch_size
        end = start + sub_batch_size
        sub_batch = batch[start:end]
        
        # Collate this shard
        s_xs, s_Hs, s_ys = zip(*sub_batch)
        c_x = np.concatenate(s_xs, axis=0)
        c_H = jax.scipy.linalg.block_diag(*s_Hs) # Use JAX block_diag
        c_y = np.concatenate(s_ys, axis=0)
        if c_y.ndim == 1:
            c_y = c_y[:, None]
        
        collated_subs.append((c_x, c_H, c_y))
        max_n = max(max_n, c_x.shape[0])
        max_m = max(max_m, c_H.shape[1])
        
    # Second pass: Pad and stack
    # Determine shapes and types from first sample
    if xs[0].ndim == 1:
        # 1D features (e.g. token IDs)
        final_x = np.zeros((num_devices, max_n), dtype=np.int32)
    else:
        feature_dim = xs[0].shape[1]
        final_x = np.zeros((num_devices, max_n, feature_dim), dtype=np.float32)
    
    final_H = np.zeros((num_devices, max_n, max_m), dtype=np.float32)
    
    # Determine target shape and dtype
    y_sample = ys[0]
    y_dtype = y_sample.dtype
    if y_sample.ndim == 1:
        final_y = np.zeros((num_devices, max_n), dtype=y_dtype)
    else:
        y_dim = y_sample.shape[1]
        final_y = np.zeros((num_devices, max_n, y_dim), dtype=y_dtype)
    
    final_mask = np.zeros((num_devices, max_n), dtype=np.float32)
    
    for i, (c_x, c_H, c_y) in enumerate(collated_subs):
        n, m = c_H.shape
        if c_x.ndim == 1:
            final_x[i, :n] = c_x
        else:
            final_x[i, :n, :] = c_x
        final_H[i, :n, :m] = c_H
        
        # Handle y assignment based on dimension
        if c_y.ndim == 1:
             final_y[i, :n] = c_y
        else:
             final_y[i, :n, :] = c_y
             
        final_mask[i, :n] = 1.0
        
    return final_x, final_H, final_y, final_mask

def get_dataloader(
    data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
    drop_last: bool = True,
    num_devices: int = 1
):
    """
    Generator for batched data with optional sharding support.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(data))
    
    if shuffle:
        rng.shuffle(indices)
        
    for start_idx in range(0, len(data), batch_size):
        end_idx = start_idx + batch_size
        
        # Drop last incomplete batch if requested
        if drop_last and end_idx > len(data):
            break
            
        # Handle last batch if not dropping (might crash sharding if not divisible)
        if end_idx > len(data):
            end_idx = len(data)
            
        batch_indices = indices[start_idx:end_idx]
        
        # Skip if batch is empty
        if len(batch_indices) == 0:
            break
            
        # Ensure divisibility for sharding if not dropping last (or if last batch happened to be uneven)
        if num_devices > 1 and len(batch_indices) % num_devices != 0:
            if drop_last:
                continue
            else:
                # Warn or handle? For now, strict requirement
                print(f"Warning: Batch size {len(batch_indices)} not divisible by {num_devices}. Skipping.")
                continue
        
        batch = [data[i] for i in batch_indices]
        yield collate_hypergraphs(batch, num_devices=num_devices)

def get_streaming_dataloader(
    data_iterator: Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    batch_size: int,
    num_devices: int = 1
):
    """
    Generator for batched data from a stream with optional sharding support.
    """
    batch = []
    for item in data_iterator:
        batch.append(item)
        
        if len(batch) == batch_size:
            yield collate_hypergraphs(batch, num_devices=num_devices)
            batch = []
