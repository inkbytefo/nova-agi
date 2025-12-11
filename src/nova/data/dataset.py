## Developer: inkbytefo
## Modified: 2025-12-11

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Dict, Iterator
from jaxtyping import Float, Array
import chex
from datasets import load_dataset, interleave_datasets, concatenate_datasets

def load_turkish_corpus(
    config: Dict,
    split: str = "train",
    streaming: bool = True
):
    """
    Loads and mixes modern Turkish datasets (Cosmos, BellaTurca) based on config.
    Handles manual validation splitting for datasets without it.
    """
    sources = config.get("corpus_sources", {})
    datasets_list = []
    probabilities = []
    
    for key, source_cfg in sources.items():
        path = source_cfg.get("path")
        name = source_cfg.get("config_name", None) # Support subset/config name
        weight = source_cfg.get("weight", 1.0)
        
        try:
            # Try loading specific split, else fallback to 'train'
            # For Cosmos/BellaTurca, we likely only have 'train'
            ds = load_dataset(path, name, split="train", streaming=streaming)
            
            # Validation Logic:
            # If split is 'validation', we skip the first N samples.
            # If split is 'train', we take from N onwards (or just mix if we ignore overlap for now)
            # ideally we skip first 1000 for validation
            val_size = 1000
            
            if split == "validation":
                ds = ds.take(val_size)
            else:
                ds = ds.skip(val_size)
                
            datasets_list.append(ds)
            probabilities.append(weight)
            print(f"Loaded {key} ({path}) for split {split}")
        except Exception as e:
            print(f"Failed to load {key}: {e}")
            
    if not datasets_list:
        raise ValueError("No datasets loaded successfully!")
        
    # Normalize probabilities
    total = sum(probabilities)
    probabilities = [p/total for p in probabilities]
    
    mixed_ds = interleave_datasets(datasets_list, probabilities=probabilities, seed=42)
    return mixed_ds

def generate_synthetic_data(
    num_samples: int,
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
    Collates a batch of hypergraphs by stacking them (with padding).
    Preserves the batch dimension to allow per-sample attention.
    If num_devices > 1, reshapes to (num_devices, sub_batch, ...).
    
    Args:
        batch: List of (x, H, y) tuples.
        num_devices: Number of devices to shard for.
        
    Returns:
        x_batch: (..., N_max, d)
        H_batch: (..., N_max, M_max)
        y_batch: (..., N_max, 1) or (..., N_max, y_dim)
        mask: (..., N_max) - 1 for valid nodes, 0 for padded
    """
    xs, Hs, ys = zip(*batch)
    batch_size = len(batch)
    
    # 1. Determine Max Dimensions
    # We find the max N and M across the *entire* batch to ensure uniform shape.
    max_n = max(x.shape[0] for x in xs)
    max_m = max(h.shape[1] for h in Hs)
    
    # Determine shapes and types from first sample
    if xs[0].ndim == 1:
        # 1D features (e.g. token IDs)
        feature_shape = ()
        dtype_x = np.int32
    else:
        feature_dim = xs[0].shape[1]
        feature_shape = (feature_dim,)
        dtype_x = np.float32
        
    y_sample = ys[0]
    dtype_y = y_sample.dtype
    if y_sample.ndim == 1:
        y_shape = ()
    else:
        y_dim = y_sample.shape[1]
        y_shape = (y_dim,)
        
    # 2. Allocate Padding Arrays
    # Shapes: [Batch, MaxN, ...]
    final_x = np.zeros((batch_size, max_n) + feature_shape, dtype=dtype_x)
    final_H = np.zeros((batch_size, max_n, max_m), dtype=np.float32)
    final_y = np.zeros((batch_size, max_n) + y_shape, dtype=dtype_y)
    final_mask = np.zeros((batch_size, max_n), dtype=np.float32)
    
    # 3. Fill Arrays
    for i in range(batch_size):
        x, H, y = xs[i], Hs[i], ys[i]
        n, m = H.shape
        
        # Fill x
        if x.ndim == 1:
            final_x[i, :n] = x
        else:
            final_x[i, :n, :] = x
            
        # Fill H
        final_H[i, :n, :m] = H
        
        # Fill y
        if y.ndim == 1:
            final_y[i, :n] = y
        else:
            final_y[i, :n, :] = y
            
        # Fill mask
        final_mask[i, :n] = 1.0
        
    # 4. Handle Sharding if needed
    if num_devices > 1:
        assert batch_size % num_devices == 0, f"Batch size {batch_size} must be divisible by num_devices {num_devices}"
        sub_batch = batch_size // num_devices
        
        # Reshape to [num_devices, sub_batch, ...]
        final_x = final_x.reshape((num_devices, sub_batch, max_n) + feature_shape)
        final_H = final_H.reshape((num_devices, sub_batch, max_n, max_m))
        final_y = final_y.reshape((num_devices, sub_batch, max_n) + y_shape)
        final_mask = final_mask.reshape((num_devices, sub_batch, max_n))
        
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
