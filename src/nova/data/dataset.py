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
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generates synthetic causal hypergraph data.
    """
    rng = np.random.default_rng(seed)
    data = []
    
    for _ in range(num_samples):
        n = rng.integers(10, max_nodes + 1)
        m = rng.integers(min_edges, max_edges + 1)
        
        # Node features
        x = rng.standard_normal((n, feature_dim)).astype(np.float32)
        
        # Causal Incidence Matrices
        # H_in: Gather from past
        H_in = np.zeros((n, m), dtype=np.float32)
        # H_out: Scatter to present
        H_out = np.zeros((n, m), dtype=np.float32)
        
        for j in range(m):
            # Random edge span [start, end]
            end = rng.integers(0, n)
            length = rng.integers(1, 4) # 1 to 3 nodes
            start = max(0, end - length + 1)
            
            H_in[start:end+1, j] = 1.0
            H_out[end, j] = 1.0
        
        # Target
        d_v = H_in.sum(axis=1, keepdims=True)
        y = (d_v / m).astype(np.float32)
        
        data.append((x, H_in, H_out, y))
        
    return data

import jax.scipy.linalg

def collate_hypergraphs(
    batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    num_devices: int = 1
) -> Tuple[Float[Array, "B N_max d"], Float[Array, "B N_max M_max"], Float[Array, "B N_max M_max"], Float[Array, "B N_max 1"], Float[Array, "B N_max"]]:
    """
    Collates a batch of causal hypergraphs.
    """
    xs, H_ins, H_outs, ys = zip(*batch)
    batch_size = len(batch)
    
    # 1. Determine Max Dimensions
    max_n = max(x.shape[0] for x in xs)
    max_m_in = max(h.shape[1] for h in H_ins)
    max_m_out = max(h.shape[1] for h in H_outs)
    max_m = max(max_m_in, max_m_out)
    
    # Determine shapes and types from first sample
    if xs[0].ndim == 1:
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
    final_x = np.zeros((batch_size, max_n) + feature_shape, dtype=dtype_x)
    final_H_in = np.zeros((batch_size, max_n, max_m), dtype=np.float32)
    final_H_out = np.zeros((batch_size, max_n, max_m), dtype=np.float32)
    final_y = np.zeros((batch_size, max_n) + y_shape, dtype=dtype_y)
    final_mask = np.zeros((batch_size, max_n), dtype=np.float32)
    
    # 3. Fill Arrays
    for i in range(batch_size):
        x, H_in, H_out, y = xs[i], H_ins[i], H_outs[i], ys[i]
        n = x.shape[0]
        m_in = H_in.shape[1]
        m_out = H_out.shape[1]
        
        # Fill x
        if x.ndim == 1:
            final_x[i, :n] = x
        else:
            final_x[i, :n, :] = x
            
        # Fill H
        final_H_in[i, :n, :m_in] = H_in
        final_H_out[i, :n, :m_out] = H_out
        
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
        final_H_in = final_H_in.reshape((num_devices, sub_batch, max_n, max_m))
        final_H_out = final_H_out.reshape((num_devices, sub_batch, max_n, max_m))
        final_y = final_y.reshape((num_devices, sub_batch, max_n) + y_shape)
        final_mask = final_mask.reshape((num_devices, sub_batch, max_n))
        
    return final_x, final_H_in, final_H_out, final_y, final_mask

def get_dataloader(
    data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
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
