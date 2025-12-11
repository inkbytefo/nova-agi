## Developer: inkbytefo
## Modified: 2025-12-11

from typing import Iterator, Tuple, List
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

from nova.data.hypergraph_builder import build_incremental_H
from nova.data.tokenizer import HypergraphTokenizer

def text_to_hypergraph(token_ids: List[int], max_seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a sequence of token IDs into a hypergraph structure using optimized incremental logic.
    
    Args:
        token_ids: List of token IDs.
        max_seq_len: Maximum sequence length.
        
    Returns:
        x: Node features (token IDs) of shape (N,).
        H: Incidence matrix of shape (N, M).
        y: Target token ID (next token) of shape (N,).
    """
    # Fix for short sequences (n < 2)
    if len(token_ids) < 2:
        if len(token_ids) == 1:
            # Only 1 token: input x=[id], target y=[pad] (or 0)
            x = np.array([token_ids[0]], dtype=np.int32)
            y = np.array([0], dtype=np.int32) # Padding token usually 0
            # Return a valid H with 1 global edge (self-loop)
            H = np.ones((1, 1), dtype=np.float32)
            return x, H, y
        else:
            # Empty
            return np.array([], dtype=np.int32), np.zeros((0,0), dtype=np.float32), np.array([], dtype=np.int32)

    x = np.array(token_ids[:-1], dtype=np.int32)
    y = np.array(token_ids[1:], dtype=np.int32)
    
    n_nodes = len(x)
    
    # For training, build graph over entire sequence
    # Use full length for seq_fixed and ctx_fixed to ensure connectivity across the whole sequence
    H = build_incremental_H(n_nodes, seq_fixed=16, ctx_fixed=8)
    
    return x, H, y

class TurkishTextStream:
    """
    Streaming dataset loader for Turkish text corpora using HDCT (Hypergraph-Dynamic Chunking Tokenizer).
    """
    def __init__(self, dataset_name: str = "ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0", max_seq_len: int = 512, split: str = "train"):
        self.max_seq_len = max_seq_len
        # Switch to tokenizer-free / character-level
        self.tokenizer = HypergraphTokenizer(vocab_size=5000)
        
        try:
            self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        except Exception as e:
            if split == "validation":
                print(f"Warning: 'validation' split not found for {dataset_name}. Using first 2048 samples of 'train' as validation.")
                try:
                    # Fallback: use a subset of train for validation
                    ds = load_dataset(dataset_name, split="train", streaming=True)
                    self.dataset = ds.take(2048)
                except Exception as e2:
                    print(f"Error: Failed to load fallback validation data: {e2}")
                    self.dataset = []
            else:
                print(f"Warning: Failed to load dataset {dataset_name} (split={split}): {e}")
                self.dataset = []

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for example in self.dataset:
            text = example.get('text', '')
            if not text:
                continue
                
            # Tokenize using HDCT (Character level)
            encoded = self.tokenizer(
                text,
                max_length=self.max_seq_len,
                padding="max_length",
                return_tensors="np"
            )
            
            # Extract IDs
            if "input_ids" in encoded:
                # Handle possible batch dimension if provided by some tokenizer implementations,
                # but our class returns [1, L] for return_tensors='np'
                ids = encoded["input_ids"][0].tolist()
                
                x, H, y = text_to_hypergraph(ids, self.max_seq_len)
                
                if len(x) > 0:
                    yield x, H, y
