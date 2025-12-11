## Developer: inkbytefo
## Modified: 2025-12-11

from typing import Iterator, Tuple, List
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

def text_to_hypergraph(token_ids: List[int], max_seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a sequence of token IDs into a hypergraph structure.
    
    Args:
        token_ids: List of token IDs.
        max_seq_len: Maximum sequence length.
        
    Returns:
        x: Node features (token IDs) of shape (N,).
        H: Incidence matrix of shape (N, M).
        y: Target token ID (next token) of shape (N,).
    """
    # Use numpy for efficient array handling
    # Input x: token_ids[:-1]
    # Target y: token_ids[1:]
    
    # Ensure we have enough tokens
    if len(token_ids) < 3:
        # If too short, just return zeros or handle gracefully?
        # For now, let's pad with 0 manually if needed, or assume tokenizer handles it.
        # But if tokenizer was called with padding='max_length', we are fine.
        pass

    x = np.array(token_ids[:-1], dtype=np.int32)
    y = np.array(token_ids[1:], dtype=np.int32)
    
    n_nodes = len(x)
    
    # Incidence Matrix Construction
    # Edge Type 1 (Sequential): Sliding window of size 2 (connect t_i, t_{i+1})
    # Edge Type 2 (Context): Sliding window of size 3 (connect t_i, t_{i+1}, t_{i+2})
    # Edge Type 3 (Global): One hyperedge connecting ALL tokens
    
    edges = []
    
    # 1. Sequential (Window 2)
    for i in range(n_nodes - 1):
        edges.append([i, i+1])
        
    # 2. Context (Window 3)
    for i in range(n_nodes - 2):
        edges.append([i, i+1, i+2])
        
    # 3. Global
    if n_nodes > 0:
        edges.append(list(range(n_nodes)))
    
    n_edges = len(edges)
    
    # If no edges (e.g. very short sequence), create a dummy edge or empty H
    if n_edges == 0:
        # Fallback: self loops?
        H = np.eye(n_nodes, dtype=np.float32)
    else:
        H = np.zeros((n_nodes, n_edges), dtype=np.float32)
        for e_idx, nodes in enumerate(edges):
            H[nodes, e_idx] = 1.0
            
    return x, H, y

class TurkishTextStream:
    def __init__(self, max_seq_len: int = 512, split: str = "train"):
        """
        Streaming dataset for Turkish text using C4.
        
        Args:
            max_seq_len: Maximum sequence length for tokenization.
        """
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        self.dataset = self._load_tr_stream(split)

    @staticmethod
    def _load_tr_stream(split: str):
        paths = [
            f"hf://datasets/allenai/c4@refs/convert/parquet/tr/{split}/*.parquet",
            f"hf://datasets/mc4@refs/convert/parquet/tr/{split}/*.parquet",
        ]
        for p in paths:
            try:
                return load_dataset(
                    "parquet",
                    data_files=p,
                    split=split,
                    streaming=True,
                    columns=["text"],
                )
            except Exception:
                continue
        raise RuntimeError("Turkish C4/mC4 parquet dataset could not be loaded")
        
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Yields:
            (x, H, y) tuples.
            x: (N,) int32
            H: (N, M) float32
            y: (N,) int32
        """
        for example in self.dataset:
            text = example['text']
            
            # Tokenize
            # We request max_seq_len + 1 because we need one extra token for the target of the last input token?
            # Actually, if we use max_seq_len, we get N tokens. x is N-1, y is N-1.
            # This is fine.
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_len,
                padding="max_length",
                return_tensors="np"
            )
            
            ids = encoded["input_ids"][0].tolist()
            
            x, H, y = text_to_hypergraph(ids, self.max_seq_len)
            
            yield x, H, y
