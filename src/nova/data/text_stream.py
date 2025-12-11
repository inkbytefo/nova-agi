## Developer: inkbytefo
## Modified: 2025-12-11

from typing import Iterator, Tuple, List
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

from nova.data.hypergraph_builder import build_incremental_H

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
            # H for 1 node: Just 1 global edge (self-loop) or empty?
            # User suggested "if n_nodes >= 2: edges.append(list(range(n_nodes)))" for global.
            # But generate.py handles n=1 case.
            # Let's return a valid H with 1 global edge (self-loop) so model doesn't crash on convolution.
            H = np.ones((1, 1), dtype=np.float32)
            return x, H, y
        else:
            # Empty
            return np.array([], dtype=np.int32), np.zeros((0,0), dtype=np.float32), np.array([], dtype=np.int32)

    x = np.array(token_ids[:-1], dtype=np.int32)
    y = np.array(token_ids[1:], dtype=np.int32)
    
    n_nodes = len(x)
    
    # Use separate fixed counts
    H = build_incremental_H(n_nodes, seq_fixed=8, ctx_fixed=4)
    
    return x, H, y

class TurkishTextStream:
    """
    Streaming dataset loader for Turkish text corpora.
    """
    def __init__(self, dataset_name: str = "ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0", max_seq_len: int = 512, split: str = "train"):
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        
        try:
            self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        except Exception as e:
            print(f"Warning: Failed to load dataset {dataset_name} (split={split}): {e}")
            self.dataset = []

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for example in self.dataset:
            text = example.get('text', '')
            if not text:
                continue
                
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_len,
                padding="max_length",
                return_tensors="np"
            )
            
            ids = encoded["input_ids"][0].tolist()
            
            x, H, y = text_to_hypergraph(ids, self.max_seq_len)
            
            if len(x) > 0:
                yield x, H, y
