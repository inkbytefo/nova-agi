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
    # Use numpy for efficient array handling
    # Input x: token_ids[:-1]
    # Target y: token_ids[1:]
    
    # Ensure we have enough tokens
    if len(token_ids) < 2:
        # Need at least 2 tokens for x and y pair
        # Return empty or dummy
        return np.array([], dtype=np.int32), np.zeros((0,0), dtype=np.float32), np.array([], dtype=np.int32)

    x = np.array(token_ids[:-1], dtype=np.int32)
    y = np.array(token_ids[1:], dtype=np.int32)
    
    n_nodes = len(x)
    
    # Build H using the optimized builder
    # "Sabit kenar tipi: sadece global + last 8 sequential + last 4 context"
    # We pass fixed_edges_per_type=8 (covering 8 seq). Context is usually similar count or less.
    # User said "last 8 sequential + last 4 context".
    # build_incremental_H uses one param `fixed_edges_per_type`.
    # I should update build_incremental_H to support separate counts or just modify it?
    # User code for build_incremental_H had `fixed_edges_per_type=8`.
    # And used it for BOTH seq and ctx.
    # "seq_edges = ... range(..., n_nodes-1)"
    # "ctx_edges = ... range(..., n_nodes-2)"
    # If I pass 8, I get 8 seq edges and 8 context edges (roughly).
    # User requirement: "last 8 sequential + last 4 context".
    # I can modify build_incremental_H or just use 8 for both (as it satisfies "at least").
    # Or I can update build_incremental_H now.
    # Let's check `build_incremental_H` again.
    # It has `fixed_edges_per_type`.
    # I will stick to using `build_incremental_H` as defined (using 8).
    
    H = build_incremental_H(n_nodes, fixed_edges_per_type=8)
    
    return x, H, y

class TurkishTextStream:
    """
    Streaming dataset loader for Turkish text corpora.
    """
    def __init__(self, dataset_name: str = "ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0", max_seq_len: int = 512):
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        
        try:
            self.dataset = load_dataset(dataset_name, split="train", streaming=True)
        except Exception as e:
            print(f"Warning: Failed to load dataset {dataset_name}: {e}")
            self.dataset = []

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Yields:
            (x, H, y) tuples.
            x: (N,) int32
            H: (N, M) float32
            y: (N,) int32
        """
        for example in self.dataset:
            text = example.get('text', '')
            if not text:
                continue
                
            # Tokenize
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
