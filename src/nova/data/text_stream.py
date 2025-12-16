## Developer: inkbytefo
## Modified: 2025-12-11

from typing import Iterator, Tuple, List, Optional
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

from nova.data.hypergraph_builder import build_causal_H
from nova.data.tokenizer import HDCTTokenizer

def text_to_hypergraph(token_ids: List[int], max_seq_len: int, topology_edges: Optional[List[List[int]]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a sequence of token IDs into a CAUSAL hypergraph structure.
    
    Args:
        token_ids: List of token IDs.
        max_seq_len: Maximum sequence length.
        topology_edges: List of edges (lists of node indices) provided by HDCT.
        
    Returns:
        x: Node features (token IDs) of shape (N,).
        H_in: Gather Incidence matrix (N, M).
        H_out: Scatter Incidence matrix (N, M).
        y: Target token ID (next token) of shape (N,).
    """
    # Fix for short sequences (n < 2)
    if len(token_ids) < 2:
        if len(token_ids) == 1:
            # Only 1 token: input x=[id], target y=[pad] (or 0)
            x = np.array([token_ids[0]], dtype=np.int32)
            y = np.array([0], dtype=np.int32) 
            # 1 node, 1 edge (self-loop)
            H_in = np.ones((1, 1), dtype=np.float32)
            H_out = np.ones((1, 1), dtype=np.float32)
            return x, H_in, H_out, y
        else:
            # Empty
            return np.array([], dtype=np.int32), np.zeros((0,0), dtype=np.float32), np.zeros((0,0), dtype=np.float32), np.array([], dtype=np.int32)

    x = np.array(token_ids[:-1], dtype=np.int32)
    y = np.array(token_ids[1:], dtype=np.int32)
    
    n_nodes = len(x)
    
    # Define fixed max_edges for padding (crucial for JIT)
    # Estimate: ~3-4 edges per node (bigram, trigram, semantic)
    max_edges = max_seq_len * 8 
    
    # 1. Base Causal Topology
    H_in, H_out = build_causal_H(n_nodes, max_edges=max_edges, window_sizes=[2, 3])
    
    # 2. Semantic/Structural Topology (Graph Injection)
    if topology_edges:
        # Determine where to start adding new edges
        # Find the last column that has any entries
        edge_sums = H_in.sum(axis=0)
        used_indices = np.where(edge_sums > 0)[0]
        next_idx = used_indices[-1] + 1 if len(used_indices) > 0 else 0
        
        for edge_indices in topology_edges:
            # Filter indices that are within the current input scope x
            valid_indices = [i for i in edge_indices if i < n_nodes]
            
            if len(valid_indices) > 1: # An edge must connect at least 2 nodes
                if next_idx >= max_edges:
                    break
                    
                # Causal Logic: The edge "belongs" to the latest node it connects
                target_node = max(valid_indices)
                
                # Gather from all constituents
                H_in[valid_indices, next_idx] = 1.0
                
                # Scatter to the target (latest) node
                H_out[target_node, next_idx] = 1.0
                
                next_idx += 1
                
    return x, H_in, H_out, y

class TurkishTextStream:
    """
    Streaming dataset loader for Turkish text corpora using HDCT (Hypergraph-Dynamic Chunking Tokenizer).
    """
    def __init__(self, dataset_name: str = "ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0", max_seq_len: int = 512, split: str = "train"):
        self.max_seq_len = max_seq_len
        # Switch to tokenizer-free / character-level
        self.tokenizer = HDCTTokenizer(vocab_size=16000)
        
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
                
            # Tokenize using HDCT (Semantic-Topo)
            # encode_with_topology returns (ids, edges)
            ids, edges = self.tokenizer.encode_with_topology(text)
            
            # Truncate if necessary (naive truncation for stream)
            if len(ids) > self.max_seq_len:
                ids = ids[:self.max_seq_len]
                # Also filter edges that go beyond max_seq_len?
                # text_to_hypergraph handles index checking, so it's safe to pass raw edges
            
            # Pad if needed (optional, text_to_hypergraph handles variable length somewhat, 
            # but usually we want fixed size or batching. Here we just return what we have)
            # For simplicity in this stream, we return the processed sample.
            # If batching is done downstream, padding might happen there.
            
            x, H, y = text_to_hypergraph(ids, self.max_seq_len, topology_edges=edges)
            
            if len(x) > 0:
                yield x, H, y
