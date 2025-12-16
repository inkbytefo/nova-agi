
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import os

class HDCTTokenizer:
    """
    Hypergraph-Dynamic Chunking Tokenizer (HDCT) v2.0 - Semantic-Topo Tokenizer
    
    A hybrid tokenizer designed for AGI architectures (NovaNet).
    It combines character-level fallback with common morphological roots.
    Critically, it produces 'Initial Topology Hints' (Hyperedges) representing
    linguistic structures (words, morphemes) to be injected into the Hypergraph.
    """
    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        self.next_id = 0
        self.vocab: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        
        # 1. Special Tokens
        self._add_token("<PAD>") # 0
        self._add_token("<UNK>") # 1
        self._add_token("<BOS>") # 2
        self._add_token("<EOS>") # 3
        self._add_token(" ")     # 4
        
        # 2. Character Level (Fallback)
        # Turkish + ASCII
        chars = "abcçdefgğhıijklmnoöpqrsştuüvwxyzABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVWXYZ0123456789.,!?'\"-():;/"
        for char in chars:
            self._add_token(char)
            
        # 3. Common Roots/Morphemes (Simulation of Statistical Corpus)
        # In a real scenario, this would be loaded from a computed vocab file.
        self.roots = self._load_common_roots()
        for root in self.roots:
            self._add_token(root)
            
    def _add_token(self, token: str):
        if token not in self.vocab:
            self.vocab[token] = self.next_id
            self.id_to_char[self.next_id] = token
            self.next_id += 1
            
    def _load_common_roots(self) -> Set[str]:
        """
        Loads a set of common Turkish roots and suffixes.
        Ideally, this comes from BPE merge files or Unigram prob tables.
        Here we use a small curated list for the 'Inductive Bias'.
        """
        # A small subset of common Turkish roots and suffixes for demonstration
        roots = {
            # Pronouns/Common
            "ben", "sen", "o", "biz", "siz", "onlar", "bu", "şu",
            "ve", "ile", "ama", "fakat", "için", "çünkü",
            # Verbs (Roots)
            "yap", "gel", "git", "al", "ver", "bak", "gör", "bil", "dur", "koş",
            "konuş", "yaz", "oku", "çalış", "sev", "iste",
            # Nouns
            "insan", "yapay", "zeka", "zaman", "gün", "yıl", "iş", "şey", "hayat",
            "dünya", "bilgi", "veri", "model", "sistem", "türk", "türkiye",
            # Suffixes (approximation, as standalone tokens for gluing)
            "ler", "lar", "nin", "nın", "den", "dan", "yi", "yı", "de", "da",
            "yor", "cek", "cak", "miş", "mış", "dı", "di"
        }
        return roots

    def _decompose(self, word: str) -> List[str]:
        """
        Decomposes a word into [root, suffix, ...] or [char, char, ...]
        Simple greedy matching for demonstration.
        """
        tokens = []
        remaining = word
        
        while remaining:
            longest_match = ""
            # Try to find the longest prefix that is a known root
            for i in range(len(remaining), 0, -1):
                chunk = remaining[:i]
                if chunk in self.roots:
                    longest_match = chunk
                    break
            
            if longest_match:
                tokens.append(longest_match)
                remaining = remaining[len(longest_match):]
            else:
                # Fallback to character
                tokens.append(remaining[0])
                remaining = remaining[1:]
                
        return tokens

    def encode_with_topology(self, text: str) -> Tuple[List[int], List[List[int]]]:
        """
        Tokenizes text and generates Initial Topology (Hyperedges).
        
        Returns:
            token_ids: List of integer IDs
            edges: List of edges, where each edge is a list of node indices relative to this sequence.
                   e.g. [[0, 1, 2], [3, 4]] means first word covers nodes 0,1,2, second covers 3,4.
        """
        words = text.split(" ")
        all_ids = []
        edges = [] 
        
        # BOS
        all_ids.append(self.vocab["<BOS>"])
        # Global index tracks the position in the final token sequence
        # Start at 1 because 0 is BOS
        global_idx = 1
        
        for i, word in enumerate(words):
            if not word: 
                continue
                
            # Decompose word
            word_tokens = self._decompose(word)
            word_ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in word_tokens]
            
            start_node = global_idx
            all_ids.extend(word_ids)
            global_idx += len(word_ids)
            end_node = global_idx
            
            # Create a hyperedge for the word if it consists of multiple tokens
            # This tells the model: "These tokens form a single semantic unit"
            if len(word_ids) > 1:
                # Indices in the current sequence
                word_edge = list(range(start_node, end_node))
                edges.append(word_edge)
            
            # Add space after word if not the last word
            if i < len(words) - 1:
                all_ids.append(self.vocab[" "])
                global_idx += 1
                
        # EOS
        all_ids.append(self.vocab["<EOS>"])
        
        return all_ids, edges

    def encode(self, text: str, max_length: int = None, padding: str = "max_length", return_tensors: str = "np") -> Dict:
        """
        Standard interface for compatibility. 
        Note: This discards the topology info unless we change the return signature standard.
        For NovaNet specific pipelines, use `encode_with_topology`.
        """
        ids, _ = self.encode_with_topology(text)
        
        # Truncation
        if max_length and len(ids) > max_length:
            ids = ids[:max_length]
            ids[-1] = self.vocab["<EOS>"]
            
        # Padding
        if max_length and padding == "max_length":
            if len(ids) < max_length:
                ids.extend([self.vocab["<PAD>"]] * (max_length - len(ids)))
                
        if return_tensors == "np":
            return {"input_ids": np.array([ids], dtype=np.int32)}
        
        return {"input_ids": ids}

    def decode(self, token_ids: List[int]) -> str:
        chars = []
        for tid in token_ids:
            if tid in self.id_to_char:
                char = self.id_to_char[tid]
                if char in ["<PAD>", "<BOS>", "<EOS>"]:
                    continue
                chars.append(char)
            else:
                pass 
        return "".join(chars)
    
    def __call__(self, text: str, **kwargs):
        return self.encode(text, **kwargs)

# Alias for backward compatibility
HypergraphTokenizer = HDCTTokenizer
