
from typing import List, Dict
import numpy as np

class HypergraphTokenizer:
    """
    Tokenizer-free character-level processor for NovaNet HDCT.
    
    Attributes:
        vocab (Dict[str, int]): Character to ID mapping.
        id_to_char (Dict[int, str]): ID to character mapping.
        vocab_size (int): Size of the character vocabulary.
    """
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            " ": 4  # Explicit space handling
        }
        self.id_to_char: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.next_id = 5
        
        # Pre-populate with common ASCII and likely Turkish characters
        common_chars = "abcçdefgğhıijklmnoöpqrsştuüvwxyzABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVWXYZ0123456789.,!?'\"-():;/"
        for char in common_chars:
            if char not in self.vocab:
                self.vocab[char] = self.next_id
                self.id_to_char[self.next_id] = char
                self.next_id += 1

    def encode(self, text: str, max_length: int = None, padding: str = "max_length", return_tensors: str = "np") -> Dict:
        """
        Encodes text into character IDs.
        """
        ids = []
        # BOS
        ids.append(self.vocab["<BOS>"])
        
        for char in text:
            if char not in self.vocab:
                # Dynamic vocab expansion if space allows
                if self.next_id < self.vocab_size:
                    self.vocab[char] = self.next_id
                    self.id_to_char[self.next_id] = char
                    self.next_id += 1
                    ids.append(self.vocab[char])
                else:
                    ids.append(self.vocab["<UNK>"])
            else:
                ids.append(self.vocab[char])
                
        # EOS
        ids.append(self.vocab["<EOS>"])
        
        # Truncation
        if max_length and len(ids) > max_length:
            ids = ids[:max_length]
            ids[-1] = self.vocab["<EOS>"] # Ensure EOS is present if truncated
            
        # Padding
        if max_length and padding == "max_length":
            if len(ids) < max_length:
                ids.extend([self.vocab["<PAD>"]] * (max_length - len(ids)))
                
        if return_tensors == "np":
            return {"input_ids": np.array([ids], dtype=np.int32)}
        
        return {"input_ids": ids}

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes character IDs back to string.
        """
        chars = []
        for tid in token_ids:
            if tid in self.id_to_char:
                char = self.id_to_char[tid]
                if char in ["<PAD>", "<BOS>", "<EOS>"]:
                    continue
                chars.append(char)
            else:
                chars.append("")
        return "".join(chars)
    
    def __call__(self, text: str, **kwargs):
        return self.encode(text, **kwargs)
