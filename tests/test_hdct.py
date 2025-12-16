
import sys
from unittest.mock import MagicMock

# Mock datasets before importing nova.data.text_stream
sys.modules["datasets"] = MagicMock()
sys.modules["transformers"] = MagicMock()

import pytest
import numpy as np
from nova.data.tokenizer import HDCTTokenizer
from nova.data.text_stream import text_to_hypergraph

def test_hdct_tokenizer_basics():
    tokenizer = HDCTTokenizer(vocab_size=1000)
    
    # Test vocab loaded
    assert "yap" in tokenizer.roots
    assert "ler" in tokenizer.roots
    assert tokenizer.vocab["<BOS>"] == 2
    
    # Test simple word decomposition
    # "yapay" -> "yapay" (if in roots) or "yap" + "ay" (if "yap" in roots)
    # "yap" is in roots. "yapay" is in roots.
    # Since greedy match, "yapay" should be matched as a whole if "yapay" is in roots.
    
    tokens = tokenizer._decompose("yapay")
    assert tokens == ["yapay"]
    
    tokens = tokenizer._decompose("yapayzeka")
    # "yapay" in roots, "zeka" in roots
    assert tokens == ["yapay", "zeka"]
    
    tokens = tokenizer._decompose("bilgisayar")
    # "bilgi" in roots. "sayar" not in roots.
    # "bil" in roots. 
    # Logic: try longest prefix. "bilgi" is in roots? Yes.
    # Remaining: "sayar".
    # "sayar" -> "s", "a", "y", "a", "r" (fallback)
    # Expected: ["bilgi", "s", "a", "y", "a", "r"]
    assert tokens[0] == "bilgi"
    assert tokens[1] == "s"

def test_encode_with_topology():
    tokenizer = HDCTTokenizer()
    text = "yapay zeka"
    
    ids, edges = tokenizer.encode_with_topology(text)
    
    # Structure: BOS, yapay, SPACE, zeka, EOS
    # IDs should correspond to these tokens
    assert len(ids) == 5 
    assert ids[0] == tokenizer.vocab["<BOS>"]
    assert ids[1] == tokenizer.vocab["yapay"]
    assert ids[2] == tokenizer.vocab[" "]
    assert ids[3] == tokenizer.vocab["zeka"]
    assert ids[4] == tokenizer.vocab["<EOS>"]
    
    # Edges: "yapay" (1 token) -> No edge?
    # Logic says: if len(word_ids) > 1: create edge.
    # "yapay" is 1 token. "zeka" is 1 token.
    # So edges should be empty?
    assert len(edges) == 0
    
    # Try a word that decomposes: "yapıyor" -> "yap" + "ı" + "yor" ? 
    # "yap" in roots. "ıyor" -> "ı", "y", "o", "r" ?
    # "yor" is in roots (suffix).
    # "yap" (found), remaining "ıyor".
    # "ı" (char). "yor" (found).
    # So "yapıyor" -> "yap", "ı", "yor".
    # This has 3 tokens -> should create 1 edge covering these 3.
    
    text2 = "yapıyor"
    ids2, edges2 = tokenizer.encode_with_topology(text2)
    # BOS, yap, ı, yor, EOS
    # Indices: 0, 1, 2, 3, 4
    # Edge should cover 1, 2, 3
    
    assert len(edges2) == 1
    assert edges2[0] == [1, 2, 3]

def test_text_to_hypergraph_with_topology():
    # Manual setup
    # ids: [100, 1, 2, 3, 200] (BOS, p1, p2, p3, EOS)
    # edge: [1, 2, 3] (indices in ids)
    
    token_ids = [100, 1, 2, 3, 200]
    edges = [[1, 2, 3]]
    max_len = 10
    
    x, H, y = text_to_hypergraph(token_ids, max_len, topology_edges=edges)
    
    # x: [100, 1, 2, 3] (len 4)
    # Nodes 0, 1, 2, 3 correspond to indices 0, 1, 2, 3 in token_ids
    # Edge [1, 2, 3] covers nodes 1, 2, 3 in x.
    
    # Check H shape
    # H_base width?
    # H_semantic width should be 1.
    
    # Verify semantic edge column exists
    # It should be the last column (or one of them)
    # Column should have 1s at rows 1, 2, 3
    
    semantic_col = H[:, -1]
    assert semantic_col[1] == 1.0
    assert semantic_col[2] == 1.0
    assert semantic_col[3] == 1.0
    assert semantic_col[0] == 0.0

if __name__ == "__main__":
    test_hdct_tokenizer_basics()
    test_encode_with_topology()
    test_text_to_hypergraph_with_topology()
    print("All tests passed!")
