from nova.data.tokenizer import HDCTTokenizer
import os
import sys

# Add src to path if needed, though running with PYTHONPATH is better
sys.path.append(os.path.join(os.getcwd(), 'src'))

def check_tokenizer():
    print("Initializing HDCTTokenizer...")
    try:
        tok = HDCTTokenizer() 
        text = "yapay zeka insanlık için önemlidir"
        print(f"Encoding text: '{text}'")
        ids, edges = tok.encode_with_topology(text) 
        
        tokens = [tok.id_to_char.get(i, '') for i in ids]
        print(f"Tokens: {tokens}") 
        print(f"Edges: {edges}")
        
        if len(ids) > 0 and len(edges) > 0:
            print("Tokenizer check: SUCCESS")
        else:
            print("Tokenizer check: FAILED (Empty output)")
            
    except Exception as e:
        print(f"Tokenizer check FAILED with error: {e}")

if __name__ == "__main__":
    check_tokenizer()
