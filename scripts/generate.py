## Developer: inkbytefo
## Modified: 2025-12-11

import jax
import jax.numpy as jnp
from flax.training import checkpoints
from transformers import AutoTokenizer
import hydra
from omegaconf import DictConfig
import os

from nova.models.nova import NovaNet
from nova.core.generate import generate_text

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(f"Loading model with config: {cfg.model}")
    
    # 1. Initialize Tokenizer
    tokenizer_name = "dbmdz/bert-base-turkish-cased"
    print(f"Loading tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 2. Initialize Model
    # We need to replicate the init logic to get the correct shapes
    vocab_size = tokenizer.vocab_size
    model = NovaNet(
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        out_dim=vocab_size,
        dropout_rate=0.0, # No dropout during inference
        vocab_size=vocab_size,
        embedding_dim=cfg.model.get("embedding_dim", 256)
    )

    # dummy input for initialization
    rng = jax.random.PRNGKey(0)
    # x shape: (1, seq_len) -> (1, 1) for init
    dummy_x = jnp.zeros((1, 10), dtype=jnp.int32)
    # H shape: (1, seq_len, num_edges) -> let's say (1, 10, 5)
    dummy_H = jnp.zeros((1, 10, 5))
    
    print("Initializing model...")
    variables = model.init(rng, dummy_x, dummy_H)
    
    # 3. Load Checkpoint
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint directory {ckpt_dir} does not exist. Using initialized weights (Random).")
        params = variables["params"]
    else:
        print(f"Restoring checkpoint from {ckpt_dir}...")
        # restore_checkpoint returns the target (which matches the structure of variables)
        # We need to pass a target with the same structure as what we want to load
        # Usually it's train_state, but here we might just want params if we didn't save the whole state structure identically or if we just want to load params into the model structure.
        # In train.py we saved 'save_state' which is a TrainState.
        # So we should probably reconstruct a TrainState or just try to load params if the checkpoint allows.
        # flax.checkpoints.restore_checkpoint can load into a target.
        
        # Let's try to load just the raw dictionary first, or use the variables as target if structure matches.
        # But wait, train.py saves TrainState. variables['params'] is just the params.
        # If we pass target=None, it returns a dictionary. We can extract params from it.
        loaded_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)
        
        if loaded_state is None:
             print("No checkpoint found. Using random weights.")
             params = variables["params"]
        else:
            # Depending on how it was saved, it might be nested under 'params'
            if 'params' in loaded_state:
                params = loaded_state['params']
                print("Params loaded successfully.")
            else:
                print("Could not find 'params' in loaded state. Structure might be different.")
                print(f"Keys: {loaded_state.keys()}")
                params = variables["params"]

    print("\nModel ready. Starting interactive generation loop.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("Prompt: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            if not user_input.strip():
                continue

            generated = generate_text(
                model=model,
                params=params,
                tokenizer=tokenizer,
                prompt=user_input,
                max_new_tokens=50,
                temperature=0.7
            )
            
            print(f"Generated: {generated}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
