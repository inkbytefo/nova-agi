## Developer: inkbytefo
## Modified: 2025-12-11

import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import jax

# Add src to path to ensure modules are found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from nova.models.nova import NovaNet
from nova.data.dataset import generate_synthetic_data
from nova.data.zinc import load_zinc_subset
from nova.data.text_stream import TurkishTextStream
from datasets import load_dataset
from nova.train import Trainer
from nova.data.curriculum import CurriculumLoader

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Check JAX devices
    num_devices = jax.local_device_count()
    print(f"JAX Local Devices: {num_devices}")
    print(f"Devices: {jax.devices()}")
    
    # Log Execution Mode
    if num_devices > 1:
        print(f"🚀 Execution Mode: MULTI-DEVICE (PMAP) on {num_devices} devices.")
    else:
        print(f"🚗 Execution Mode: SINGLE-DEVICE (JIT).")

    # Adjust batch size for TPU/Multi-GPU
    # If using multiple devices, we want larger batches to saturate cores
    # We assume at least 16 samples per core is decent for small graphs
    if num_devices > 1:
        min_batch_size = num_devices * 16
        if cfg.training.batch_size < min_batch_size:
            print(f"WARNING: Batch size {cfg.training.batch_size} is small for {num_devices} devices.")
            print(f"Scaling batch size to {min_batch_size} for better utilization.")
            # We can't easily mutate cfg if it's frozen, but we pass a dict to Trainer
            # So we will handle this when creating config_dict
    
    # 1. Data Generation
    dataset_cfg = cfg.dataset
    print(f"Dataset: {dataset_cfg.name}")
    
    if dataset_cfg.name == "zinc":
        print("Loading ZINC data...")
        full_data = load_zinc_subset(n_samples=dataset_cfg.num_samples)
        # ZINC loader currently produces 2 features (AtomicNum, IsAromatic)
        # We need to ensure model input_dim matches
        OmegaConf.set_struct(cfg, False)
        cfg.model.input_dim = 2
        OmegaConf.set_struct(cfg, True)
        print(f"Updated model input_dim to {cfg.model.input_dim} for ZINC data.")
    elif dataset_cfg.name in ["c4_tr", "turkish_v2"]:
        mode = dataset_cfg.get("mode", "stream")
        if mode == "curriculum":
            print("Initializing Curriculum Learning stream...")
            # Pass the raw config dict to CurriculumLoader so it sees the new 'sources'
            dataset_dict = OmegaConf.to_container(dataset_cfg, resolve=True)
            train_data = CurriculumLoader(
                max_seq_len=dataset_cfg.get("max_seq_len", 128),
                datasets_config=dataset_dict # Pass full config to handle custom sources
            )
            val_data = TurkishTextStream(
                max_seq_len=dataset_cfg.get("max_seq_len", 128),
                split="validation"
            )
        else:
            print(f"Loading {dataset_cfg.name} Streaming data...")
            train_data = TurkishTextStream(
                max_seq_len=dataset_cfg.get("max_seq_len", 128),
                split="train"
            )
            val_data = TurkishTextStream(
                 max_seq_len=dataset_cfg.get("max_seq_len", 128),
                 split="validation"
            )
        
        # Model config update
        OmegaConf.set_struct(cfg, False)
        cfg.model.input_dim = 1 # Token IDs are scalar integers
        cfg.model.output_dim = 32000 # Vocab size for prediction
        # Ensure we set steps_per_epoch if not present
        if "steps_per_epoch" not in cfg.training:
            cfg.training.steps_per_epoch = 1000
            print(f"Set steps_per_epoch to {cfg.training.steps_per_epoch} for streaming.")
        OmegaConf.set_struct(cfg, True)
        print("Initialized Streaming Datasets.")
    else:
        print("Generating synthetic data...")
        full_data = generate_synthetic_data(
            num_samples=dataset_cfg.num_samples,
            min_nodes=dataset_cfg.min_nodes,
            max_nodes=dataset_cfg.max_nodes,
            min_edges=dataset_cfg.min_edges,
            max_edges=dataset_cfg.max_edges,
            feature_dim=cfg.model.input_dim, # Using input_dim from model config
            seed=cfg.training.seed
        )
    
    # Simple split
    if dataset_cfg.name not in ["c4_tr", "turkish_v2"]:
        # Simple split
        split_idx = int(len(full_data) * 0.8)
        train_data = full_data[:split_idx]
        val_data = full_data[split_idx:]
        
        print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # 2. Model Initialization
    model = NovaNet(
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.layers,
        out_dim=cfg.model.output_dim,
        use_attention=cfg.model.use_attention,
        dropout_rate=cfg.model.dropout,
        vocab_size=32000, # Explicitly setting for Turkish BERT
        embedding_dim=512 # Explicitly setting
    )
    
    # 3. Trainer Initialization
    # Convert DictConfig to primitive dict for easier handling in Trainer
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Apply batch size override if needed
    if num_devices > 1:
        min_batch_size = num_devices * 16
        if config_dict["training"]["batch_size"] < min_batch_size:
            # On second thought, let's keep it simple for now as we are on dev PC
            # config_dict["training"]["batch_size"] = min_batch_size
            # print(f"Overridden batch_size to {min_batch_size}")
            pass
    
    trainer = Trainer(
        config=config_dict,
        model=model,
        work_dir=os.getcwd() # Hydra changes CWD, so this is safe
    )
    
    # 4. Fit
    trainer.fit(train_data, val_data)

if __name__ == "__main__":
    main()
