## Developer: inkbytefo
## Modified: 2025-12-11

import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
from flax import linen as nn
from flax.jax_utils import replicate, unreplicate
import optax
import wandb
from typing import Any, Dict, Optional
from functools import partial
import os
import time
from transformers import AutoTokenizer

from nova.models.nova import NovaNet
from nova.core.loss import nova_loss
from nova.data.dataset import get_dataloader, get_streaming_dataloader
from nova.core.topology import update_topology
from nova.core.generate import generate_text
from nova.data.curriculum import CurriculumLoader

class TrainState(train_state.TrainState):
    # Add any extra state if needed, for now standard is fine
    pass

class Trainer:
    def __init__(
        self,
        config: Dict,
        model: NovaNet,
        work_dir: str = "checkpoints"
    ):
        self.config = config
        self.model = model
        self.work_dir = work_dir
        
        # Initialize WandB
        if config.get("use_wandb", False):
            wandb.init(
                project=config.get("wandb_project", "nova-agi"),
                config=config,
                name=config.get("run_name", None)
            )
            
        # Detect Execution Mode
        self.num_devices = jax.local_device_count()
        self.is_pmap = self.num_devices > 1
        
        # Select Step Functions
        if self.is_pmap:
            self.train_step = self._train_step_pmap
            self.eval_step = self._eval_step_pmap
        else:
            self.train_step = self._train_step_jit
            self.eval_step = self._eval_step_jit
            
        # Initialize Tokenizer for Generation Monitoring
        # Only if we are likely using text data or if explicit request
        # But per instructions, we just do it.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            self.tokenizer = None
            
        self.val_prompts = ["Yapay zeka", "Türkiye'nin başkenti", "Bilim ve teknoloji", "Merhaba dünya"]

    def create_train_state(self, rng, sample_x, sample_H):
        """Initializes the model and optimizer."""
        variables = self.model.init(rng, sample_x, sample_H)
        params = variables['params']
        
        tx = optax.adamw(learning_rate=self.config["training"]["lr"])
        
        state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=tx
        )
        
        if self.is_pmap:
            state = replicate(state)
            
        return state

    @staticmethod
    def _train_logic(state, x, H, y, mask, alpha, beta, dropout_key, refine_topology):
        """Core training logic (shared by JIT and PMAP)."""
        # 1. Topology Refinement (Optional)
        def refine_H(operand):
            _H, _x, _params = operand
            _, embeddings = state.apply_fn({'params': _params}, _x, _H, train=False)
            new_H = update_topology(_H, embeddings)
            return new_H

        def identity_H(operand):
            return operand[0]

        H_used = jax.lax.cond(
            refine_topology,
            refine_H,
            identity_H,
            (H, x, state.params)
        )
        
        # 2. Gradient Descent Step
        def loss_fn(params):
            logits, embeddings = state.apply_fn(
                {'params': params}, x, H_used, train=True,
                rngs={'dropout': dropout_key}
            )
            loss, metrics = nova_loss(
                params, logits, y, embeddings, mask=mask, alpha=alpha, beta=beta
            )
            return loss, metrics
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)
        metrics["loss"] = loss
        return grads, metrics

    @staticmethod
    @partial(jax.jit, static_argnames=['refine_topology'])
    def _train_step_jit(state, x, H, y, mask, alpha, beta, dropout_key, refine_topology: bool = False):
        grads, metrics = Trainer._train_logic(
            state, x, H, y, mask, alpha, beta, dropout_key, refine_topology
        )
        state = state.apply_gradients(grads=grads)
        return state, metrics

    @staticmethod
    @partial(jax.pmap, axis_name='batch', in_axes=(0, 0, 0, 0, 0, None, None, 0, None))
    def _train_step_pmap(state, x, H, y, mask, alpha, beta, dropout_key, refine_topology: bool = False):
        grads, metrics = Trainer._train_logic(
            state, x, H, y, mask, alpha, beta, dropout_key, refine_topology
        )
        # Aggregate gradients across devices
        grads = jax.lax.pmean(grads, axis_name='batch')
        state = state.apply_gradients(grads=grads)
        # Aggregate metrics
        metrics = jax.lax.pmean(metrics, axis_name='batch')
        return state, metrics

    @staticmethod
    def _eval_logic(state, x, H, y, mask, alpha, beta):
        logits, embeddings = state.apply_fn({'params': state.params}, x, H, train=False)
        loss, metrics = nova_loss(
            state.params, logits, y, embeddings, mask=mask, alpha=alpha, beta=beta
        )
        metrics["loss"] = loss
        return metrics

    @staticmethod
    @partial(jax.jit)
    def _eval_step_jit(state, x, H, y, mask, alpha, beta):
        return Trainer._eval_logic(state, x, H, y, mask, alpha, beta)

    @staticmethod
    @partial(jax.pmap, axis_name='batch', in_axes=(0, 0, 0, 0, 0, None, None))
    def _eval_step_pmap(state, x, H, y, mask, alpha, beta):
        metrics = Trainer._eval_logic(state, x, H, y, mask, alpha, beta)
        return jax.lax.pmean(metrics, axis_name='batch')

    def fit(self, train_data, val_data):
        """Main training loop."""
        rng = jax.random.PRNGKey(self.config["training"]["seed"])
        rng, init_rng = jax.random.split(rng)
        
        # Check if streaming
        is_streaming = not hasattr(train_data, "__len__")
        is_curriculum = isinstance(train_data, CurriculumLoader)
        
        # Initialize state with a dummy batch
        if is_streaming:
            # Consume one item to init
            # Note: For streaming, losing one sample is fine.
            sample_batch = next(get_streaming_dataloader(iter(train_data), batch_size=1, num_devices=1))
        else:
            sample_batch = next(get_dataloader(train_data, batch_size=1, shuffle=False, num_devices=1))
            
        sample_x, sample_H, _, _ = sample_batch 
        
        state = self.create_train_state(init_rng, sample_x, sample_H)
        
        epochs = self.config["training"]["epochs"]
        batch_size = self.config["training"]["batch_size"]
        alpha = self.config["training"]["alpha"] # Energy weight
        beta = self.config["training"]["beta"]   # Entropy weight
        
        # Topology update settings
        # Refine every 5 epochs
        
        current_phase_idx = -1
        global_step = 0
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            epoch_metrics = []
            start_time = time.time()
            
            # Determine if we refine topology this epoch
            # User request: "topology refinement at epoch%5==0"
            refine_topology = (epoch % 5 == 0)
            
            # Curriculum Update
            # Create new loader with current epoch to update ratios
            if is_curriculum:
                print(f"Epoch {epoch}: Updating CurriculumLoader...")
                # We assume train_data was initialized with config compatible with re-init
                # Since we don't have the original config here easily, we create a new one 
                # OR we just instantiate it if we know the class.
                # Ideally, we should pass the config to fit or store it.
                # For now, we assume standard init parameters or rely on the fact that 
                # the user creates the loader.
                # BUT the user said: "train.py'ye ekle: Her epoch'ta yeni loader oluştur"
                # So we must recreate it.
                train_loader_instance = CurriculumLoader(epoch=epoch, max_seq_len=self.config["dataset"]["max_seq_len"])
                
                if is_streaming:
                     steps_per_epoch = self.config["training"].get("steps_per_epoch", 1000)
                     train_loader = get_streaming_dataloader(
                         iter(train_loader_instance), batch_size, num_devices=self.num_devices
                     )
                else:
                     # Fallback if curriculum is somehow not streaming (unlikely)
                     train_loader = get_dataloader(
                        train_loader_instance, batch_size, shuffle=True, seed=epoch, 
                        drop_last=True, num_devices=self.num_devices
                     )
            else:
                # Standard Data Loading
                if is_streaming:
                     steps_per_epoch = self.config["training"].get("steps_per_epoch", 1000)
                     train_loader = get_streaming_dataloader(
                         iter(train_data), batch_size, num_devices=self.num_devices
                     )
                else:
                     train_loader = get_dataloader(
                        train_data, batch_size, shuffle=True, seed=epoch, 
                        drop_last=True, num_devices=self.num_devices
                     )
            
            step_count = 0
            for x, H, y, mask in train_loader:
                if is_streaming and step_count >= steps_per_epoch:
                    break
                
                # Split RNG for dropout
                rng, dropout_key = jax.random.split(rng)
                
                if self.is_pmap:
                    # Split dropout key across devices
                    dropout_keys = jax.random.split(dropout_key, self.num_devices)
                    
                    state, metrics = self.train_step(
                        state, x, H, y, mask, alpha, beta, dropout_keys, 
                        refine_topology
                    )
                else:
                    state, metrics = self.train_step(
                        state, x, H, y, mask, alpha, beta, dropout_key, 
                        refine_topology
                    )
                epoch_metrics.append(metrics)
                step_count += 1
                global_step += 1
            
            # Aggregate Training Metrics
            if self.is_pmap:
                # Unreplicate one instance since they are identical
                epoch_metrics = [unreplicate(m) for m in epoch_metrics]

            train_metrics = jax.tree_util.tree_map(lambda *args: jnp.mean(jnp.stack(args)), *epoch_metrics)
            
            # Validation Loop
            val_metrics_list = []
            if is_streaming:
                # Use a small number of steps for validation
                val_steps = self.config["training"].get("val_steps", 100)
                val_loader = get_streaming_dataloader(
                    iter(val_data), batch_size, num_devices=self.num_devices
                )
            else:
                val_loader = get_dataloader(
                    val_data, batch_size, shuffle=False, 
                    drop_last=True, num_devices=self.num_devices
                )
            
            val_step_count = 0
            for x, H, y, mask in val_loader:
                if is_streaming and val_step_count >= val_steps:
                    break
                metrics = self.eval_step(state, x, H, y, mask, alpha, beta)
                val_metrics_list.append(metrics)
                val_step_count += 1
            
            val_metrics = {}
            if val_metrics_list:
                if self.is_pmap:
                    val_metrics_list = [unreplicate(m) for m in val_metrics_list]
                val_metrics = jax.tree_util.tree_map(lambda *args: jnp.mean(jnp.stack(args)), *val_metrics_list)
            
            end_time = time.time()
            
            # Logging
            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
            log_dict["epoch"] = epoch
            log_dict["time"] = end_time - start_time
            if refine_topology:
                log_dict["topology/refined"] = 1.0
            
            if self.config.get("use_wandb", False):
                wandb.log(log_dict)
            
            refine_status = " [Topo-Refined]" if refine_topology else ""
            print(f"Epoch {epoch}{refine_status} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics.get('loss', 0.0):.4f}")
            
            # Checkpointing and Generation
            if epoch % 10 == 0 or epoch == epochs:
                ckpt_dir = os.path.join(self.work_dir, f"epoch_{epoch}")
                # If pmap, we must unreplicate state before saving
                save_state = unreplicate(state) if self.is_pmap else state
                checkpoints.save_checkpoint(ckpt_dir=self.work_dir, target=save_state, step=epoch, overwrite=True)
                
                # Generation Monitoring
                if self.tokenizer is not None:
                    print(f"\n--- Generating Text at Epoch {epoch} ---")
                    
                    # Get inference params
                    infer_params = save_state.params
                    
                    for prompt in self.val_prompts:
                        gen_text = generate_text(
                            model=self.model,
                            params=infer_params,
                            tokenizer=self.tokenizer,
                            prompt=prompt,
                            max_new_tokens=20,
                            temperature=0.7
                        )
                        print(f"Gen: '{prompt}' -> '{gen_text}'")
                        
                        if self.config.get("use_wandb", False):
                            # Log as simple text or HTML
                            wandb.log({f"gen/{prompt}": gen_text}, step=epoch)
                
        print("Training finished.")
        return state
