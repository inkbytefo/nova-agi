## Developer: inkbytefo
## Modified: 2025-12-11

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Float, Array, PyTree
import chex
from typing import Optional, Tuple, Dict

def nova_loss(
    params: PyTree, # Kept for interface compatibility, unused for energy
    logits: Float[Array, "n k"],
    targets: Array,
    embeddings: Float[Array, "n d"],
    mask: Optional[Float[Array, "n"]] = None,
    alpha: float = 0.0, # Deprecated/Unused (Energy)
    beta: float = 0.01  # Diversity weight
) -> Tuple[Float[Array, ""], Dict[str, Float[Array, ""]]]:
    """
    Computes the Nova Loss (Task Loss + Diversity Reward).
    
    Refactored to fix logic errors:
    1. Diversity is now rewarded (negative similarity).
    2. Energy penalty is removed (conflicts with AdamW).
    3. Masking is strictly binary.
    
    Args:
        params: Model parameters (unused, kept for API compatibility).
        logits: Predicted logits (n, vocab_size).
        targets: Target indices (n,).
        embeddings: Node embeddings (n, d) for diversity calculation.
        mask: Valid token mask (n,).
        alpha: Unused (Energy weight).
        beta: Weight for Diversity term (subtracted from loss).
        
    Returns:
        total_loss: Scalar loss.
        metrics: Dictionary of loss components.
    """
    # Determine shapes
    vocab_size = logits.shape[-1]
    # Handle batched input (B, N, D) or single (N, D)
    if logits.ndim == 3:
        b, n, _ = logits.shape
        is_batched = True
    else:
        n = logits.shape[0]
        b = 1
        is_batched = False
    
    if mask is None:
        if is_batched:
            mask = jnp.ones((b, n), dtype=jnp.float32)
        else:
            mask = jnp.ones((n,), dtype=jnp.float32)
        
    # Ensure mask is binary (0.0 or 1.0)
    mask = (mask > 0.5).astype(jnp.float32)
    valid_count = mask.sum() + 1e-8
    
    # 1. Task Loss (Cross Entropy)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    task_loss = (ce * mask).sum() / valid_count
    
    # 2. Diversity Loss (Maximizing Dissimilarity)
    # Normalize embeddings along feature dim (last axis)
    norm_embeddings = embeddings / (jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-7)
    
    # Compute similarity matrix
    # If batched: (B, N, D) @ (B, D, N) -> (B, N, N)
    # If single: (N, D) @ (D, N) -> (N, N)
    sim = jnp.matmul(norm_embeddings, jnp.swapaxes(norm_embeddings, -1, -2))
    sim = jnp.clip(sim, -1.0, 1.0)
    
    # Mask matrix for valid pairs
    # Expand dims for broadcasting mask multiplication
    # mask: (B, N) or (N,)
    mask_col = jnp.expand_dims(mask, axis=-1) # (..., N, 1)
    mask_row = jnp.expand_dims(mask, axis=-2) # (..., 1, N)
    mask_mat = mask_col * mask_row # (..., N, N)
    
    # Off-diagonal only
    # Eye is (N, N). We broadcast to (B, N, N) if needed.
    eye_mask = 1.0 - jnp.eye(n)
    if is_batched:
        # Expand eye to (1, N, N) so it broadcasts to (B, N, N)
        eye_mask = jnp.expand_dims(eye_mask, axis=0)

    off_diag_sim = sim * eye_mask * mask_mat
    off_diag_mask = eye_mask * mask_mat
    
    # Average similarity of valid off-diagonal pairs
    # Count of valid pairs across everything
    pair_count = off_diag_mask.sum() + 1e-8
    mean_sim = off_diag_sim.sum() / pair_count
    
    # Diversity Score (Higher is better)
    diversity_score = 1.0 - mean_sim
    
    # Total Loss
    total_loss = task_loss - beta * diversity_score
    
    return total_loss, {"task_loss": task_loss, "diversity_score": diversity_score, "mean_sim": mean_sim}

# Alias for backward compatibility
thermodynamic_loss = nova_loss
