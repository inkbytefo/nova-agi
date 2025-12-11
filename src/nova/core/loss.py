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
    n = logits.shape[0]
    
    if mask is None:
        mask = jnp.ones((n,), dtype=jnp.float32)
        
    # Ensure mask is binary (0.0 or 1.0)
    mask = (mask > 0.5).astype(jnp.float32)
    valid_count = mask.sum() + 1e-8
    
    # 1. Task Loss (Cross Entropy)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    task_loss = (ce * mask).sum() / valid_count
    
    # 2. Diversity Loss (Maximizing Dissimilarity)
    # We want embeddings to be diverse, i.e., low cosine similarity off-diagonal.
    # sim matrix: (n, n)
    # Normalize embeddings first for true cosine similarity? 
    # The user's snippet used: sim = jnp.clip(embeddings @ embeddings.T, -1.0, 1.0)
    # Assuming embeddings are already normalized or we accept dot product.
    # But let's normalize to be safe and standard for "similarity".
    norm_embeddings = embeddings / (jnp.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-7)
    sim = jnp.clip(norm_embeddings @ norm_embeddings.T, -1.0, 1.0)
    
    # Mask matrix for valid pairs
    mask_mat = mask[:, None] * mask[None, :]
    
    # Off-diagonal only
    eye_mask = 1.0 - jnp.eye(n)
    off_diag_sim = sim * eye_mask * mask_mat
    off_diag_mask = eye_mask * mask_mat
    
    # Average similarity of valid off-diagonal pairs
    # Count of valid pairs: sum(off_diag_mask)
    pair_count = off_diag_mask.sum() + 1e-8
    mean_sim = off_diag_sim.sum() / pair_count
    
    # Diversity Score (Higher is better)
    # We want low similarity.
    # diversity_score = 1.0 - mean_sim
    diversity_score = 1.0 - mean_sim
    
    # Total Loss
    # We subtract diversity_score to maximize it.
    # total = task - beta * diversity
    total_loss = task_loss - beta * diversity_score
    
    return total_loss, {"task_loss": task_loss, "diversity_score": diversity_score, "mean_sim": mean_sim}

# Alias for backward compatibility
thermodynamic_loss = nova_loss
