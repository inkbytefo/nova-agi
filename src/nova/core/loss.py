## Developer: inkbytefo
## Modified: 2025-12-11

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Float, Array, PyTree
import chex
from typing import Optional, Tuple

def thermodynamic_loss(
    params: PyTree,
    logits: Float[Array, "n k"],
    targets: Array, # Can be Float or Int
    embeddings: Float[Array, "n d"],
    mask: Optional[Float[Array, "n"]] = None,
    alpha: float = 1e-4,  # Energy weight
    beta: float = 1e-2    # Entropy weight
) -> tuple[Float[Array, ""], dict]:
    """
    Computes the Thermodynamic Loss combining Task, Energy, and Entropy.
    
    Args:
        params: Model parameters (PyTree) for Energy calculation.
        logits: Predicted logits.
        targets: Ground truth targets (Float for regression, Int for classification).
        embeddings: Node embeddings for Entropy calculation.
        mask: Optional mask for valid nodes (1.0 for valid, 0.0 for padded).
        alpha: Weight for Energy penalty (L2 regularization).
        beta: Weight for Entropy/Diversity term.
        
    Returns:
        total_loss: Scalar loss value.
        metrics: Dictionary of individual loss components.
    """
    
    if mask is None:
        mask = jnp.ones((logits.shape[0],), dtype=jnp.float32)
        
    # Ensure mask shape matches (n, 1) for broadcasting
    mask_bc = mask[:, None]
    valid_count = jnp.sum(mask) + 1e-7
    
    # 1. Task Loss
    if jnp.issubdtype(targets.dtype, jnp.integer):
        # Cross Entropy for Classification / Token Prediction
        # logits: (n, vocab_size), targets: (n,)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        # ce is (n,)
        task_loss = jnp.sum(ce * mask) / valid_count
    else:
        # Mean Squared Error masked (Regression)
        # targets: (n, k)
        sq_diff = optax.l2_loss(logits, targets)
        task_loss = jnp.sum(sq_diff * mask_bc) / valid_count
    
    # 2. Energy Penalty (L2 Regularization of weights)
    # Sum of squared weights
    leaves = jax.tree_util.tree_leaves(params)
    energy_loss = sum(jnp.sum(jnp.square(x)) for x in leaves)
    
    # 3. Entropy/Diversity Term
    # "Calculate the Cosine Similarity matrix of the final node_embeddings."
    
    # Normalize embeddings
    norm_embeddings = embeddings / (jnp.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-7)
    
    # Cosine Similarity Matrix: S = E @ E.T
    similarity_matrix = jnp.matmul(norm_embeddings, norm_embeddings.T)
    
    # Mask similarity matrix for padding
    # mask_mat[i, j] = 1 if both i and j are valid
    mask_mat = jnp.matmul(mask_bc, mask_bc.T)
    
    # Mask diagonal
    diag_mask = jnp.ones_like(similarity_matrix) - jnp.eye(similarity_matrix.shape[0])
    
    # Combined mask
    final_mask = mask_mat * diag_mask
    
    off_diag_similarity = similarity_matrix * final_mask
    
    # Mean of off-diagonal entries
    # Count valid off-diagonal pairs: valid_count * (valid_count - 1)
    pair_count = valid_count * (valid_count - 1) + 1e-7
    mean_similarity = jnp.sum(off_diag_similarity) / pair_count
    
    diversity_loss = mean_similarity
    
    # Total Loss
    total_loss = task_loss + alpha * energy_loss + beta * diversity_loss
    
    metrics = {
        "loss": total_loss,
        "loss_task": task_loss,
        "loss_energy": energy_loss,
        "loss_diversity": diversity_loss
    }
    
    return total_loss, metrics
