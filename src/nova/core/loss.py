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
    
    # Average similarity of valid off-diagonal pairs
    # Count of valid pairs: sum(mask_mat) - sum(mask) (removing diagonal)
    pair_count = mask_mat.sum() - mask.sum() + 1e-8
    mean_sim = off_diag_sim.sum() / pair_count
    
    # We want to MINIMIZE mean_sim (make them diverse).
    # So we add mean_sim to loss?
    # User said: "Diversity = düşük benzerlik -> yüksek çeşitlilik"
    # "total_loss = task_loss - beta * diversity_loss" where diversity_loss is...
    # Wait, user snippet:
    # diversity_loss = off_diag.sum() / ... (which is mean_similarity)
    # total_loss = task_loss - beta * diversity_loss
    # If we subtract mean_similarity, we are encouraging similarity! (Minimizing (Loss - Sim) -> Maximizing Sim).
    #
    # User's text in table: "Diversity loss = -mean_similarity" or "1.0 - mean_similarity".
    # User's code in Section 3: 
    # "total_loss = task_loss - beta * diversity_loss # negatif!"
    # If diversity_loss IS mean_similarity (positive), and we subtract it, we minimize (Task - Beta*Sim).
    # Minimizing -Sim is Maximizing Sim. This encourages collapsing!
    #
    # Let's re-read Table 1 Item 1:
    # "diversity_loss = mean_similarity -> bu aslında similarity_loss!"
    # "Çeşitliliği artırmak istiyorsan negatif olmalı: diversity_loss = -mean_similarity"
    #
    # If I set `diversity_loss = -mean_sim` and `total = task + diversity`, then `total = task - sim`. Min total -> Max Sim. Wrong.
    #
    # If I want high diversity -> Low similarity.
    # I want to MINIMIZE Similarity.
    # So Loss = Task + lambda * Similarity.
    #
    # Let's look at user code in Section 3 again carefully.
    # `total_loss = task_loss - beta * diversity_loss`
    # If `diversity_loss` is `mean_sim` (as calculated in snippet), then `total = task - beta * mean_sim`.
    # Minimizing this maximizes `mean_sim`. This contradicts "Diversity".
    #
    # HOWEVER, if `diversity_loss` represents "Amount of Diversity" (high is good), then `total = task - Diversity` is correct.
    # But the calculation `off_diag.sum() / count` calculates Similarity.
    #
    # Let's look at Table 1 again.
    # "Doğru Mantık: ... diversity_loss = 1.0 - mean_similarity (tercihen ikincisi)"
    # If `diversity_metric = 1 - sim` (High when sim is low).
    # Then `Loss = Task - beta * diversity_metric` -> `Loss = Task - beta * (1 - sim) = Task - beta + beta * sim`.
    # Minimizing this minimizes `sim`. This is correct.
    #
    # So I should define `diversity_metric = 1.0 - mean_sim` or similar.
    #
    # User Code in Section 3:
    # `diversity_loss = off_diag.sum() / ...` (This is Mean Sim)
    # `total_loss = task_loss - beta * diversity_loss`
    # This implies the user might have made a sign error in the snippet or I am misinterpreting "diversity_loss".
    # If the variable is named `diversity_loss`, usually we want to minimize it.
    # If we minimize `diversity_loss` (where `diversity_loss = mean_sim`), we minimize similarity -> Maximize diversity.
    # In that case, `total = task + beta * diversity_loss` (where diversity_loss = sim).
    #
    # But user wrote `total_loss = task_loss - beta * diversity_loss`.
    # AND "negatif!" comment.
    #
    # If user wants to "Apply immediately", I should probably follow the "Correct Logic" column in Table 1 more than the snippet if they conflict, OR fix the sign in the snippet.
    # Table 1: "diversity_loss = -mean_similarity" (make it negative).
    # If `div_loss = -sim`.
    # `Total = Task + div_loss = Task - sim`. Minimizing Total -> Maximizing Sim. Still wrong.
    #
    # Wait.
    # Goal: Diversity Up -> Sim Down.
    # Loss function is minimized.
    # We want to add a term that is small when Sim is small.
    # Term = + Sim.
    # `Loss = Task + beta * Sim`.
    #
    # Let's see what "diversity_loss" means in the User's mind.
    # "Çeşitliliği artırmak istiyorsan negatif olmalı" -> Maybe they mean the GRADIENT direction?
    #
    # Let's go with the safest mathematical interpretation:
    # I want to Penalize Similarity.
    # Penalty = Mean Similarity.
    # Loss = Task + beta * Penalty.
    #
    # User's snippet: `total_loss = task_loss - beta * diversity_loss`.
    # If this structure is fixed, then `diversity_loss` must be something we want to MAXIMIZE (like Entropy or Dissimilarity).
    # `diversity_loss` (as variable name) usually implies something to minimize, but here it is subtracted.
    # So `diversity_loss` variable holds "Diversity Score".
    #
    # If `diversity_loss` variable = `mean_sim` (from snippet), then `total = task - beta * sim`. This maximizes Sim.
    # This MUST be a mistake in the user's snippet or my understanding.
    #
    # Let's check Table 1 again.
    # "diversity_loss = 1.0 - mean_similarity".
    # If `div_score = 1 - sim`.
    # `Total = Task - beta * div_score = Task - beta * (1 - sim) = Task - beta + beta * sim`.
    # Minimizing Total -> Minimizing Sim. CORRECT.
    #
    # So I will calculate `mean_sim`.
    # Then `diversity_score = 1.0 - mean_sim`.
    # Then `total_loss = task_loss - beta * diversity_score`.
    # And return `{"diversity": diversity_score}`.
    
    mean_sim = off_diag_sim.sum() / pair_count
    diversity_score = 1.0 - mean_sim
    
    # We subtract diversity score (rewarding diversity)
    total_loss = task_loss - beta * diversity_score
    
    return total_loss, {"task_loss": task_loss, "diversity_score": diversity_score, "mean_sim": mean_sim}

# Alias for backward compatibility
thermodynamic_loss = nova_loss
