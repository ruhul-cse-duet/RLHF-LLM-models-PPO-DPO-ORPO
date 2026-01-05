"""
Metrics calculation for RL LLM training
Implements metrics for PPO, DPO, and ORPO algorithms
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import defaultdict


def compute_rewards(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reward_model: Optional[torch.nn.Module] = None,
    kl_penalty: float = 0.1,
    reference_logprobs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute rewards for RL training
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        reward_model: Optional reward model for scoring
        kl_penalty: KL divergence penalty coefficient
        reference_logprobs: Log probabilities from reference model
        
    Returns:
        rewards: Computed rewards [batch_size]
    """
    batch_size = logits.size(0)
    
    # Base reward from log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Mask padding tokens
    mask = (labels != -100).float()
    base_reward = (selected_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    
    rewards = base_reward
    
    # Add KL penalty if reference logprobs provided
    if reference_logprobs is not None:
        kl_div = (selected_log_probs - reference_logprobs) * mask
        kl_penalty_value = kl_div.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        rewards = rewards - kl_penalty * kl_penalty_value
    
    # Add reward model score if provided
    if reward_model is not None:
        with torch.no_grad():
            reward_scores = reward_model(logits).squeeze(-1)
            rewards = rewards + reward_scores
    
    return rewards


def compute_advantages(    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE (Generalized Advantage Estimation)
    
    Args:
        rewards: Rewards at each step [batch_size, seq_len]
        values: Value estimates [batch_size, seq_len]
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        advantages: Computed advantages [batch_size, seq_len]
        returns: Computed returns [batch_size, seq_len]
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    gae = 0
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = 0
        else:
            next_value = values[:, t + 1]
        
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        gae = delta + gamma * lam * gae
        advantages[:, t] = gae
        returns[:, t] = gae + values[:, t]
    
    return advantages, returns


def compute_dpo_metrics(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Dict[str, float]:
    """
    Compute DPO (Direct Preference Optimization) metrics
    
    Args:
        policy_chosen_logps: Log probs for chosen responses
        policy_rejected_logps: Log probs for rejected responses
        reference_chosen_logps: Reference log probs for chosen
        reference_rejected_logps: Reference log probs for rejected
        beta: DPO temperature parameter
        
    Returns:
        metrics: Dictionary of DPO metrics
    """
    # Compute policy and reference log ratios
    policy_log_ratios = policy_chosen_logps - policy_rejected_logps
    reference_log_ratios = reference_chosen_logps - reference_rejected_logps
    
    # DPO loss
    logits = beta * (policy_log_ratios - reference_log_ratios)
    loss = -F.logsigmoid(logits).mean()
    
    # Implicit reward
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    
    # Accuracy
    accuracy = (policy_log_ratios > 0).float().mean()
    
    return {
        "loss": loss.item(),
        "accuracy": accuracy.item(),
        "chosen_rewards": chosen_rewards.mean().item(),
        "rejected_rewards": rejected_rewards.mean().item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        "policy_log_ratios": policy_log_ratios.mean().item(),
        "reference_log_ratios": reference_log_ratios.mean().item(),
    }


def compute_orpo_metrics(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    sft_loss: torch.Tensor,
    lambda_param: float = 0.5,
) -> Dict[str, float]:
    """
    Compute ORPO (Odds Ratio Preference Optimization) metrics
    
    Args:
        policy_chosen_logps: Log probs for chosen responses
        policy_rejected_logps: Log probs for rejected responses
        sft_loss: Supervised fine-tuning loss
        lambda_param: Weight for odds ratio loss
        
    Returns:
        metrics: Dictionary of ORPO metrics
    """
    # Compute odds ratio
    log_odds = policy_chosen_logps - policy_rejected_logps
    odds_ratio_loss = -F.logsigmoid(log_odds).mean()
    
    # Combined loss
    total_loss = sft_loss + lambda_param * odds_ratio_loss
    
    # Metrics
    accuracy = (log_odds > 0).float().mean()
    
    return {
        "loss": total_loss.item(),
        "sft_loss": sft_loss.item(),
        "odds_ratio_loss": odds_ratio_loss.item(),
        "accuracy": accuracy.item(),
        "log_odds": log_odds.mean().item(),
        "chosen_logps": policy_chosen_logps.mean().item(),
        "rejected_logps": policy_rejected_logps.mean().item(),
    }


def compute_ppo_metrics(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float = 0.2,
    value_clip_range: float = 0.2,
) -> Dict[str, float]:
    """
    Compute PPO (Proximal Policy Optimization) metrics
    
    Args:
        policy_logprobs: Current policy log probs
        old_logprobs: Old policy log probs
        advantages: Advantage values
        values: Value estimates
        returns: Target returns
        clip_range: PPO clipping range
        value_clip_range: Value function clipping range
        
    Returns:
        metrics: Dictionary of PPO metrics
    """
    # Policy loss with clipping
    ratio = torch.exp(policy_logprobs - old_logprobs)
    policy_loss_1 = -advantages * ratio
    policy_loss_2 = -advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
    
    # Value loss with clipping
    value_loss_unclipped = (values - returns) ** 2
    values_clipped = old_logprobs + torch.clamp(
        values - old_logprobs, -value_clip_range, value_clip_range
    )
    value_loss_clipped = (values_clipped - returns) ** 2
    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
    
    # KL divergence
    kl_div = (old_logprobs - policy_logprobs).mean()
    
    # Clip fraction (how often we clip)
    clip_fraction = ((ratio - 1).abs() > clip_range).float().mean()
    
    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "kl_divergence": kl_div.item(),
        "clip_fraction": clip_fraction.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item(),
        "advantages_mean": advantages.mean().item(),
        "advantages_std": advantages.std().item(),
    }


def log_batch_metrics(
    metrics_dict: Dict[str, float],
    step: int,
    epoch: int,
    prefix: str = "train",
) -> None:
    """
    Log batch-level metrics
    
    Args:
        metrics_dict: Dictionary of metrics to log
        step: Current training step
        epoch: Current epoch
        prefix: Prefix for metric names (train/val/test)
    """
    print(f"\n[{prefix.upper()}] Epoch {epoch}, Step {step}")
    print("-" * 60)
    for key, value in metrics_dict.items():
        print(f"{key:30s}: {value:.6f}")
    print("-" * 60)


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss"""
    return np.exp(loss)


def compute_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute token-level accuracy
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        labels: Ground truth labels [batch_size, seq_len]
        ignore_index: Index to ignore in accuracy calculation
        
    Returns:
        accuracy: Token-level accuracy
    """
    predictions = logits.argmax(dim=-1)
    mask = labels != ignore_index
    correct = (predictions == labels) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple batches
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        aggregated: Aggregated metrics dictionary
    """
    if not metrics_list:
        return {}
    
    aggregated = defaultdict(list)
    for metrics in metrics_list:
        for key, value in metrics.items():
            aggregated[key].append(value)
    
    return {key: np.mean(values) for key, values in aggregated.items()}


def compute_reward_statistics(
    rewards: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute statistics for rewards
    
    Args:
        rewards: Reward values [batch_size]
        
    Returns:
        stats: Dictionary of reward statistics
    """
    return {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "reward_min": rewards.min().item(),
        "reward_max": rewards.max().item(),
        "reward_median": rewards.median().item(),
    }
