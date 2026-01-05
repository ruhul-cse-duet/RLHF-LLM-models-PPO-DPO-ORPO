"""
Training configurations for DPO, ORPO, and PPO algorithms.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization (DPO) training."""
    
    # Training hyperparameters
    beta: float = 0.1
    learning_rate: float = 5e-7
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # Optimization
    optim: str = "paged_adamw_8bit"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    
    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    output_dir: str = "./outputs/dpo"
    save_total_limit: int = 3
    logging_dir: str = "./logs/dpo"
    
    # Mixed precision
    bf16: bool = True
    fp16: bool = False
    
    # DPO specific
    max_length: int = 2048
    max_prompt_length: int = 1024
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo, kto
    
    # Evaluation
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

@dataclass
class ORPOConfig:
    """Configuration for Odds Ratio Preference Optimization (ORPO) training."""
    
    # Training hyperparameters
    lambda_orpo: float = 0.5
    learning_rate: float = 8e-6
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    
    # Optimization
    optim: str = "paged_adamw_8bit"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    
    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    output_dir: str = "./outputs/orpo"
    save_total_limit: int = 3
    logging_dir: str = "./logs/orpo"
    
    # Mixed precision
    bf16: bool = True
    fp16: bool = False
    
    # ORPO specific
    max_length: int = 2048
    max_prompt_length: int = 1024
    
    # Evaluation
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


@dataclass
class PPOConfig:
    """Configuration for Proximal Policy Optimization (PPO) training."""
    
    # Training hyperparameters
    learning_rate: float = 1.4e-5
    num_train_epochs: int = 1
    batch_size: int = 128
    mini_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    
    # PPO specific
    ppo_epochs: int = 4
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    adap_kl_ctrl: bool = True
    
    # Generation settings
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = True
    
    # Optimization
    optim: str = "adamw"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Logging and checkpointing
    log_with: str = "tensorboard"
    logging_steps: int = 10
    save_freq: int = 100
    output_dir: str = "./outputs/ppo"
    
    # Reward model
    reward_model_name: Optional[str] = None
    
    # Model settings
    model_max_length: int = 1024
    query_max_length: int = 512
    response_max_length: int = 512
    
    # Mixed precision
    bf16: bool = True
    fp16: bool = False
