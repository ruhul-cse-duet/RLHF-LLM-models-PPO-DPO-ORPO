"""
Utils Package for RL LLM Training
Provides metrics, logging, and helper utilities
"""

from .metrics import (
    compute_rewards,
    compute_advantages,
    compute_dpo_metrics,
    compute_orpo_metrics,
    compute_ppo_metrics,
    log_batch_metrics,
)

from .logger import (
    setup_logger,
    Logger,
    MetricsTracker,
    log_training_step,
    log_evaluation_results,
)

from .helpers import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    format_time,
    count_parameters,
    get_lr,
    AverageMeter,
    EarlyStopping,
)

__all__ = [
    # Metrics
    'compute_rewards',
    'compute_advantages',
    'compute_dpo_metrics',
    'compute_orpo_metrics',
    'compute_ppo_metrics',
    'log_batch_metrics',
    
    # Logger
    'setup_logger',
    'Logger',
    'MetricsTracker',
    'log_training_step',
    'log_evaluation_results',
    
    # Helpers
    'set_seed',
    'get_device',
    'save_checkpoint',
    'load_checkpoint',
    'format_time',
    'count_parameters',
    'get_lr',
    'AverageMeter',
    'EarlyStopping',
]

__version__ = "1.0.0"
