"""
Logging utilities for RL LLM training
Provides comprehensive logging and metrics tracking
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List
from collections import defaultdict
import torch


def setup_logger(
    name: str,
    log_dir: str,
    level: int = logging.INFO,
    console: bool = True,
    log_file: bool = True,
) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        console: Whether to log to console
        log_file: Whether to log to file
        
    Returns:
        logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Remove existing handlers
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = os.path.join(log_dir, f'{name}_{timestamp}.log')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file_path}")
    
    return logger


class Logger:
    """Advanced logger with metrics tracking and file output"""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
    ):
        """
        Initialize Logger
        
        Args:
            log_dir: Directory for logs
            experiment_name: Name of the experiment
            use_wandb: Whether to use Weights & Biases
            use_tensorboard: Whether to use TensorBoard
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        # Setup basic logger
        self.logger = setup_logger(
            experiment_name,
            str(self.log_dir),
            level=logging.INFO
        )
        
        # Initialize W&B
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.logger.info("Weights & Biases initialized")
            except ImportError:
                self.logger.warning("wandb not installed, disabling W&B logging")
                self.use_wandb = False
        
        # Initialize TensorBoard
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(self.log_dir / "tensorboard")
                self.logger.info(f"TensorBoard logs at {self.log_dir / 'tensorboard'}")
            except ImportError:
                self.logger.warning("tensorboard not installed, disabling TB logging")
                self.use_tensorboard = False
        
        # Metrics file
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.step_counter = 0
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """
        Log metrics to all configured backends
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step (if None, uses internal counter)
            prefix: Prefix for metric names
        """
        if step is None:
            step = self.step_counter
            self.step_counter += 1
        
        # Add prefix to metrics
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Log to file
        log_entry = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Log to W&B
        if self.use_wandb:
            self.wandb.log(metrics, step=step)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
        
        # Log summary to console
        metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                  for k, v in metrics.items()])
        self.logger.info(f"Step {step} | {metric_str}")
    
    def log_hyperparameters(self, config: Dict[str, Any]) -> None:
        """Log hyperparameters"""
        config_file = self.log_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        if self.use_wandb:
            self.wandb.config.update(config)
        
        self.logger.info(f"Hyperparameters saved to {config_file}")
    
    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
    
    def close(self) -> None:
        """Close all logging backends"""
        if self.use_tensorboard:
            self.tb_writer.close()
        if self.use_wandb:
            self.wandb.finish()


class MetricsTracker:
    """Track and aggregate metrics during training"""
    
    def __init__(self):
        """Initialize MetricsTracker"""
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
    
    def update(self, metrics_dict: Dict[str, float]) -> None:
        """
        Update metrics with new values
        
        Args:
            metrics_dict: Dictionary of metric names and values
        """
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)
    
    def get_average(self, reset: bool = True) -> Dict[str, float]:
        """
        Get average of tracked metrics
        
        Args:
            reset: Whether to reset metrics after averaging
            
        Returns:
            averages: Dictionary of averaged metrics
        """
        averages = {}
        for key, values in self.metrics.items():
            if values:
                averages[key] = sum(values) / len(values)
        
        if reset:
            self.metrics.clear()
        
        return averages
    
    def reset(self) -> None:
        """Reset all tracked metrics"""
        self.metrics.clear()
    
    def save_epoch_metrics(self, epoch: int) -> None:
        """Save current metrics as epoch metrics"""
        avg_metrics = self.get_average(reset=False)
        self.epoch_metrics[epoch] = avg_metrics
    
    def get_best_metric(self, metric_name: str, mode: str = "min") -> tuple:
        """
        Get best value of a metric across epochs
        
        Args:
            metric_name: Name of the metric
            mode: "min" or "max"
            
        Returns:
            (best_value, best_epoch)
        """
        if mode == "min":
            best_epoch = min(self.epoch_metrics.keys(), 
                           key=lambda e: self.epoch_metrics[e].get(metric_name, float('inf')))
        else:
            best_epoch = max(self.epoch_metrics.keys(),
                           key=lambda e: self.epoch_metrics[e].get(metric_name, float('-inf')))
        
        best_value = self.epoch_metrics[best_epoch].get(metric_name)
        return best_value, best_epoch


def log_training_step(
    logger: Logger,
    step: int,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    learning_rate: float,
) -> None:
    """
    Log training step information
    
    Args:
        logger: Logger instance
        step: Current step
        epoch: Current epoch
        loss: Loss value
        metrics: Additional metrics
        learning_rate: Current learning rate
    """
    log_dict = {
        "epoch": epoch,
        "loss": loss,
        "learning_rate": learning_rate,
        **metrics
    }
    logger.log_metrics(log_dict, step=step, prefix="train")


def log_evaluation_results(
    logger: Logger,
    epoch: int,
    eval_loss: float,
    eval_metrics: Dict[str, float],
    best_metric: Optional[float] = None,
) -> None:
    """
    Log evaluation results
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        eval_loss: Evaluation loss
        eval_metrics: Evaluation metrics
        best_metric: Best metric value so far (optional)
    """
    log_dict = {
        "epoch": epoch,
        "eval_loss": eval_loss,
        **eval_metrics
    }
    
    if best_metric is not None:
        log_dict["best_metric"] = best_metric
    
    logger.log_metrics(log_dict, prefix="eval")
    
    # Log to console with formatting
    logger.info("=" * 80)
    logger.info(f"EVALUATION RESULTS - Epoch {epoch}")
    logger.info("=" * 80)
    logger.info(f"Eval Loss: {eval_loss:.4f}")
    for key, value in eval_metrics.items():
        logger.info(f"{key}: {value:.4f}")
    if best_metric is not None:
        logger.info(f"Best Metric: {best_metric:.4f}")
    logger.info("=" * 80)
