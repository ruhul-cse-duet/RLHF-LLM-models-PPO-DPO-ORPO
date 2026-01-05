"""
Helper utilities for RL LLM training
Provides common utilities for training, checkpointing, and device management
"""

import os
import random
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import timedelta


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get compute device (CUDA, MPS, or CPU)
    
    Args:
        device: Device string (cuda/mps/cpu) or None for auto-detect
        
    Returns:
        device: PyTorch device
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    save_dir: str,
    filename: Optional[str] = None,
    scheduler: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
    keep_last_n: int = 5,
) -> str:
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        step: Current step
        loss: Current loss
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename (auto-generated if None)
        scheduler: Optional learning rate scheduler
        metrics: Optional additional metrics
        keep_last_n: Number of recent checkpoints to keep
        
    Returns:
        checkpoint_path: Path to saved checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch{epoch}_step{step}.pt"
    
    checkpoint_path = os.path.join(save_dir, filename)
    
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Clean up old checkpoints
    if keep_last_n > 0:
        cleanup_old_checkpoints(save_dir, keep_last_n)
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint to
        
    Returns:
        checkpoint: Loaded checkpoint dictionary
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
    
    return checkpoint


def cleanup_old_checkpoints(save_dir: str, keep_last_n: int) -> None:
    """
    Remove old checkpoints, keeping only the most recent ones
    
    Args:
        save_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
    """
    checkpoint_files = sorted(
        Path(save_dir).glob("checkpoint_*.pt"),
        key=lambda p: p.stat().st_mtime
    )
    
    if len(checkpoint_files) > keep_last_n:
        for old_checkpoint in checkpoint_files[:-keep_last_n]:
            old_checkpoint.unlink()
            print(f"Removed old checkpoint: {old_checkpoint}")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        formatted: Formatted time string
    """
    return str(timedelta(seconds=int(seconds)))


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        num_params: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        lr: Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics with new value
        
        Args:
            val: New value
            n: Batch size
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'


class EarlyStopping:
    """Early stopping to stop training when metric stops improving"""
    
    def __init__(
        self,
        patience: int = 7,
        mode: str = "min",
        delta: float = 0.0,
        verbose: bool = True,
    ):
        """
        Initialize EarlyStopping
        
        Args:
            patience: Number of epochs to wait before stopping
            mode: "min" or "max" for metric optimization
            delta: Minimum change to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == "min":
            self.monitor_op = lambda x, y: x < y - delta
        else:
            self.monitor_op = lambda x, y: x > y + delta
    
    def __call__(self, metric: float, epoch: int) -> bool:
        """
        Check if training should stop
        
        Args:
            metric: Current metric value
            epoch: Current epoch
            
        Returns:
            should_stop: Whether to stop training
        """
        if self.best_score is None:
            self.best_score = metric
            self.best_epoch = epoch
            if self.verbose:
                print(f"EarlyStopping: Initial best score: {metric:.4f}")
        elif self.monitor_op(metric, self.best_score):
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: Metric improved to {metric:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Triggered! Best was {self.best_score:.4f} at epoch {self.best_epoch}")
        
        return self.early_stop
