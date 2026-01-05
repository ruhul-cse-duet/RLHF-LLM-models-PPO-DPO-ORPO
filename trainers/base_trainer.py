"""
Base Trainer class for all RL training methods.
"""

import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from torch.utils.data import DataLoader
import json
from pathlib import Path


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    Provides common functionality for PPO, DPO, and ORPO trainers.
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_length: int = 512,
        num_epochs: int = 3,        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        device: Optional[str] = None,
        use_fp16: bool = False,
        **kwargs
    ):
        """
        Initialize base trainer.
        
        Args:
            model_name: Hugging Face model name or path
            output_dir: Directory to save checkpoints and logs
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            gradient_accumulation_steps: Steps to accumulate gradients
            max_length: Maximum sequence length
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_fp16: Use mixed precision training
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.use_fp16 = use_fp16
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf')
        
        print(f"Trainer initialized with device: {self.device}")
    
    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None
        )
        
        if not torch.cuda.is_available() or not self.use_fp16:
            self.model = self.model.to(self.device)
        
        print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),            eps=1e-8,
            weight_decay=0.01
        )
        
        # Calculate total training steps
        # This will be updated in the train method
        total_steps = 1000  # Placeholder
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
    
    def save_checkpoint(self, save_dir: str, metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
        }
        
        if metrics:
            state['metrics'] = metrics
        
        torch.save(state, os.path.join(save_dir, 'trainer_state.pt'))
        
        print(f"Checkpoint saved to {save_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_dir}")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = self.model.to(self.device)
        
        # Load training state
        state_path = os.path.join(checkpoint_dir, 'trainer_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state.get('global_step', 0)
            self.current_epoch = state.get('current_epoch', 0)
            self.best_metric = state.get('best_metric', float('inf'))
            
            if self.optimizer and 'optimizer_state' in state:
                self.optimizer.load_state_dict(state['optimizer_state'])
            if self.scheduler and 'scheduler_state' in state:
                self.scheduler.load_state_dict(state['scheduler_state'])
    
    @abstractmethod
    def train(self, train_dataset, eval_dataset=None):
        """
        Train the model. Must be implemented by subclasses.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        pass
    
    @abstractmethod
    def compute_loss(self, batch):
        """
        Compute loss for a batch. Must be implemented by subclasses.
        
        Args:
            batch: Input batch
            
        Returns:
            Loss tensor
        """
        pass
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log training metrics."""
        log_str = f"Step {step}: "
        log_str += " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                               for k, v in metrics.items()])
        print(log_str)
