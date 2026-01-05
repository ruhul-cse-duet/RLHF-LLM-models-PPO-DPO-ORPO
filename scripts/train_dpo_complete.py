"""
Complete DPO (Direct Preference Optimization) Training Script
Implements end-to-end DPO training for preference-based alignment
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    Logger,
    MetricsTracker,
    compute_dpo_metrics,
    AverageMeter,
    EarlyStopping,
)


class DPOConfig:
    """Configuration for DPO training"""
    
    def __init__(self):
        # Model settings
        self.model_name = "gpt2"
        self.max_length = 512
        
        # DPO hyperparameters
        self.beta = 0.1  # DPO temperature parameter
        self.reference_free = False  # Use reference-free DPO
        
        # Training settings
        self.num_epochs = 3
        self.batch_size = 4
        self.gradient_accumulation_steps = 4
        self.learning_rate = 5e-7
        self.weight_decay = 0.01
        self.warmup_steps = 100
        self.max_grad_norm = 1.0
        
        # Data settings
        self.dataset_name = "Anthropic/hh-rlhf"
        self.train_split = "train"
        self.eval_split = "test"
        self.max_train_samples = None
        self.max_eval_samples = 1000
        
        # Checkpoint settings
        self.output_dir = "./output/dpo"
        self.save_steps = 500
        self.eval_steps = 500
        self.logging_steps = 10
        
        # Other settings
        self.seed = 42
        self.fp16 = True
        self.device = None
        self.num_workers = 4
        
        # Early stopping
        self.patience = 5
        self.early_stopping_metric = "accuracy"
        
        # Logging
        self.use_wandb = False
        self.use_tensorboard = True
        self.log_dir = "./logs/dpo"


class DPODataset(torch.utils.data.Dataset):
    """Dataset for DPO training with chosen and rejected pairs"""
    
    def __init__(
        self,
        data,
        tokenizer,
        max_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize chosen and rejected responses
        chosen = self.tokenizer(
            item["chosen"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        rejected = self.tokenizer(
            item["rejected"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "chosen_input_ids": chosen["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected["attention_mask"].squeeze(0),
        }


class DPOTrainer:
    """DPO Trainer for preference optimization"""
    
    def __init__(
        self,
        config: DPOConfig,
        model: nn.Module,
        reference_model: nn.Module,
        tokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        logger: Logger,
    ):
        self.config = config
        self.model = model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.logger = logger
        
        # Setup device
        self.device = get_device(config.device)
        self.model.to(self.device)
        self.reference_model.to(self.device)
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Setup scheduler
        total_steps = len(train_dataloader) * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Setup tracking
        self.metrics_tracker = MetricsTracker()
        self.global_step = 0
        self.best_accuracy = 0.0
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            mode="max",
            verbose=True,
        )
        
        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
    
    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities for sequences
        
        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            log_probs: Log probabilities of the sequences
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # Shift for next token prediction
        labels = input_ids[:, 1:]  # Shift labels
        
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding tokens
        mask = (labels != self.tokenizer.pad_token_id).float()
        log_probs_sum = (selected_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        
        return log_probs_sum
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Batch of training data
            
        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        
        # Move batch to device
        chosen_input_ids = batch["chosen_input_ids"].to(self.device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
        rejected_input_ids = batch["rejected_input_ids"].to(self.device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.config.fp16):
            # Compute log probs for policy model
            policy_chosen_logps = self.compute_log_probs(
                self.model, chosen_input_ids, chosen_attention_mask
            )
            policy_rejected_logps = self.compute_log_probs(
                self.model, rejected_input_ids, rejected_attention_mask
            )
            
            # Compute log probs for reference model
            with torch.no_grad():
                reference_chosen_logps = self.compute_log_probs(
                    self.reference_model, chosen_input_ids, chosen_attention_mask
                )
                reference_rejected_logps = self.compute_log_probs(
                    self.reference_model, rejected_input_ids, rejected_attention_mask
                )
            
            # Compute DPO metrics
            metrics = compute_dpo_metrics(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=self.config.beta,
            )
            
            loss = metrics["loss"]
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set
        
        Returns:
            avg_metrics: Averaged evaluation metrics
        """
        self.model.eval()
        eval_tracker = MetricsTracker()
        
        pbar = tqdm(self.eval_dataloader, desc="Evaluating")
        for batch in pbar:
            # Move batch to device
            chosen_input_ids = batch["chosen_input_ids"].to(self.device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
            rejected_input_ids = batch["rejected_input_ids"].to(self.device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
            
            # Compute log probs
            policy_chosen_logps = self.compute_log_probs(
                self.model, chosen_input_ids, chosen_attention_mask
            )
            policy_rejected_logps = self.compute_log_probs(
                self.model, rejected_input_ids, rejected_attention_mask
            )
            
            reference_chosen_logps = self.compute_log_probs(
                self.reference_model, chosen_input_ids, chosen_attention_mask
            )
            reference_rejected_logps = self.compute_log_probs(
                self.reference_model, rejected_input_ids, rejected_attention_mask
            )
            
            # Compute metrics
            metrics = compute_dpo_metrics(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=self.config.beta,
            )
            
            eval_tracker.update(metrics)
            pbar.set_postfix({"accuracy": f"{metrics['accuracy']:.4f}"})
        
        return eval_tracker.get_average()
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting DPO training...")
        self.logger.info(f"Total epochs: {self.config.num_epochs}")
        self.logger.info(f"Training samples: {len(self.train_dataloader.dataset)}")
        self.logger.info(f"Evaluation samples: {len(self.eval_dataloader.dataset)}")
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            self.model.train()
            epoch_tracker = MetricsTracker()
            
            pbar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}")
            for step, batch in enumerate(pbar):
                # Training step
                metrics = self.train_step(batch)
                epoch_tracker.update(metrics)
                
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "acc": f"{metrics['accuracy']:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                })
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_metrics = epoch_tracker.get_average(reset=False)
                    self.logger.log_metrics(
                        avg_metrics,
                        step=self.global_step,
                        prefix="train",
                    )
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    self.logger.log_metrics(
                        eval_metrics,
                        step=self.global_step,
                        prefix="eval",
                    )
                    
                    # Update best accuracy
                    if eval_metrics["accuracy"] > self.best_accuracy:
                        self.best_accuracy = eval_metrics["accuracy"]
                        save_checkpoint(
                            self.model,
                            self.optimizer,
                            epoch,
                            self.global_step,
                            metrics["loss"],
                            self.config.output_dir,
                            filename="best_model.pt",
                            scheduler=self.scheduler,
                            metrics=eval_metrics,
                        )
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        self.global_step,
                        metrics["loss"],
                        self.config.output_dir,
                        scheduler=self.scheduler,
                    )
            
            # End of epoch evaluation
            epoch_metrics = self.evaluate()
            self.logger.log_metrics(epoch_metrics, prefix="eval_epoch")
            
            # Early stopping check
            if self.early_stopping(epoch_metrics["accuracy"], epoch):
                self.logger.info("Early stopping triggered!")
                break
        
        self.logger.info("Training completed!")



def prepare_datasets(config: DPOConfig, tokenizer):
    """Prepare training and evaluation datasets"""
    
    # Load dataset
    dataset = load_dataset(config.dataset_name)
    
    # Limit samples if specified
    if config.max_train_samples:
        train_data = dataset[config.train_split].select(range(config.max_train_samples))
    else:
        train_data = dataset[config.train_split]
    
    if config.max_eval_samples:
        eval_data = dataset[config.eval_split].select(range(config.max_eval_samples))
    else:
        eval_data = dataset[config.eval_split]
    
    # Create datasets
    train_dataset = DPODataset(train_data, tokenizer, config.max_length)
    eval_dataset = DPODataset(eval_data, tokenizer, config.max_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    return train_dataloader, eval_dataloader


def main():
    """Main training function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="DPO Training")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="./output/dpo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    
    # Setup config
    config = DPOConfig()
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Set seed
    set_seed(config.seed)
    
    # Setup logger
    logger = Logger(
        log_dir=config.log_dir,
        experiment_name="dpo_training",
        use_wandb=config.use_wandb,
        use_tensorboard=config.use_tensorboard,
    )
    
    # Log config
    logger.log_hyperparameters(vars(config))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load models
    logger.info("Loading models...")
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    reference_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataloader, eval_dataloader = prepare_datasets(config, tokenizer)
    
    # Create trainer
    trainer = DPOTrainer(
