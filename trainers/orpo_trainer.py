"""
ORPO (Odds Ratio Preference Optimization) Trainer for LLM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .base_trainer import BaseTrainer


class ORPOTrainer(BaseTrainer):
    """
    Trainer for Odds Ratio Preference Optimization (ORPO).
    Combines SFT and preference alignment in a single stage.
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        lambda_orpo: float = 0.1,
        **kwargs
    ):
        """
        Initialize ORPO trainer.
        
        Args:
            model_name: Model to train
            output_dir: Output directory
            lambda_orpo: Weight for ORPO loss component
        """
        super().__init__(model_name, output_dir, **kwargs)
        
        self.lambda_orpo = lambda_orpo
        
        print(f"ORPO Trainer initialized with lambda_orpo={lambda_orpo}")
    
    def get_batch_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities of labels given logits.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Log probabilities [batch_size]
        """
        # Shift for next token prediction
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        attention_mask = attention_mask[:, 1:]
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        
        # Sum over sequence
        return (per_token_logps * attention_mask).sum(-1)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute ORPO loss (SFT + Odds Ratio preference loss).
        
        Args:
            batch: Batch containing:
                - chosen_input_ids: Preferred responses
                - chosen_attention_mask: Attention mask for chosen
                - rejected_input_ids: Rejected responses
                - rejected_attention_mask: Attention mask for rejected
                
        Returns:
            Dictionary with loss and metrics
        """
        # Get inputs
        chosen_input_ids = batch['chosen_input_ids']
        chosen_attention_mask = batch['chosen_attention_mask']
        rejected_input_ids = batch['rejected_input_ids']
        rejected_attention_mask = batch['rejected_attention_mask']
        
        # Forward pass
        chosen_outputs = self.model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
        rejected_outputs = self.model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
        
        # Compute log probabilities
        chosen_logps = self.get_batch_logps(chosen_outputs.logits, chosen_input_ids, chosen_attention_mask)
        rejected_logps = self.get_batch_logps(rejected_outputs.logits, rejected_input_ids, rejected_attention_mask)
        
        # SFT loss (negative log-likelihood on chosen responses)
        sft_loss = -chosen_logps.mean()
        
        # Compute odds ratio
        log_odds_chosen = chosen_logps - torch.log(-torch.expm1(chosen_logps) + 1e-8)
        log_odds_rejected = rejected_logps - torch.log(-torch.expm1(rejected_logps) + 1e-8)
        log_odds_ratio = log_odds_chosen - log_odds_rejected
        
        # ORPO preference loss
        orpo_loss = -F.logsigmoid(log_odds_ratio).mean()
        
        # Combined loss
        total_loss = sft_loss + self.lambda_orpo * orpo_loss
        
        # Compute metrics
        with torch.no_grad():
            accuracy = (log_odds_ratio > 0).float().mean()
        
        return {
            'loss': total_loss,
            'sft_loss': sft_loss,
            'orpo_loss': orpo_loss,
            'log_odds_ratio': log_odds_ratio.mean(),
            'accuracy': accuracy
        }
    
    def train(self, train_dataset, eval_dataset=None):
        """
        Train the model using ORPO.
        
        Args:
            train_dataset: Training dataset with chosen/rejected pairs
            eval_dataset: Optional evaluation dataset
        """
        self.load_model()
        self.setup_optimizer()
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        print(f"Starting ORPO training for {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            epoch_metrics = {'loss': [], 'sft_loss': [], 'orpo_loss': [], 'accuracy': []}
            
            self.model.train()
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Compute loss
                loss_dict = self.compute_loss(batch)
                loss = loss_dict['loss']
                
                # Backward pass
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                # Update weights
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                # Log metrics
                for key in ['loss', 'sft_loss', 'orpo_loss', 'accuracy']:
                    epoch_metrics[key].append(loss_dict[key].item())
                
                if self.global_step % self.logging_steps == 0:
                    metrics = {k: np.mean(v[-self.logging_steps:]) for k, v in epoch_metrics.items()}
                    self.log_metrics(metrics, self.global_step)
                    progress_bar.set_postfix(metrics)
                
                if self.global_step % self.save_steps == 0:
                    save_dir = f"{self.output_dir}/checkpoint-{self.global_step}"
                    self.save_checkpoint(save_dir, metrics)
            
            # End of epoch
            avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
            print(f"\nEpoch {epoch+1} completed. Avg metrics: {avg_metrics}")
        
        # Final save
        self.save_checkpoint(f"{self.output_dir}/final", avg_metrics)
        print("Training completed!")
    
    def _collate_fn(self, batch):
        """Collate function for ORPO dataloader."""
        prompts = [item['prompt'] for item in batch]
        chosen = [item['chosen'] for item in batch]
        rejected = [item['rejected'] for item in batch]
        
        # Tokenize chosen responses
        chosen_full = [p + c for p, c in zip(prompts, chosen)]
        chosen_encoded = self.tokenizer(
            chosen_full,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize rejected responses
        rejected_full = [p + r for p, r in zip(prompts, rejected)]
        rejected_encoded = self.tokenizer(
            rejected_full,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_encoded['input_ids'],
            'chosen_attention_mask': chosen_encoded['attention_mask'],
            'rejected_input_ids': rejected_encoded['input_ids'],
            'rejected_attention_mask': rejected_encoded['attention_mask']
        }
