"""
DPO (Direct Preference Optimization) Trainer for LLM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .base_trainer import BaseTrainer


class DPOTrainer(BaseTrainer):
    """
    Trainer for Direct Preference Optimization (DPO).
    Implements DPO algorithm for aligning language models with human preferences.
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",  # "sigmoid" or "hinge" or "ipo"
        **kwargs
    ):
        """
        Initialize DPO trainer.
        
        Args:
            model_name: Model to train
            output_dir: Output directory
            beta: DPO temperature parameter
            label_smoothing: Label smoothing factor
            loss_type: Type of DPO loss ("sigmoid", "hinge", or "ipo")
        """
        super().__init__(model_name, output_dir, **kwargs)
        
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        
        # Reference model (frozen)
        self.ref_model = None
        
        print(f"DPO Trainer initialized with beta={beta}, loss_type={loss_type}")
    
    def load_model(self):
        """Load policy model and reference model."""
        super().load_model()
        
        # Create reference model (frozen copy)
        from copy import deepcopy
        self.ref_model = deepcopy(self.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        print(f"Reference model created and frozen")
    
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
        # Shift logits and labels for next token prediction
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        attention_mask = attention_mask[:, 1:]
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        
        # Sum log probs over sequence length
        return (per_token_logps * attention_mask).sum(-1)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute DPO loss.
        
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
        
        # Forward pass on policy model
        chosen_outputs = self.model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
        rejected_outputs = self.model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
        
        # Compute log probabilities
        policy_chosen_logps = self.get_batch_logps(chosen_outputs.logits, chosen_input_ids, chosen_attention_mask)
        policy_rejected_logps = self.get_batch_logps(rejected_outputs.logits, rejected_input_ids, rejected_attention_mask)
        
        # Forward pass on reference model
        with torch.no_grad():
            ref_chosen_outputs = self.ref_model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
            ref_rejected_outputs = self.ref_model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
            
            ref_chosen_logps = self.get_batch_logps(ref_chosen_outputs.logits, chosen_input_ids, chosen_attention_mask)
            ref_rejected_logps = self.get_batch_logps(ref_rejected_outputs.logits, rejected_input_ids, rejected_attention_mask)
        
        # Compute DPO loss
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios
        
        if self.loss_type == "sigmoid":
            # Standard DPO loss
            loss = -F.logsigmoid(self.beta * logits).mean()
        elif self.loss_type == "hinge":
            # Hinge loss variant
            loss = torch.relu(1 - self.beta * logits).mean()
        elif self.loss_type == "ipo":
            # IPO (Identity Preference Optimization) loss
            loss = (logits - 1 / (2 * self.beta)) ** 2
            loss = loss.mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Compute metrics
        with torch.no_grad():
            chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)
            reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        return {
            'loss': loss,
            'chosen_rewards': chosen_rewards.mean(),
            'rejected_rewards': rejected_rewards.mean(),
            'reward_accuracy': reward_accuracy,
            'reward_margin': (chosen_rewards - rejected_rewards).mean()
        }
    
    def train(self, train_dataset, eval_dataset=None):
        """
        Train the model using DPO.
        
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
        
        print(f"Starting DPO training for {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            epoch_metrics = {'loss': [], 'reward_accuracy': [], 'reward_margin': []}
            
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
                epoch_metrics['loss'].append(loss_dict['loss'].item())
                epoch_metrics['reward_accuracy'].append(loss_dict['reward_accuracy'].item())
                epoch_metrics['reward_margin'].append(loss_dict['reward_margin'].item())
                
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
        """Collate function for DPO dataloader."""
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
