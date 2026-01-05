"""
PPO (Proximal Policy Optimization) Trainer for LLM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .base_trainer import BaseTrainer


class PPOTrainer(BaseTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO).
    Implements the PPO algorithm for fine-tuning language models with RL.
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        reward_model_name: Optional[str] = None,
        ppo_epochs: int = 4,
        clip_range: float = 0.2,
        vf_coef: float = 0.1,
        kl_penalty: float = 0.01,        entropy_coef: float = 0.01,
        gamma: float = 1.0,
        lam: float = 0.95,
        **kwargs
    ):
        """
        Initialize PPO trainer.
        
        Args:
            model_name: Model to train
            output_dir: Output directory
            reward_model_name: Optional reward model
            ppo_epochs: Number of PPO update epochs
            clip_range: PPO clipping parameter
            vf_coef: Value function coefficient
            kl_penalty: KL divergence penalty
            entropy_coef: Entropy bonus coefficient
            gamma: Discount factor
            lam: GAE lambda parameter
        """
        super().__init__(model_name, output_dir, **kwargs)
        
        self.ppo_epochs = ppo_epochs
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.kl_penalty = kl_penalty
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        
        # Reference model for KL penalty
        self.ref_model = None
        
        # Value head
        self.value_head = None
        
        # Reward model
        self.reward_model_name = reward_model_name
        self.reward_model = None
        
        print(f"PPO Trainer initialized with clip_range={clip_range}, kl_penalty={kl_penalty}")
    
    def load_model(self):
        """Load policy model, reference model, and value head."""
        super().load_model()
        
        # Create reference model (frozen copy)
        from copy import deepcopy
        self.ref_model = deepcopy(self.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Create value head
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)
        self.value_head = self.value_head.to(self.device)
        
        # Load reward model if provided
        if self.reward_model_name:
            from transformers import AutoModelForSequenceClassification
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.reward_model_name
            )
            self.reward_model = self.reward_model.to(self.device)
            self.reward_model.eval()
            for param in self.reward_model.parameters():
                param.requires_grad = False
    
    def compute_rewards(self, input_ids: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """
        Compute rewards for generated responses.
        
        Args:
            input_ids: Input token IDs
            responses: Generated response token IDs
            
        Returns:
            Reward tensor
        """
        if self.reward_model is not None:
            # Use reward model
            full_sequences = torch.cat([input_ids, responses], dim=1)
            with torch.no_grad():
                rewards = self.reward_model(full_sequences).logits.squeeze(-1)
        else:
            # Simple heuristic reward based on length and diversity
            length_reward = (responses != self.tokenizer.pad_token_id).sum(dim=1).float()
            length_reward = torch.clamp(length_reward / 50.0, 0, 1)
            rewards = length_reward
        
        return rewards
    
    def compute_advantages(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GAE (Generalized Advantage Estimation).
        
        Args:
            rewards: Reward tensor
            values: Value estimates
            masks: Attention masks
            
        Returns:
            Advantages and returns
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
            
            delta = rewards[:, t] + self.gamma * next_value * masks[:, t] - values[:, t]
            gae = delta + self.gamma * self.lam * masks[:, t] * gae
            advantages[:, t] = gae
            returns[:, t] = advantages[:, t] + values[:, t]
        
        return advantages, returns
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss.
        
        Args:
            batch: Batch containing input_ids, attention_mask, old_log_probs, 
                   advantages, returns, old_values
                   
        Returns:
            Dictionary with loss components
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        old_values = batch['old_values']
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(log_probs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        # Value predictions
        values = self.value_head(hidden_states).squeeze(-1)
        
        # Policy loss (PPO clip)
        ratio = torch.exp(selected_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values[:, :-1], returns)
        
        # KL divergence with reference model
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits
        
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        kl_div = (torch.exp(selected_log_probs) * (selected_log_probs - 
                  torch.gather(ref_log_probs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1))).mean()
        
        # Entropy bonus
        probs = F.softmax(logits[:, :-1], dim=-1)
        entropy = -(probs * log_probs[:, :-1]).sum(dim=-1).mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     self.vf_coef * value_loss + 
                     self.kl_penalty * kl_div - 
                     self.entropy_coef * entropy)
        
        return {
            'loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'kl_div': kl_div,
            'entropy': entropy
        }
    
    def train(self, train_dataset, eval_dataset=None):
        """
        Train the model using PPO.
        
        Args:
            train_dataset: Training dataset with prompts
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
        
        print(f"Starting PPO training for {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            epoch_metrics = {'policy_loss': [], 'value_loss': [], 'kl_div': [], 'entropy': []}
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Generate responses
                with torch.no_grad():
                    self.model.eval()
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=self.max_length,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    
                    responses = outputs[:, input_ids.shape[1]:]
                    
                    # Compute rewards
                    rewards = self.compute_rewards(input_ids, responses)
                    
                    # Get values and log probs from current policy
                    policy_outputs = self.model(input_ids=outputs, output_hidden_states=True)
                    logits = policy_outputs.logits
                    log_probs = F.log_softmax(logits, dim=-1)
                    old_log_probs = torch.gather(log_probs[:, :-1], 2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1)
                    
                    hidden_states = policy_outputs.hidden_states[-1]
                    old_values = self.value_head(hidden_states).squeeze(-1)[:, :-1]
                    
                    # Compute advantages
                    response_rewards = torch.zeros_like(old_log_probs)
                    response_rewards[:, -1] = rewards
                    masks = (outputs[:, 1:] != self.tokenizer.pad_token_id).float()
                    
                    advantages, returns = self.compute_advantages(response_rewards, old_values, masks)
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # PPO update
                self.model.train()
                for ppo_epoch in range(self.ppo_epochs):
                    ppo_batch = {
                        'input_ids': outputs,
                        'attention_mask': (outputs != self.tokenizer.pad_token_id).long(),
                        'old_log_probs': old_log_probs.detach(),
                        'advantages': advantages.detach(),
                        'returns': returns.detach(),
                        'old_values': old_values.detach()
                    }
                    
                    loss_dict = self.compute_loss(ppo_batch)
                    loss = loss_dict['loss']
                    
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()
                    
                    if (ppo_epoch + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.global_step += 1
                    
                    # Log metrics
                    for key in ['policy_loss', 'value_loss', 'kl_div', 'entropy']:
                        epoch_metrics[key].append(loss_dict[key].item())
                
                if self.global_step % self.logging_steps == 0:
                    metrics = {k: np.mean(v[-self.logging_steps:]) for k, v in epoch_metrics.items() if v}
                    self.log_metrics(metrics, self.global_step)
                    progress_bar.set_postfix(metrics)
                
                if self.global_step % self.save_steps == 0:
                    save_dir = f"{self.output_dir}/checkpoint-{self.global_step}"
                    self.save_checkpoint(save_dir, metrics)
            
            # End of epoch
            avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items() if v}
            print(f"\nEpoch {epoch+1} completed. Avg metrics: {avg_metrics}")
        
        # Final save
        self.save_checkpoint(f"{self.output_dir}/final", avg_metrics)
        print("Training completed!")
    
    def _collate_fn(self, batch):
        """Collate function for dataloader."""
        prompts = [item['prompt'] if isinstance(item, dict) else item for item in batch]
        
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length // 2,  # Leave room for generation
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
