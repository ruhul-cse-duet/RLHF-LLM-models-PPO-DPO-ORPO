"""
Reward model for PPO training.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class RewardModel:
    """Reward model for scoring responses in PPO."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """Initialize reward model."""
        logger.info(f"Loading reward model: {model_name}")
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            torch_dtype=torch.bfloat16,
        ).to(device)
        
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info("Reward model loaded successfully")
    
    @torch.no_grad()
    def get_reward(self, query: str, response: str) -> float:
        """Get reward score for query-response pair."""
        text = f"{query}\n{response}"
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        outputs = self.model(**inputs)
        reward = outputs.logits[0, 0].item()
        
        return reward
    
    @torch.no_grad()
    def get_rewards(self, queries: list, responses: list) -> list:
        """Get reward scores for batch of query-response pairs."""
        rewards = []
        
        for query, response in zip(queries, responses):
            reward = self.get_reward(query, response)
            rewards.append(reward)
        
        return rewards
