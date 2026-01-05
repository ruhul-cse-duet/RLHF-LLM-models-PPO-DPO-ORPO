"""
Data collators for batch preparation.
"""
import torch
from dataclasses import dataclass
from typing import Dict, List, Any
from transformers import PreTrainedTokenizerBase


@dataclass
class DPODataCollator:
    """Collator for DPO/ORPO training data."""
    
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 2048
    max_prompt_length: int = 1024
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepare batch for DPO training."""
        batch = {
            "prompt": [],
            "chosen": [],
            "rejected": []
        }
        
        for feature in features:
            batch["prompt"].append(feature["prompt"])
            batch["chosen"].append(feature["chosen"])
            batch["rejected"].append(feature["rejected"])
        
        return batch


@dataclass
class PPODataCollator:
    """Collator for PPO training data."""
    
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 1024
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepare batch for PPO training."""
        queries = [feature["query"] for feature in features]
        
        # Tokenize queries
        tokenized = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "query": queries
        }
