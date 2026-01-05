"""
Data preprocessor for formatting and preparing data for training.
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess data for RL training."""
    
    def __init__(self, tokenizer, config):
        """Initialize preprocessor."""
        self.tokenizer = tokenizer
        self.config = config
        
    def format_chat_template(
        self,
        prompt: str,
        response: str,
        system_message: Optional[str] = None
    ) -> str:
        """Format text using chat template."""
        messages = []
        
        if system_message or self.config.system_message:
            messages.append({
                "role": "system",
                "content": system_message or self.config.system_message
            })
        
        messages.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ])
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            return f"{prompt}\n{response}"
    
    def preprocess_preference_data(self, examples: Dict) -> Dict:
        """Preprocess preference data for DPO/ORPO."""
        processed = {
            "prompt": [],
            "chosen": [],
            "rejected": []
        }
        
        prompts = examples[self.config.prompt_column]
        chosen = examples[self.config.chosen_column]
        rejected = examples[self.config.rejected_column]
        
        for prompt, chosen_text, rejected_text in zip(prompts, chosen, rejected):
            if self.config.use_chat_template:
                chosen_formatted = self.format_chat_template(prompt, chosen_text)
                rejected_formatted = self.format_chat_template(prompt, rejected_text)
            else:
                chosen_formatted = f"{prompt}\n{chosen_text}"
                rejected_formatted = f"{prompt}\n{rejected_text}"
            
            processed["prompt"].append(prompt)
            processed["chosen"].append(chosen_formatted)
            processed["rejected"].append(rejected_formatted)
        
        return processed
    
    def preprocess_ppo_data(self, examples: Dict) -> Dict:
        """Preprocess data for PPO training."""
        processed = {"query": []}
        
        queries = examples.get(
            self.config.query_column or self.config.prompt_column,
            []
        )
        
        for query in queries:
            if self.config.use_chat_template:
                messages = [{"role": "user", "content": query}]
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    formatted = query
            else:
                formatted = query
            
            processed["query"].append(formatted)
        
        return processed
