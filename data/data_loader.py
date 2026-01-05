"""
Data loader for HuggingFace datasets.
"""
from datasets import load_dataset, DatasetDict
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare datasets from HuggingFace."""
    
    def __init__(self, config):
        """Initialize data loader with config."""
        self.config = config
        
    def load_preference_dataset(
        self,
        split: Optional[str] = None
    ) -> Union[DatasetDict, dict]:
        """Load preference dataset for DPO/ORPO training."""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split=split or self.config.dataset_split,
                cache_dir=self.config.cache_dir,
                streaming=self.config.streaming
            )
            
            if self.config.max_samples and not self.config.streaming:
                if isinstance(dataset, DatasetDict):
                    for key in dataset.keys():
                        dataset[key] = dataset[key].select(
                            range(min(self.config.max_samples, len(dataset[key])))
                        )
                else:
                    dataset = dataset.select(
                        range(min(self.config.max_samples, len(dataset)))
                    )
            
            logger.info(f"Dataset loaded successfully")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def load_ppo_dataset(self, split: Optional[str] = None):
        """Load dataset for PPO training."""
        logger.info(f"Loading PPO dataset: {self.config.dataset_name}")
        
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split=split or self.config.dataset_split,
                cache_dir=self.config.cache_dir,
                streaming=self.config.streaming
            )
            
            if self.config.max_samples and not self.config.streaming:
                dataset = dataset.select(
                    range(min(self.config.max_samples, len(dataset)))
                )
            
            logger.info(f"PPO dataset loaded successfully")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading PPO dataset: {e}")
            raise
    
    def split_dataset(self, dataset, test_size: float = 0.1):
        """Split dataset into train and test sets."""
        if isinstance(dataset, DatasetDict):
            return dataset
        
        split_dataset = dataset.train_test_split(
            test_size=test_size,
            seed=self.config.seed,
            shuffle=self.config.shuffle
        )
        
        return split_dataset
