"""
Data configuration for dataset loading and preprocessing.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Dataset settings
    dataset_name: str = "Anthropic/hh-rlhf"
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    eval_dataset_split: str = "test"
    
    # Streaming and caching
    streaming: bool = False
    cache_dir: Optional[str] = None
    
    # Data processing
    max_samples: Optional[int] = None
    shuffle: bool = True
    seed: int = 42
    
    # Column mapping
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    
    # Text formatting
    use_chat_template: bool = True
    chat_template: Optional[str] = None
    system_message: Optional[str] = None
    
    # Data splits
    train_test_split: float = 0.9
    validation_split: float = 0.1
    
    # Preprocessing
    remove_unused_columns: bool = True
    num_proc: int = 4
    
    # For PPO
    query_column: Optional[str] = "query"
    response_column: Optional[str] = "response"
