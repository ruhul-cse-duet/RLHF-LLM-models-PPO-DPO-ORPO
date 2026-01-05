"""
Model configuration for LLM training with quantization and LoRA support.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch


@dataclass
class ModelConfig:
    """Configuration for model loading and optimization."""
    
    # Model selection
    model_name: str = "meta-llama/Llama-2-7b-hf"
    model_revision: str = "main"
    trust_remote_code: bool = True
    
    # Quantization settings
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_bias: str = "none"
    
    # Model parameters
    torch_dtype: str = "auto"
    device_map: str = "auto"
    max_seq_length: int = 2048
    
    # Cache and optimization
    use_cache: bool = False
    gradient_checkpointing: bool = True
    
    def get_quantization_config(self):
        """Get BitsAndBytes quantization configuration."""
        from transformers import BitsAndBytesConfig
        
        if not (self.load_in_4bit or self.load_in_8bit):
            return None
            
        compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
        
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )
    
    def get_torch_dtype(self):
        """Get torch dtype for model."""
        if self.torch_dtype == "auto":
            return "auto"
        return getattr(torch, self.torch_dtype)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
