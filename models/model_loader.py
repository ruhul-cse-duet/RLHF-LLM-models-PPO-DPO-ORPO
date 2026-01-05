"""
Model loader for LLM initialization with quantization and LoRA.
"""
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, get_peft_model

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and prepare models for training."""
    
    def __init__(self, config):
        """Initialize model loader."""
        self.config = config
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with quantization."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.model_revision
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Get quantization config
        quantization_config = self.config.get_quantization_config()
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            torch_dtype=self.config.get_torch_dtype(),
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.model_revision,
            use_cache=self.config.use_cache
        )
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Prepare for k-bit training if quantized
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(model)
        
        logger.info("Model loaded successfully")
        return model, tokenizer
    
    def apply_peft(self, model, peft_config):
        """Apply PEFT (LoRA) to model."""
        logger.info("Applying LoRA...")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        logger.info("LoRA applied successfully")
        return model
