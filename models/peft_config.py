"""
PEFT (LoRA) configuration.
"""
from peft import LoraConfig, TaskType
import logging

logger = logging.getLogger(__name__)


def get_peft_config(model_config, task_type: str = "CAUSAL_LM"):
    """Get PEFT configuration for LoRA."""
    
    if not model_config.use_lora:
        logger.info("LoRA disabled")
        return None
    
    logger.info("Creating LoRA configuration...")
    
    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_CLS": TaskType.SEQ_CLS,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM
    }
    
    peft_config = LoraConfig(
        task_type=task_type_map.get(task_type, TaskType.CAUSAL_LM),
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        target_modules=model_config.lora_target_modules,
        bias=model_config.lora_bias,
        inference_mode=False
    )
    
    logger.info(f"LoRA config created: r={model_config.lora_r}, "
                f"alpha={model_config.lora_alpha}")
    
    return peft_config
