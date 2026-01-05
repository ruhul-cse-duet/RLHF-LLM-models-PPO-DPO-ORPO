"""
Complete Training Script for DPO (Direct Preference Optimization)

This script demonstrates a full DPO training pipeline from start to finish.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(model_name, use_4bit=True):
    """Setup model and tokenizer with quantization."""
    logger.info(f"Loading model: {model_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # Setup chat template
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'system' %}"
            "{{ '<<SYS>> ' + message['content'] + ' <</SYS>>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + '</s>' }}"
            "{% endif %}"
            "{% endfor %}"
        )
    
    # Model with quantization
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
