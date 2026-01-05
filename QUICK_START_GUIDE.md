# 🚀 RLHF Quick Start Guide - Complete Training Scripts

## Overview
This guide provides ready-to-run training scripts for all three RL techniques.

---

## ✅ Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (if using gated models)
huggingface-cli login

# Setup Wandb (optional, for experiment tracking)
wandb login
```

---

## 📊 Training Scripts

### 1. DPO Training - Complete Script

**File**: `scripts/train_dpo_complete.py`

```python
#!/usr/bin/env python3
"""
Complete DPO Training Script - Ready to Run
Usage: python scripts/train_dpo_complete.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== CONFIGURATION ==================
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"
OUTPUT_DIR = "./outputs/dpo_mistral"
SAMPLE_RATIO = 0.01  # Use 1% of data for quick testing (change to 1.0 for full training)

# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 5e-5
BETA = 0.1  # DPO temperature parameter

# LoRA configuration
LORA_R = 64
LORA_ALPHA = 64
LORA_DROPOUT = 0.0

# ================== STEP 1: LOAD TOKENIZER ==================
logger.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Setup chat template
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

# ================== STEP 2: LOAD DATASET ==================
logger.info(f"Loading dataset: {DATASET_NAME}")
train_dataset = load_dataset(DATASET_NAME, split="train_prefs")
eval_dataset = load_dataset(DATASET_NAME, split="test_prefs")

# Sample for faster training
if SAMPLE_RATIO < 1.0:
    train_size = int(len(train_dataset) * SAMPLE_RATIO)
    eval_size = int(len(eval_dataset) * SAMPLE_RATIO)
    train_dataset = train_dataset.shuffle(seed=42).select(range(train_size))
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(eval_size))
    logger.info(f"Using {train_size} training examples and {eval_size} eval examples")

# Preprocess dataset - apply chat template
def process_example(example):
    example["chosen"] = tokenizer.apply_chat_template(example["chosen"], tokenize=False)
    example["rejected"] = tokenizer.apply_chat_template(example["rejected"], tokenize=False)
    return example

logger.info("Preprocessing dataset...")
train_dataset = train_dataset.map(process_example, num_proc=4)
eval_dataset = eval_dataset.map(process_example, num_proc=4)

# ================== STEP 3: LOAD MODEL ==================
logger.info(f"Loading model: {MODEL_NAME}")

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager"  # Use "flash_attention_2" for Ampere+ GPUs
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.pad_token_id = tokenizer.pad_token_id

# ================== STEP 4: ADD LORA ADAPTERS ==================
logger.info("Adding LoRA adapters...")
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ================== STEP 5: SETUP TRAINING ARGUMENTS ==================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    fp16=True,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="wandb",  # Change to "none" to disable Wandb
    run_name=f"dpo_mistral_{SAMPLE_RATIO}",
    logging_dir=f"{OUTPUT_DIR}/logs",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# ================== STEP 6: INITIALIZE DPO TRAINER ==================
logger.info("Initializing DPO Trainer...")
trainer = DPOTrainer(
    model=model,
    args=training_args,
    beta=BETA,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=2048,
    max_prompt_length=1024,
    max_target_length=1024,
    loss_type="sigmoid"
)

# ================== STEP 7: TRAIN ==================
logger.info("Starting DPO training...")
trainer.train()

# ================== STEP 8: SAVE MODEL ==================
logger.info(f"Saving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

logger.info("✅ DPO Training completed successfully!")
```

---

### 2. ORPO Training - Complete Script

**File**: `scripts/train_orpo_complete.py`

```python
#!/usr/bin/env python3
"""
Complete ORPO Training Script - Ready to Run
Usage: python scripts/train_orpo_complete.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl.experimental.orpo import ORPOTrainer, ORPOConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== CONFIGURATION ==================
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"
OUTPUT_DIR = "./outputs/orpo_mistral"
SAMPLE_RATIO = 0.01  # Use 1% of data for quick testing

# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 8e-6
LAMBDA_PARAM = 0.1  # ORPO odds ratio weight

# LoRA configuration
LORA_R = 64
LORA_ALPHA = 64

# ================== STEP 1: LOAD TOKENIZER ==================
logger.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Setup chat template
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

# ================== STEP 2: LOAD DATASET ==================
logger.info(f"Loading dataset: {DATASET_NAME}")
train_dataset = load_dataset(DATASET_NAME, split="train_prefs")
eval_dataset = load_dataset(DATASET_NAME, split="test_prefs")

# Sample for faster training
if SAMPLE_RATIO < 1.0:
    train_size = int(len(train_dataset) * SAMPLE_RATIO)
    eval_size = int(len(eval_dataset) * SAMPLE_RATIO)
    train_dataset = train_dataset.shuffle(seed=42).select(range(train_size))
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(eval_size))
    logger.info(f"Using {train_size} training examples and {eval_size} eval examples")

# Preprocess dataset
def process_example(example):
    example["chosen"] = tokenizer.apply_chat_template(example["chosen"], tokenize=False)
    example["rejected"] = tokenizer.apply_chat_template(example["rejected"], tokenize=False)
    return example

logger.info("Preprocessing dataset...")
train_dataset = train_dataset.map(process_example, num_proc=4)
eval_dataset = eval_dataset.map(process_example, num_proc=4)

# ================== STEP 3: LOAD MODEL ==================
logger.info(f"Loading model: {MODEL_NAME}")

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager"
)

# Prepare for training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.pad_token_id = tokenizer.pad_token_id

# ================== STEP 4: ADD LORA ADAPTERS ==================
logger.info("Adding LoRA adapters...")
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ================== STEP 5: SETUP ORPO CONFIGURATION ==================
orpo_config = ORPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    fp16=True,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="wandb",
    run_name=f"orpo_mistral_{SAMPLE_RATIO}",
    # ORPO-specific parameters
    lambda_param=LAMBDA_PARAM,
    max_length=2048,
    max_prompt_length=1024,
    max_completion_length=1024
)

# ================== STEP 6: INITIALIZE ORPO TRAINER ==================
logger.info("Initializing ORPO Trainer...")
trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# ================== STEP 7: TRAIN ==================
logger.info("Starting ORPO training...")
trainer.train()

# ================== STEP 8: SAVE MODEL ==================
logger.info(f"Saving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

logger.info("✅ ORPO Training completed successfully!")
```

---

## 📋 Quick Command Reference

### Run DPO Training
```bash
# Full dataset training
python scripts/train_dpo_complete.py

# Quick test with 1% data (modify SAMPLE_RATIO in script)
python scripts/train_dpo_complete.py
```

### Run ORPO Training
```bash
# Full dataset training
python scripts/train_orpo_complete.py

# Quick test with 1% data
python scripts/train_orpo_complete.py
```

---

## 🎛️ Configuration Cheat Sheet

### Key Hyperparameters

**DPO**:
- `beta`: Temperature (0.1 - 0.5). Higher = more aggressive preference learning
- `learning_rate`: 5e-5 to 1e-4
- `loss_type`: "sigmoid" (default), "hinge", "ipo"

**ORPO**:
- `lambda_param`: Odds ratio weight (0.05 - 0.2). Higher = stronger preference signal
- `learning_rate`: 5e-6 to 1e-5 (lower than DPO)

**LoRA**:
- `r` (rank): 8, 16, 32, 64, 128. Higher = more parameters, better quality
- `lora_alpha`: Usually same as `r` or 2x `r`
- `target_modules`: All attention + MLP layers for best results

---

## 📊 Expected Results

### Training Time (Tesla T4)
- **1% dataset** (~600 examples): ~15 minutes
- **10% dataset** (~6,000 examples): ~2 hours  
- **Full dataset** (~60,000 examples): ~18 hours

### Memory Usage
- **4-bit + LoRA (r=64)**: ~6-8 GB VRAM
- **8-bit + LoRA (r=64)**: ~10-12 GB VRAM
- **Full precision**: 24+ GB VRAM

---

## 🐛 Troubleshooting

### OOM (Out of Memory)
```python
# Reduce batch size
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8  # Keep effective batch size

# Reduce LoRA rank
LORA_R = 16
LORA_ALPHA = 16

# Reduce sequence length
max_length = 1024
```

### Slow Training
```python
# Enable gradient checkpointing
gradient_checkpointing=True

# Use Flash Attention (Ampere+ GPUs)
attn_implementation="flash_attention_2"

# Increase batch size
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 2
```

### NaN Loss
```python
# Lower learning rate
LEARNING_RATE = 1e-5

# Enable gradient clipping
max_grad_norm = 1.0

# Use bf16 instead of fp16 (if supported)
bf16=True
fp16=False
```

---

## ✅ Next Steps

1. **Test Training**: Run with `SAMPLE_RATIO=0.01` first
2. **Full Training**: Set `SAMPLE_RATIO=1.0` for production
3. **Hyperparameter Tuning**: Experiment with beta/lambda_param
4. **Evaluation**: Use the trained model for inference
5. **Deployment**: Push to HuggingFace Hub

---

## 📚 Additional Resources

- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [ORPO Paper](https://arxiv.org/abs/2403.07691)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Unsloth Docs](https://github.com/unslothai/unsloth)

---

**Happy Training! 🚀**
```

Save both complete scripts and run them directly!
