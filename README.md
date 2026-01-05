# RLHF Modular Project - PPO, DPO & ORPO

A comprehensive, production-ready implementation of Reinforcement Learning from Human Feedback (RLHF) techniques for Large Language Models.

## 🚀 Supported Techniques

### 1. **PPO (Proximal Policy Optimization)**
- Traditional RLHF approach
- Requires separate reward model
- Stable policy updates with clipping
- Best for: Complex reward functions

### 2. **DPO (Direct Preference Optimization)**
- Simplified RLHF without reward model
- Direct optimization on preference pairs
- More stable than PPO
- Best for: Preference-based alignment

### 3. **ORPO (Odds Ratio Preference Optimization)**
- Single-stage training (SFT + Preference learning)
- Most efficient approach
- Strong performance with less compute
- Best for: Resource-constrained environments

## 📁 Project Structure

```
rlhf_modular/
├── config/                 # Configuration files
│   ├── __init__.py
│   ├── model_config.py    # Model configurations
│   ├── training_config.py # Training hyperparameters
│   └── data_config.py     # Dataset configurations
├── data/                   # Data processing
│   ├── __init__.py
│   ├── data_loader.py     # Dataset loading
│   ├── preprocessor.py    # Data preprocessing
│   └── collators.py       # Data collators
├── models/                 # Model components
│   ├── __init__.py
│   ├── model_loader.py    # Model initialization
│   ├── peft_config.py     # LoRA/QLoRA configs
│   └── reward_model.py    # Reward model (for PPO)
├── trainers/               # Training logic
│   ├── __init__.py
│   ├── base_trainer.py    # Base trainer class
│   ├── ppo_trainer.py     # PPO implementation
│   ├── dpo_trainer.py     # DPO implementation
│   └── orpo_trainer.py    # ORPO implementation
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── metrics.py         # Evaluation metrics
│   ├── logger.py          # Logging utilities
│   └── helpers.py         # Helper functions
├── scripts/                # Execution scripts
│   ├── train_ppo.py       # Train with PPO
│   ├── train_dpo.py       # Train with DPO
│   ├── train_orpo.py      # Train with ORPO
│   └── evaluate.py        # Model evaluation
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_ppo_training.ipynb
│   ├── 03_dpo_training.ipynb
│   └── 04_orpo_training.ipynb
├── requirements.txt
├── setup.py
└── README.md
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/ruhul-cse-duet/RLHF-LLM-models-PPO-DPO-ORPO.git
cd RLHF-LLM-models-PPO-DPO-ORPO

# Install dependencies
pip install -r requirements.txt
```

## 💻 Quick Start

### Train with DPO
```bash
python scripts/train_dpo.py \
    --model_name mistralai/Mistral-7B-v0.1 \
    --dataset HuggingFaceH4/ultrafeedback_binarized \
    --output_dir ./outputs/dpo \
    --num_epochs 3
```

### Train with ORPO
```bash
python scripts/train_orpo.py \
    --model_name mistralai/Mistral-7B-v0.1 \
    --dataset HuggingFaceH4/ultrafeedback_binarized \
    --output_dir ./outputs/orpo \
    --num_epochs 3
```

### Train with PPO
```bash
python scripts/train_ppo.py \
    --model_name gpt2 \
    --reward_model <reward-model-path> \
    --dataset <dataset-name> \
    --output_dir ./outputs/ppo
```

## 📊 Features

- ✅ Modular and extensible architecture
- ✅ Support for multiple RL techniques (PPO, DPO, ORPO)
- ✅ 4-bit quantization (QLoRA) support
- ✅ LoRA/QLoRA fine-tuning
- ✅ Mixed precision training
- ✅ Comprehensive logging and metrics
- ✅ Easy configuration management
- ✅ Production-ready code

## 📈 Performance Comparison

| Technique | Training Speed | Memory Usage | Performance | Complexity |
|-----------|---------------|--------------|-------------|------------|
| PPO       | ⭐⭐          | ⭐⭐          | ⭐⭐⭐⭐     | ⭐⭐⭐⭐⭐    |
| DPO       | ⭐⭐⭐⭐      | ⭐⭐⭐        | ⭐⭐⭐⭐     | ⭐⭐⭐      |
| ORPO      | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐      | ⭐⭐⭐⭐⭐   | ⭐⭐        |

## 📚 Documentation

For detailed documentation, see:
- [Configuration Guide](docs/configuration.md)
- [Training Guide](docs/training.md)
- [API Reference](docs/api.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Hugging Face TRL library
- Unsloth for fast training
- OpenAI for pioneering RLHF
- Anthropic for DPO research

## Author
[Md Ruhul Amin](https://www.linkedin.com/in/ruhul-duet-cse/);  
Email: ruhul.cse.duet@gmail.com