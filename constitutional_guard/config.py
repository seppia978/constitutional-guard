"""
Centralized configuration for Constitutional Guard

All cache directories and paths are defined here.
"""

import os

# HuggingFace cache directory (where models will be downloaded)
HF_CACHE_DIR = "/ephemeral/shared/spoppi/huggingface/hub/"

# Ensure directory exists
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Set environment variables for HuggingFace
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = HF_CACHE_DIR

# Project directories
PROJECT_ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
TRAINING_DIR = os.path.join(PROJECT_ROOT, 'training')
EVAL_DIR = os.path.join(PROJECT_ROOT, 'evaluation')

# Data files
TRAINING_DATASET = os.path.join(DATA_DIR, 'training_dataset.json')
FLIP_TEST_DATASET = os.path.join(DATA_DIR, 'flip_test_dataset.json')
POLICY_ZOO = os.path.join(DATA_DIR, 'policy_zoo.json')

# Model checkpoints
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B"
LORA_OUTPUT_DIR = os.path.join(TRAINING_DIR, 'constitutional_guard_lora')

# Training hyperparameters
DEFAULT_TRAINING_CONFIG = {
    'num_epochs': 3,
    'batch_size': 4,
    'gradient_accumulation_steps': 4,
    'learning_rate': 2e-4,
    'warmup_steps': 100,
    'max_length': 2048,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
}

# Evaluation config
FLIP_RATE_TARGET = 0.95  # 95% target flip-rate

print(f"✓ Configuration loaded")
print(f"  HF_CACHE_DIR: {HF_CACHE_DIR}")
print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
