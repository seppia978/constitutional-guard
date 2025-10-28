"""
Training Pipeline for Constitutional Guard

Trains Llama 3.1 8B base with:
- LoRA/PEFT for efficiency
- Multi-task losses (classification + clause + flip-consistency)
- Custom dataset with flip pairs
- Evaluation on flip-rate

Hardware requirements:
- Minimum: 24GB GPU (LoRA)
- Recommended: 40GB GPU (faster training)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from typing import Dict, List
import numpy as np

# Import centralized config
from config import HF_CACHE_DIR as CACHE_DIR, DEFAULT_TRAINING_CONFIG, LORA_OUTPUT_DIR


class ConstitutionalGuardDataset(Dataset):
    """
    Dataset for Constitutional Guard training.

    Handles:
    - Standard examples (policy, input) → JSON output
    - Flip pairs for flip-consistency loss
    """

    def __init__(self, data_file: str, tokenizer, max_length: int = 2048):
        """
        Args:
            data_file: Path to training_dataset.json
            tokenizer: HuggingFace tokenizer
            max_length: Max sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        with open(data_file, 'r') as f:
            data = json.load(f)

        self.examples = data['standard_examples']
        self.flip_pairs = data['flip_pairs']

        print(f"✓ Loaded {len(self.examples)} standard examples")
        print(f"✓ Loaded {len(self.flip_pairs)} flip pairs")

    def __len__(self):
        return len(self.examples) + len(self.flip_pairs)

    def __getitem__(self, idx):
        # Interleave standard examples and flip pairs
        if idx < len(self.examples):
            return self._get_standard_example(idx)
        else:
            return self._get_flip_pair(idx - len(self.examples))

    def _create_prompt(self, policy_text: str, input_text: str, output_json: str = None) -> str:
        """Create training prompt"""
        prompt = f"""[POLICY]
{policy_text}
[/POLICY]

[INPUT]
{input_text}
[/INPUT]

[TASK]
Classify the input according to the policy provided above.
Output valid JSON: {{"is_safe": bool, "violations": [...], "evidence": [...]}}
[/TASK]

[OUTPUT]
"""
        if output_json:
            prompt += output_json + self.tokenizer.eos_token

        return prompt

    def _get_standard_example(self, idx):
        """Get standard training example"""
        ex = self.examples[idx]

        # Create prompt with output
        output_json = json.dumps(ex['output'], indent=2)
        full_text = self._create_prompt(ex['policy_text'], ex['input_text'], output_json)

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),  # Standard language modeling
            'is_flip_pair': False
        }

    def _get_flip_pair(self, idx):
        """
        Get flip pair example.

        Returns both examples (with/without clause) for flip-consistency loss.
        """
        pair = self.flip_pairs[idx]

        # Example WITH clause
        output_with = json.dumps(pair['output_with'], indent=2)
        text_with = self._create_prompt(pair['policy_with'], pair['input_text'], output_with)

        # Example WITHOUT clause
        output_without = json.dumps(pair['output_without'], indent=2)
        text_without = self._create_prompt(pair['policy_without'], pair['input_text'], output_without)

        # Tokenize both
        enc_with = self.tokenizer(
            text_with,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        enc_without = self.tokenizer(
            text_without,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # For flip pairs, we return BOTH examples
        # The trainer will need to handle this specially
        # For now, just return the "with" example and store flip info in metadata
        return {
            'input_ids': enc_with['input_ids'].squeeze(),
            'attention_mask': enc_with['attention_mask'].squeeze(),
            'labels': enc_with['input_ids'].squeeze(),
            'is_flip_pair': True,
            'flip_pair_data': {
                'input_ids_without': enc_without['input_ids'].squeeze(),
                'attention_mask_without': enc_without['attention_mask'].squeeze(),
                'expected_flip': pair['output_with']['is_safe'] != pair['output_without']['is_safe']
            }
        }


def setup_lora_model(model_id: str = "meta-llama/Llama-3.1-8B",
                     lora_r: int = 16,
                     lora_alpha: int = 32,
                     lora_dropout: float = 0.05):
    """
    Setup model with LoRA for efficient training.

    Args:
        model_id: Base model ID
        lora_r: LoRA rank (lower = fewer parameters, faster but less expressive)
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: Dropout rate

    Returns:
        (model, tokenizer)
    """
    print(f"Loading {model_id}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        padding_side="right"  # Important for training
    )

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model in 8-bit for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,  # 8-bit quantization
        device_map="auto",
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],  # Llama 3 attention + MLP
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(
    model_id: str = "meta-llama/Llama-3.1-8B",
    data_file: str = "../data/training_dataset.json",
    output_dir: str = "./constitutional_guard_lora",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
    gradient_accumulation_steps: int = 4,
    max_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32
):
    """
    Train Constitutional Guard with LoRA.

    Args:
        model_id: Base model
        data_file: Path to training_dataset.json
        output_dir: Where to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Peak learning rate
        warmup_steps: Warmup steps
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        gradient_accumulation_steps: Gradient accumulation
        max_length: Max sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
    """
    print("=" * 80)
    print("CONSTITUTIONAL GUARD TRAINING")
    print("=" * 80)

    # Setup model with LoRA
    model, tokenizer = setup_lora_model(
        model_id=model_id,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )

    # Load dataset
    print(f"\nLoading dataset from {data_file}...")
    dataset = ConstitutionalGuardDataset(
        data_file=data_file,
        tokenizer=tokenizer,
        max_length=max_length
    )

    # Split train/eval (90/10)
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )

    print(f"✓ Train examples: {len(train_dataset)}")
    print(f"✓ Eval examples: {len(eval_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,  # Mixed precision
        dataloader_pin_memory=True,
        gradient_checkpointing=True,  # Save memory
        optim="paged_adamw_8bit",  # 8-bit optimizer
        report_to="none",  # Disable wandb/tensorboard for now
        remove_unused_columns=False,  # Keep our custom fields
    )

    # Data collator (handles padding)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Total training steps: {len(train_dataset) // (batch_size * gradient_accumulation_steps) * num_epochs}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print()

    trainer.train()

    # Save final model
    print("\n✓ Training complete!")
    print(f"✓ Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Model: {model_id}")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Epochs: {num_epochs}")
    print(f"Output: {output_dir}")
    print("\nNext steps:")
    print("1. Run evaluation on flip-test dataset")
    print("2. Measure flip-rate (target: ≥95%)")
    print("3. Compare with Llama Guard 3 baseline")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Constitutional Guard")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B",
                       help="Base model ID")
    parser.add_argument("--data_file", type=str,
                       default="../data/training_dataset.json",
                       help="Training dataset JSON")
    parser.add_argument("--output_dir", type=str,
                       default="./constitutional_guard_lora",
                       help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")

    args = parser.parse_args()

    # Check GPU
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training will be very slow on CPU.")
        print("Recommendation: Use a GPU instance with ≥24GB VRAM.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit(0)

    # Train
    train(
        model_id=args.model_id,
        data_file=args.data_file,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
