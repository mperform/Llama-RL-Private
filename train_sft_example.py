#!/usr/bin/env python3
"""
Example SFT Training Script for DeepSeek-R1-Distill-Llama-8B on PMC-VQA

This script demonstrates how to fine-tune the DeepSeek-R1-Distill-Llama-8B model
on the preprocessed PMC-VQA dataset using Hugging Face Transformers.

Requirements:
    pip install transformers datasets accelerate peft bitsandbytes torch

Usage:
    python train_sft_example.py [--full_precision] [--no_lora]
"""

import os
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(model_path, use_lora=True, use_8bit=True):
    """
    Load and setup the model and tokenizer.
    
    Args:
        model_path: Path to the model directory
        use_lora: Whether to use LoRA for efficient fine-tuning
        use_8bit: Whether to use 8-bit quantization
    
    Returns:
        model, tokenizer
    """
    logger.info(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate precision
    if use_8bit:
        logger.info("Loading model with 8-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info("Loading model in full precision...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    
    # Setup LoRA if requested
    if use_lora:
        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA scaling
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def load_preprocessed_dataset(dataset_path):
    """Load the preprocessed PMC-VQA dataset."""
    logger.info(f"Loading preprocessed dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    logger.info(f"Dataset loaded: {dataset}")
    return dataset


def create_training_args(output_dir, num_epochs=3, batch_size=4, grad_accum=8, learning_rate=2e-4):
    """Create training arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        
        # Optimizer settings
        optim="adamw_torch",
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        
        # Logging and saving
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=3,
        
        # Evaluation and best model
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Mixed precision
        fp16=False,
        bf16=True,  # Use bfloat16 for better stability
        
        # Other settings
        report_to="none",  # Change to "wandb" or "tensorboard" if you want logging
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )


def compute_metrics(eval_pred):
    """Compute evaluation metrics (optional)."""
    # Can add custom metrics here if needed
    return {}


def main():
    parser = argparse.ArgumentParser(description="SFT training for DeepSeek-R1-Distill-Llama-8B on PMC-VQA")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/scratch/ece598f25s002_class_root/ece598f25s002_class/mperform/DeepSeek-R1-Distill-Llama-8B",
        help="Path to the model directory"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/scratch/ece598f25s002_class_root/ece598f25s002_class/mperform/PMC-VQA-Processed",
        help="Path to the preprocessed dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/ece598f25s002_class_root/ece598f25s002_class/mperform/PMC-VQA-SFT-Output",
        help="Output directory for the fine-tuned model"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch size = batch_size * grad_accum)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Don't use LoRA (full fine-tuning)"
    )
    parser.add_argument(
        "--full_precision",
        action="store_true",
        help="Use full precision instead of 8-bit quantization"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of training samples (for testing)"
    )
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("DeepSeek-R1-Distill-Llama-8B SFT Training on PMC-VQA")
    logger.info("="*70)
    
    # Load dataset
    dataset = load_preprocessed_dataset(args.dataset_path)
    
    # Optionally limit dataset size for testing
    if args.max_samples is not None:
        logger.info(f"Limiting dataset to {args.max_samples} samples for testing...")
        dataset['train'] = dataset['train'].select(range(min(args.max_samples, len(dataset['train']))))
        dataset['test'] = dataset['test'].select(range(min(args.max_samples // 10, len(dataset['test']))))
    
    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Test samples: {len(dataset['test'])}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.model_path,
        use_lora=not args.no_lora,
        use_8bit=not args.full_precision
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Create training arguments
    training_args = create_training_args(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
    )
    
    # Train
    logger.info("="*70)
    logger.info("Starting training...")
    logger.info("="*70)
    logger.info(f"Effective batch size: {args.batch_size * args.grad_accum}")
    logger.info(f"Number of epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"LoRA enabled: {not args.no_lora}")
    logger.info(f"8-bit quantization: {not args.full_precision}")
    logger.info("="*70)
    
    trainer.train()
    
    # Save the final model
    logger.info("="*70)
    logger.info("Training complete! Saving model...")
    logger.info("="*70)
    
    output_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    
    logger.info(f"Model saved to: {output_model_dir}")
    
    # Evaluate on test set
    logger.info("="*70)
    logger.info("Evaluating on test set...")
    logger.info("="*70)
    
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Save evaluation results
    import json
    results_file = os.path.join(args.output_dir, "eval_results.json")
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {results_file}")
    
    logger.info("="*70)
    logger.info("✅ Training completed successfully!")
    logger.info("="*70)
    logger.info(f"\nTo use the fine-tuned model:")
    logger.info(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    logger.info(f"  model = AutoModelForCausalLM.from_pretrained('{output_model_dir}')")
    logger.info(f"  tokenizer = AutoTokenizer.from_pretrained('{output_model_dir}')")


if __name__ == "__main__":
    main()

