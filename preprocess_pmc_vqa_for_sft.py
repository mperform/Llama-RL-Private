#!/usr/bin/env python3
"""
PMC-VQA Dataset Preprocessing for SFT on DeepSeek-R1-Distill-Llama-8B

This script preprocesses the PMC-VQA dataset for supervised fine-tuning (SFT)
on the DeepSeek-R1-Distill-Llama-8B model. It converts the medical VQA data
into the chat template format expected by the model.

Author: Generated for ECE598 Biomedical AI Project
Date: October 2025
"""

import os
import json
import pandas as pd
import psutil
import gc
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm


class PMCVQAPreprocessor:
    """Preprocessor for PMC-VQA dataset for DeepSeek-R1-Distill-Llama-8B SFT."""
    
    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        max_length: int = 2048,
        include_choices: bool = True,
        add_reasoning: bool = False
    ):
        """
        Initialize the preprocessor.
        
        Args:
            model_path: Path to the DeepSeek-R1-Distill-Llama-8B model directory
            dataset_path: Path to the PMC-VQA dataset (CSV or HF dataset)
            max_length: Maximum sequence length for tokenization
            include_choices: Whether to include multiple choice options in prompt
            add_reasoning: Whether to add reasoning prompts (for R1 thinking)
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.max_length = max_length
        self.include_choices = include_choices
        self.add_reasoning = add_reasoning
        
        # Load tokenizer
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        print(f"Max model length: {self.tokenizer.model_max_length}")
        print(f"BOS token: {self.tokenizer.bos_token}")
        print(f"EOS token: {self.tokenizer.eos_token}")
    
    def print_memory_usage(self, stage: str = ""):
        """Print current memory usage."""
        memory = psutil.virtual_memory()
        print(f"Memory {stage}: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB) - Available: {memory.available / 1024**3:.1f}GB")
        
    def format_multiple_choice(self, row: Dict) -> str:
        """Format the multiple choice options."""
        choices = []
        for label in ['A', 'B', 'C', 'D']:
            choice_text = row.get(f'Choice {label}', '').strip()
            if choice_text:
                choices.append(f"{label}: {choice_text}")
        
        if choices:
            return "\n".join(choices)
        return ""
    
    def create_conversation(self, row: Dict) -> List[Dict[str, str]]:
        """
        Create a conversation format from a dataset row.
        
        Args:
            row: Dictionary containing question, answer, and choices
            
        Returns:
            List of message dictionaries in chat format
        """
        # Extract fields
        question = row.get('Question', '').strip() if row.get('Question') else ''
        answer = row.get('Answer', '').strip() if row.get('Answer') else ''
        figure_path = row.get('Figure_path', '').strip() if row.get('Figure_path') else ''
        
        # Build the user prompt
        user_prompt = f"You are a medical expert. Answer the following medical question based on the image."
        
        if figure_path:
            user_prompt += f"\n\nImage: {figure_path}"
        
        user_prompt += f"\n\nQuestion: {question}"
        
        # Add multiple choice options if requested
        if self.include_choices:
            mc_text = self.format_multiple_choice(row)
            if mc_text:
                user_prompt += f"\n\nOptions:\n{mc_text}"
                user_prompt += "\n\nProvide the correct answer and explain your reasoning."
        
        # Build the assistant response
        if self.add_reasoning:
            # For DeepSeek-R1, we can add a reasoning section
            assistant_response = f"Let me analyze this medical question step by step.\n\n"
            assistant_response += f"The correct answer is: {answer}\n\n"
            assistant_response += "This answer is based on the medical knowledge and visual information provided in the image."
        else:
            # Simple answer format
            assistant_response = answer
        
        # Create conversation
        conversation = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
        
        return conversation
    
    def apply_chat_template(self, conversation: List[Dict[str, str]]) -> str:
        """
        Apply the DeepSeek-R1 chat template to a conversation.
        
        Args:
            conversation: List of message dictionaries
            
        Returns:
            Formatted text string
        """
        # Use the tokenizer's chat template
        formatted_text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return formatted_text
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """
        Tokenize the formatted conversations.
        
        Args:
            examples: Dictionary of examples from the dataset
            
        Returns:
            Dictionary with tokenized inputs
        """
        # Get the formatted texts
        texts = examples['text']
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding=False,  # Don't pad during tokenization
            truncation=True,
            max_length=self.max_length,
            return_tensors=None  # Return lists, not tensors
        )
        
        # For causal LM training, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    def preprocess_dataset(self, dataset: Dataset, batch_size: int = 8000) -> Dataset:
        """
        Preprocess the entire dataset in memory-efficient batches.
        
        Args:
            dataset: HuggingFace Dataset object
            batch_size: Number of examples to process at once
            
        Returns:
            Preprocessed dataset with tokenized inputs
        """
        print(f"\nPreprocessing {len(dataset)} examples in batches of {batch_size}...")
        self.print_memory_usage("at start")
        
        # Process in batches to avoid memory issues
        all_tokenized_data = []
        total_batches = (len(dataset) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            
            print(f"\nProcessing batch {batch_idx + 1}/{total_batches} (examples {start_idx}-{end_idx-1})...")
            self.print_memory_usage(f"before batch {batch_idx + 1}")
            
            # Process current batch
            batch_examples = []
            for idx in tqdm(range(start_idx, end_idx), desc=f"Batch {batch_idx + 1}"):
                row = dataset[idx]
                conversation = self.create_conversation(row)
                formatted_text = self.apply_chat_template(conversation)
                batch_examples.append({
                    'text': formatted_text,
                    'question': row.get('Question', ''),
                    'answer': row.get('Answer', ''),
                    'figure_path': row.get('Figure_path', '')
                })
            
            # Tokenize current batch
            print(f"Tokenizing batch {batch_idx + 1}...")
            batch_dataset = Dataset.from_list(batch_examples)
            tokenized_batch = batch_dataset.map(
                self.tokenize_function,
                batched=True,
                batch_size=1000,
                remove_columns=['text'],
                desc=f"Tokenizing batch {batch_idx + 1}"
            )
            
            # Add to results
            all_tokenized_data.extend([dict(example) for example in tokenized_batch])
            
            # Clean up memory
            del batch_examples, batch_dataset, tokenized_batch
            gc.collect()
            
            self.print_memory_usage(f"after batch {batch_idx + 1}")
        
        # Create final dataset
        print(f"\nCreating final dataset from {len(all_tokenized_data)} examples...")
        final_dataset = Dataset.from_list(all_tokenized_data)
        
        self.print_memory_usage("at end")
        return final_dataset
    
    def load_dataset(self) -> DatasetDict:
        """Load the PMC-VQA dataset from disk or CSV."""
        print(f"\nLoading dataset from {self.dataset_path}...")
        
        if os.path.isdir(self.dataset_path):
            # Try loading as HF dataset
            hf_dataset_path = os.path.join(self.dataset_path, "hf_dataset")
            if os.path.exists(hf_dataset_path):
                dataset = load_from_disk(hf_dataset_path)
                print(f"Loaded HF dataset: {dataset}")
                return dataset
        
        # Try loading as CSV files
        train_csv = os.path.join(self.dataset_path, "train.csv")
        test_csv = os.path.join(self.dataset_path, "test.csv")
        
        if os.path.exists(train_csv) and os.path.exists(test_csv):
            print("Loading from CSV files...")
            train_df = pd.read_csv(train_csv)
            test_df = pd.read_csv(test_csv)
            
            train_dataset = Dataset.from_pandas(train_df)
            test_dataset = Dataset.from_pandas(test_df)
            
            dataset = DatasetDict({
                "train": train_dataset,
                "test": test_dataset
            })
            print(f"Loaded CSV dataset: {dataset}")
            return dataset
        
        raise ValueError(f"Could not load dataset from {self.dataset_path}")
    
    def run(self, output_dir: str, sample_size: Optional[int] = None):
        """
        Run the full preprocessing pipeline.
        
        Args:
            output_dir: Directory to save the preprocessed dataset
            sample_size: Optional number of samples to use (for testing)
        """
        # Load dataset
        dataset = self.load_dataset()
        
        # Sample if requested
        if sample_size is not None:
            print(f"\nSampling {sample_size} examples from each split...")
            dataset = DatasetDict({
                split: ds.select(range(min(sample_size, len(ds))))
                for split, ds in dataset.items()
            })
        
        # Preprocess each split
        processed_dataset = DatasetDict()
        for split in dataset.keys():
            print(f"\n{'='*60}")
            print(f"Processing {split} split")
            print(f"{'='*60}")
            processed_dataset[split] = self.preprocess_dataset(dataset[split])
        
        # Save the processed dataset
        print(f"\nSaving preprocessed dataset to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        processed_dataset.save_to_disk(output_dir)
        
        # Save a few examples for inspection
        examples_file = os.path.join(output_dir, "examples.json")
        examples = []
        for split in processed_dataset.keys():
            for i in range(min(3, len(processed_dataset[split]))):
                example = processed_dataset[split][i]
                decoded_text = self.tokenizer.decode(
                    example['input_ids'],
                    skip_special_tokens=False
                )
                examples.append({
                    'split': split,
                    'index': i,
                    'question': example['question'],
                    'answer': example['answer'],
                    'formatted_text': decoded_text,
                    'num_tokens': len(example['input_ids'])
                })
        
        with open(examples_file, 'w') as f:
            json.dump(examples, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Preprocessing complete!")
        print(f"{'='*60}")
        print(f"Processed dataset saved to: {output_dir}")
        print(f"Example outputs saved to: {examples_file}")
        print(f"\nDataset statistics:")
        for split in processed_dataset.keys():
            print(f"  {split}: {len(processed_dataset[split])} examples")
        
        return processed_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess PMC-VQA dataset for DeepSeek-R1-Distill-Llama-8B SFT"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        # default="/scratch/ece598f25s002_class_root/ece598f25s002_class/mperform/DeepSeek-R1-Distill-Llama-8B",
        default=r'D:\Github\Llama-RL-Private\DeepSeek-R1-Distill-Llama-8B',
        help="Path to the DeepSeek-R1-Distill-Llama-8B model directory"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        # default="/scratch/ece598f25s002_class_root/ece598f25s002_class/mperform/PMC-VQA",
        default=r'D:\Github\Llama-RL-Private\PMC-VQA',
        help="Path to the PMC-VQA dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        # default="/scratch/ece598f25s002_class_root/ece598f25s002_class/mperform/PMC-VQA-Processed",
        default=r'D:\Github\Llama-RL-Private\PMC-VQA-Processed',
        help="Output directory for preprocessed dataset"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to use (for testing, default: use all)"
    )
    parser.add_argument(
        "--no_choices",
        action="store_true",
        help="Don't include multiple choice options in prompts"
    )
    parser.add_argument(
        "--add_reasoning",
        action="store_true",
        help="Add reasoning prompts for DeepSeek-R1 thinking"
    )
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = PMCVQAPreprocessor(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        max_length=args.max_length,
        include_choices=not args.no_choices,
        add_reasoning=args.add_reasoning
    )
    
    # Run preprocessing
    preprocessor.run(
        output_dir=args.output_dir,
        sample_size=args.sample_size
    )


if __name__ == "__main__":
    main()

