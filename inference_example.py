#!/usr/bin/env python3
"""
Inference Example Script for Fine-Tuned DeepSeek-R1-Distill-Llama-8B

This script demonstrates how to use the fine-tuned model for inference
on PMC-VQA medical questions.

Usage:
    python inference_example.py --model_path PMC-VQA-SFT-Output/final_model
    python inference_example.py --interactive  # For interactive Q&A
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import json
from typing import List, Dict
import sys


class MedicalQAInference:
    """Inference wrapper for medical QA with DeepSeek-R1."""
    
    def __init__(self, model_path: str, use_8bit: bool = True):
        """
        Initialize the inference model.
        
        Args:
            model_path: Path to the fine-tuned model
            use_8bit: Whether to load in 8-bit mode
        """
        print(f"Loading model from {model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if use_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                load_in_8bit=True,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def format_question(
        self,
        question: str,
        choices: Dict[str, str] = None,
        image_path: str = None
    ) -> str:
        """
        Format a medical question for the model.
        
        Args:
            question: The medical question
            choices: Optional dict of choices (e.g., {'A': 'choice text', ...})
            image_path: Optional path to image
            
        Returns:
            Formatted prompt string
        """
        prompt = "You are a medical expert. Answer the following medical question."
        
        if image_path:
            prompt += f"\n\nImage: {image_path}"
        
        prompt += f"\n\nQuestion: {question}"
        
        if choices:
            prompt += "\n\nOptions:"
            for letter, choice in sorted(choices.items()):
                prompt += f"\n{letter}: {choice}"
            prompt += "\n\nProvide the correct answer and explain your reasoning."
        
        return prompt
    
    def generate_answer(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate an answer for the given prompt.
        
        Args:
            prompt: The formatted question prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated answer string
        """
        # Create conversation
        conversation = [{"role": "user", "content": prompt}]
        
        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode (only the generated part)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def answer_question(
        self,
        question: str,
        choices: Dict[str, str] = None,
        image_path: str = None,
        **generation_kwargs
    ) -> str:
        """
        Answer a medical question (convenience method).
        
        Args:
            question: The medical question
            choices: Optional multiple choice options
            image_path: Optional image path
            **generation_kwargs: Additional arguments for generation
            
        Returns:
            Model's answer
        """
        prompt = self.format_question(question, choices, image_path)
        return self.generate_answer(prompt, **generation_kwargs)


def evaluate_on_dataset(model: MedicalQAInference, dataset_path: str, num_samples: int = 10):
    """
    Evaluate the model on a subset of the test dataset.
    
    Args:
        model: The inference model
        dataset_path: Path to the preprocessed dataset
        num_samples: Number of samples to evaluate
    """
    print(f"\nLoading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    test_set = dataset['test']
    
    print(f"Evaluating on {num_samples} samples from test set...")
    print("="*80)
    
    results = []
    
    for i in range(min(num_samples, len(test_set))):
        sample = test_set[i]
        
        # Extract question and choices
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        figure_path = sample.get('figure_path', '')
        
        # Format choices if available
        choices = {}
        for letter in ['A', 'B', 'C', 'D']:
            choice_key = f'Choice {letter}'
            if choice_key in sample and sample[choice_key]:
                choices[letter] = sample[choice_key]
        
        print(f"\n{'─'*80}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"{'─'*80}")
        print(f"\nImage: {figure_path}")
        print(f"\nQuestion: {question}")
        
        if choices:
            print("\nOptions:")
            for letter, choice_text in sorted(choices.items()):
                print(f"  {letter}: {choice_text}")
        
        print(f"\nGround Truth Answer: {answer}")
        
        # Generate prediction
        print("\nGenerating prediction...")
        prediction = model.answer_question(
            question=question,
            choices=choices if choices else None,
            image_path=figure_path,
            max_new_tokens=256,
            temperature=0.7
        )
        
        print(f"\nModel Prediction: {prediction}")
        
        results.append({
            'question': question,
            'ground_truth': answer,
            'prediction': prediction,
            'image': figure_path
        })
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)
    
    return results


def interactive_mode(model: MedicalQAInference):
    """
    Interactive Q&A mode.
    
    Args:
        model: The inference model
    """
    print("\n" + "="*80)
    print("Interactive Medical Q&A Mode")
    print("="*80)
    print("\nEnter medical questions to get answers from the model.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            # Get question
            print("─"*80)
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            # Optional: ask for choices
            use_choices = input("Include multiple choice options? (y/n): ").strip().lower()
            choices = None
            
            if use_choices == 'y':
                choices = {}
                for letter in ['A', 'B', 'C', 'D']:
                    choice = input(f"  Choice {letter}: ").strip()
                    if choice:
                        choices[letter] = choice
            
            # Optional: image path
            image_path = input("Image path (optional, press Enter to skip): ").strip()
            if not image_path:
                image_path = None
            
            # Generate answer
            print("\nGenerating answer...")
            answer = model.answer_question(
                question=question,
                choices=choices if choices else None,
                image_path=image_path
            )
            
            print(f"\n{'='*80}")
            print("Model's Answer:")
            print(f"{'='*80}")
            print(answer)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/scratch/ece598f25s002_class_root/ece598f25s002_class/mperform/PMC-VQA-SFT-Output/final_model",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/scratch/ece598f25s002_class_root/ece598f25s002_class/mperform/PMC-VQA-Processed",
        help="Path to the preprocessed dataset (for evaluation)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive Q&A mode"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate on test dataset"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples for evaluation"
    )
    parser.add_argument(
        "--full_precision",
        action="store_true",
        help="Use full precision instead of 8-bit"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question to answer (for quick testing)"
    )
    
    args = parser.parse_args()
    
    # Load model
    model = MedicalQAInference(
        model_path=args.model_path,
        use_8bit=not args.full_precision
    )
    
    # Run appropriate mode
    if args.interactive:
        interactive_mode(model)
    elif args.evaluate:
        results = evaluate_on_dataset(model, args.dataset_path, args.num_samples)
        
        # Save results
        output_file = "evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    elif args.question:
        # Quick single question
        print(f"\nQuestion: {args.question}")
        answer = model.answer_question(args.question)
        print(f"\nAnswer: {answer}")
    else:
        # Demo mode with example questions
        print("\n" + "="*80)
        print("Demo Mode - Example Medical Questions")
        print("="*80)
        
        examples = [
            {
                "question": "What is the uptake pattern in the breast tissue shown?",
                "choices": {
                    "A": "Diffuse uptake pattern",
                    "B": "Focal uptake pattern",
                    "C": "No uptake pattern",
                    "D": "Cannot determine"
                }
            },
            {
                "question": "What radiological technique is most appropriate for confirming this diagnosis?",
                "choices": {
                    "A": "Mammography",
                    "B": "CT Scan",
                    "C": "MRI",
                    "D": "X-ray"
                }
            },
            {
                "question": "What pathological finding is present in the H&E stained specimen?",
                "choices": {
                    "A": "Normal tissue architecture",
                    "B": "Inflammatory cell infiltration",
                    "C": "Neoplastic changes",
                    "D": "Fibrosis"
                }
            }
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"\n{'─'*80}")
            print(f"Example {i}")
            print(f"{'─'*80}")
            print(f"\nQuestion: {example['question']}")
            print("\nOptions:")
            for letter, choice in sorted(example['choices'].items()):
                print(f"  {letter}: {choice}")
            
            answer = model.answer_question(
                question=example['question'],
                choices=example['choices']
            )
            
            print(f"\nModel's Answer:")
            print(answer)
        
        print("\n" + "="*80)
        print("Demo complete!")
        print("="*80)
        print("\nTo run in interactive mode: python inference_example.py --interactive")
        print("To evaluate on test set: python inference_example.py --evaluate")


if __name__ == "__main__":
    main()

