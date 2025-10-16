#!/usr/bin/env python3
"""
Quick test script to verify PMC-VQA preprocessing works correctly.

This script runs a small sample preprocessing and displays the results.
"""

import os
import sys
import json
from preprocess_pmc_vqa_for_sft import PMCVQAPreprocessor


def main():
    print("="*70)
    print("PMC-VQA Preprocessing Test")
    print("="*70)
    
    # Paths
    model_path = "/scratch/ece598f25s002_class_root/ece598f25s002_class/mperform/DeepSeek-R1-Distill-Llama-8B"
    dataset_path = "/scratch/ece598f25s002_class_root/ece598f25s002_class/mperform/PMC-VQA"
    output_dir = "/scratch/ece598f25s002_class_root/ece598f25s002_class/mperform/PMC-VQA-Test-Sample"
    
    # Check if paths exist
    if not os.path.exists(model_path):
        print(f"ERROR: Model path not found: {model_path}")
        return
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path not found: {dataset_path}")
        return
    
    print(f"\nModel path: {model_path}")
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    
    # Create preprocessor
    print("\n" + "="*70)
    print("Initializing preprocessor...")
    print("="*70)
    
    preprocessor = PMCVQAPreprocessor(
        model_path=model_path,
        dataset_path=dataset_path,
        max_length=2048,
        include_choices=True,
        add_reasoning=False
    )
    
    # Run on small sample
    print("\n" + "="*70)
    print("Running preprocessing on 10 samples...")
    print("="*70)
    
    try:
        preprocessor.run(
            output_dir=output_dir,
            sample_size=10
        )
        
        # Display examples
        print("\n" + "="*70)
        print("Sample Examples")
        print("="*70)
        
        examples_file = os.path.join(output_dir, "examples.json")
        if os.path.exists(examples_file):
            with open(examples_file, 'r') as f:
                examples = json.load(f)
            
            for i, example in enumerate(examples[:2]):  # Show first 2
                print(f"\n{'─'*70}")
                print(f"Example {i+1} - {example['split'].upper()} Split")
                print(f"{'─'*70}")
                print(f"\nQuestion: {example['question']}")
                print(f"\nAnswer: {example['answer']}")
                print(f"\nTokens: {example['num_tokens']}")
                print(f"\nFormatted Text:")
                print(f"{example['formatted_text']}")
        
        print("\n" + "="*70)
        print("✅ TEST PASSED - Preprocessing completed successfully!")
        print("="*70)
        print(f"\nYou can now inspect the full results at: {output_dir}")
        print("\nTo run full preprocessing, execute:")
        print("  python preprocess_pmc_vqa_for_sft.py")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

