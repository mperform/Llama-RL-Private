#!/usr/bin/env python3
"""
Installation Verification Script

This script checks that all required packages are installed and working correctly.
Run this after installing requirements to verify your setup.

Usage:
    python verify_installation.py
"""

import sys
import os


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_status(check_name, status, details=""):
    """Print status of a check."""
    symbol = "✅" if status else "❌"
    print(f"{symbol} {check_name:<40} {details}")


def check_python_version():
    """Check Python version."""
    print_header("Python Version")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    is_ok = version >= (3, 8)
    print_status("Python Version", is_ok, f"v{version_str}")
    if not is_ok:
        print("   ⚠️  Python 3.8+ required!")
    return is_ok


def check_pytorch():
    """Check PyTorch installation and CUDA."""
    print_header("PyTorch & CUDA")
    checks_passed = True
    
    try:
        import torch
        print_status("PyTorch Import", True, f"v{torch.__version__}")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print_status("CUDA Available", cuda_available, 
                    f"CUDA {torch.version.cuda}" if cuda_available else "CPU only")
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print_status("GPU Device", True, f"{gpu_name} ({gpu_count} GPU(s))")
            
            # Check memory
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_status("GPU Memory", mem_gb >= 16, f"{mem_gb:.1f} GB")
            if mem_gb < 16:
                print("   ⚠️  16+ GB GPU memory recommended")
        else:
            print("   ⚠️  GPU highly recommended for training")
            checks_passed = False
            
    except ImportError as e:
        print_status("PyTorch Import", False, str(e))
        checks_passed = False
    
    return checks_passed


def check_core_packages():
    """Check core ML packages."""
    print_header("Core ML Packages")
    checks_passed = True
    
    packages = [
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
        ("peft", "PEFT"),
        ("bitsandbytes", "BitsAndBytes"),
    ]
    
    for module_name, display_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print_status(f"{display_name} Import", True, f"v{version}")
        except ImportError as e:
            print_status(f"{display_name} Import", False, str(e))
            checks_passed = False
    
    return checks_passed


def check_utility_packages():
    """Check utility packages."""
    print_header("Utility Packages")
    checks_passed = True
    
    packages = [
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("tqdm", "tqdm"),
        ("sentencepiece", "SentencePiece"),
        ("safetensors", "SafeTensors"),
    ]
    
    for module_name, display_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print_status(f"{display_name} Import", True, f"v{version}")
        except ImportError as e:
            print_status(f"{display_name} Import", False, str(e))
            checks_passed = False
    
    return checks_passed


def check_paths():
    """Check required paths exist."""
    print_header("Project Paths")
    checks_passed = True
    
    base_dir = "/scratch/ece598f25s002_class_root/ece598f25s002_class/mperform"
    
    paths = [
        ("Model Directory", f"{base_dir}/DeepSeek-R1-Distill-Llama-8B"),
        ("Dataset Directory", f"{base_dir}/PMC-VQA"),
        ("Preprocessing Script", f"{base_dir}/preprocess_pmc_vqa_for_sft.py"),
        ("Training Script", f"{base_dir}/train_sft_example.py"),
        ("Inference Script", f"{base_dir}/inference_example.py"),
    ]
    
    for name, path in paths:
        exists = os.path.exists(path)
        print_status(name, exists, path if exists else "Not found")
        if not exists:
            checks_passed = False
    
    return checks_passed


def check_disk_space():
    """Check available disk space."""
    print_header("Disk Space")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/scratch/ece598f25s002_class_root/ece598f25s002_class/mperform")
        free_gb = free / (1024**3)
        
        is_ok = free_gb >= 50
        print_status("Free Disk Space", is_ok, f"{free_gb:.1f} GB")
        if not is_ok:
            print("   ⚠️  50+ GB recommended for full pipeline")
        
        return is_ok
    except Exception as e:
        print_status("Disk Space Check", False, str(e))
        return False


def test_model_loading():
    """Test loading the tokenizer."""
    print_header("Model Loading Test")
    
    try:
        from transformers import AutoTokenizer
        model_path = "/scratch/ece598f25s002_class_root/ece598f25s002_class/mperform/DeepSeek-R1-Distill-Llama-8B"
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print_status("Tokenizer Loading", True, f"Vocab size: {len(tokenizer)}")
        
        # Test tokenization
        test_text = "What is the medical diagnosis?"
        tokens = tokenizer(test_text, return_tensors="pt")
        print_status("Tokenization Test", True, f"{len(tokens['input_ids'][0])} tokens")
        
        return True
    except Exception as e:
        print_status("Model Loading", False, str(e))
        return False


def test_dataset_loading():
    """Test loading the dataset."""
    print_header("Dataset Loading Test")
    
    try:
        import pandas as pd
        dataset_path = "/scratch/ece598f25s002_class_root/ece598f25s002_class/mperform/PMC-VQA"
        train_csv = os.path.join(dataset_path, "train.csv")
        
        if os.path.exists(train_csv):
            print("Loading train.csv...")
            df = pd.read_csv(train_csv, nrows=10)
            print_status("CSV Loading", True, f"{len(df)} rows (sample)")
            
            # Check columns
            required_cols = ['Question', 'Answer', 'Figure_path']
            has_cols = all(col in df.columns for col in required_cols)
            print_status("Required Columns", has_cols, ", ".join(df.columns[:5]))
            
            return True
        else:
            print_status("Dataset Found", False, "train.csv not found")
            return False
            
    except Exception as e:
        print_status("Dataset Loading", False, str(e))
        return False


def main():
    """Run all verification checks."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  PMC-VQA SFT Pipeline - Installation Verification".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    results = []
    
    # Run all checks
    results.append(("Python Version", check_python_version()))
    results.append(("PyTorch & CUDA", check_pytorch()))
    results.append(("Core Packages", check_core_packages()))
    results.append(("Utility Packages", check_utility_packages()))
    results.append(("Project Paths", check_paths()))
    results.append(("Disk Space", check_disk_space()))
    results.append(("Model Loading", test_model_loading()))
    results.append(("Dataset Loading", test_dataset_loading()))
    
    # Summary
    print_header("Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        symbol = "✅" if result else "❌"
        print(f"{symbol} {name}")
    
    print("\n" + "="*70)
    if passed == total:
        print("🎉 All checks passed! You're ready to start.")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run: python test_preprocessing.py")
        print("  2. Read: QUICKSTART_GUIDE.md")
        print("  3. Start preprocessing: python preprocess_pmc_vqa_for_sft.py")
        return 0
    else:
        print(f"⚠️  {total - passed}/{total} checks failed.")
        print("="*70)
        print("\nPlease fix the issues above before proceeding.")
        print("See INSTALLATION.md for troubleshooting help.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

