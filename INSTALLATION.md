# Installation Guide

This guide covers the installation of all dependencies needed for the PMC-VQA SFT pipeline.

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **CUDA**: 11.8+ or 12.1+ (for GPU support)
- **GPU**: NVIDIA GPU with 16+ GB VRAM (24+ GB recommended)
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

### Check Your Setup

```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check CUDA availability (if GPU available)
nvidia-smi

# Check if PyTorch can see CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Installation Methods

### Method 1: Full Installation (Recommended)

This installs all dependencies including PyTorch with CUDA support.

```bash
# Navigate to project directory
cd /scratch/ece598f25s002_class_root/ece598f25s002_class/mperform

# Install all requirements
pip install -r requirements.txt
```

### Method 2: Minimal Installation

If you already have PyTorch installed or want to install it separately:

```bash
# Install PyTorch separately (example for CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install other dependencies
pip install -r requirements-minimal.txt
```

### Method 3: Custom PyTorch Installation

For specific CUDA versions, visit: https://pytorch.org/get-started/locally/

```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other requirements
pip install -r requirements-minimal.txt
```

### Method 4: Using Conda (Alternative)

```bash
# Create conda environment
conda create -n pmc-vqa python=3.10
conda activate pmc-vqa

# Install PyTorch with conda
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements-minimal.txt
```

## Verification

After installation, verify everything works:

```bash
# Test imports
python3 -c "
import torch
import transformers
import datasets
import peft
import bitsandbytes
print('✅ All core packages imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

Expected output:
```
✅ All core packages imported successfully!
PyTorch version: 2.1.0+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA A100-SXM4-40GB
```

## Quick Start After Installation

```bash
# Test the preprocessing pipeline
python test_preprocessing.py
```

If this runs without errors, you're ready to go! 🎉

## Troubleshooting

### Issue: `torch.cuda.is_available()` returns `False`

**Solution:**
1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with correct CUDA version
3. Verify CUDA installation: `nvcc --version`

### Issue: `ImportError: No module named 'transformers'`

**Solution:**
```bash
pip install --upgrade transformers
```

### Issue: `bitsandbytes` installation fails

**Solution:**
```bash
# For Linux
pip install bitsandbytes

# For Windows (WSL2 required)
pip install bitsandbytes-windows

# If still failing, try without 8-bit support:
# Use --full_precision flag when training
```

### Issue: Out of memory during installation

**Solution:**
```bash
# Install packages one by one
pip install torch
pip install transformers
pip install datasets
# ... continue with others
```

### Issue: Version conflicts

**Solution:**
```bash
# Create fresh virtual environment
python3 -m venv pmc-vqa-env
source pmc-vqa-env/bin/activate  # On Windows: pmc-vqa-env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Optional Components

### TensorBoard (for training monitoring)

```bash
pip install tensorboard

# Then in train_sft_example.py, change:
# report_to="none" → report_to="tensorboard"

# Run TensorBoard:
tensorboard --logdir PMC-VQA-SFT-Output/runs
```

### Weights & Biases (for experiment tracking)

```bash
pip install wandb
wandb login

# Then in train_sft_example.py, change:
# report_to="none" → report_to="wandb"
```

### Jupyter Notebook Support

```bash
pip install jupyter ipywidgets

# Launch Jupyter
jupyter notebook
```

## Package Details

### Core Packages

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.1.0 | Deep learning framework |
| transformers | ≥4.36.0 | Model architecture and tokenizers |
| datasets | ≥2.16.0 | Dataset loading and processing |
| accelerate | ≥0.25.0 | Distributed training support |
| peft | ≥0.7.0 | LoRA and parameter-efficient fine-tuning |
| bitsandbytes | ≥0.41.0 | 8-bit quantization |
| pandas | ≥2.0.0 | CSV data handling |
| tqdm | ≥4.65.0 | Progress bars |

### Why These Versions?

- **torch ≥2.1.0**: Better memory management, faster training
- **transformers ≥4.36.0**: DeepSeek-R1 support, improved chat templates
- **peft ≥0.7.0**: Latest LoRA improvements
- **bitsandbytes ≥0.41.0**: Stable 8-bit quantization

## Alternative: Docker Installation

If you prefer Docker:

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
WORKDIR /workspace
COPY requirements.txt .
RUN pip3 install -r requirements.txt

CMD ["/bin/bash"]
```

```bash
# Build and run
docker build -t pmc-vqa-sft .
docker run --gpus all -it -v $(pwd):/workspace pmc-vqa-sft
```

## Server/HPC Installation

If you're on a shared HPC system (like Great Lakes):

```bash
# Load required modules
module load python/3.10
module load cuda/12.1
module load gcc/11.2

# Create virtual environment
python3 -m venv ~/pmc-vqa-env
source ~/pmc-vqa-env/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

## Disk Space Requirements

- **Python packages**: ~5-10 GB
- **Model cache**: ~8-10 GB (for DeepSeek-R1-Distill-Llama-8B)
- **Dataset cache**: ~2-5 GB (processed PMC-VQA)
- **Training outputs**: ~10-20 GB (checkpoints and logs)

**Total**: ~30-50 GB recommended

## Network Requirements

First-time setup will download:
- Python packages (~2-5 GB)
- Model weights if not present (~8 GB)
- Dataset if not cached (~1 GB)

Ensure you have:
- Good internet connection for initial setup
- Access to HuggingFace (huggingface.co)
- Access to PyPI (pypi.org)

## Post-Installation Checklist

- [ ] Python 3.8+ installed
- [ ] CUDA available (for GPU training)
- [ ] All packages installed without errors
- [ ] `test_preprocessing.py` runs successfully
- [ ] Sufficient disk space (~50 GB)
- [ ] GPU accessible via PyTorch

## Getting Help

If you encounter installation issues:

1. Check the error message carefully
2. Verify Python and CUDA versions
3. Try creating a fresh virtual environment
4. Use minimal installation method
5. Check package-specific documentation

## Next Steps

After successful installation:

1. Run `python test_preprocessing.py` to verify setup
2. Read `QUICKSTART_GUIDE.md` for usage instructions
3. Start with preprocessing: `python preprocess_pmc_vqa_for_sft.py --sample_size 100`

---

**Installation complete!** 🎉

Ready to start? Run:
```bash
python test_preprocessing.py
```

