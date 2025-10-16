# PMC-VQA Fine-Tuning for DeepSeek-R1-Distill-Llama-8B

Complete pipeline for preprocessing the PMC-VQA medical visual question-answering dataset and fine-tuning the DeepSeek-R1-Distill-Llama-8B model.

## 🎯 Overview

This repository contains a complete, production-ready pipeline for:
1. **Preprocessing** the PMC-VQA dataset into DeepSeek-R1 chat format
2. **Fine-tuning** DeepSeek-R1-Distill-Llama-8B using supervised fine-tuning (SFT)
3. **Evaluating** and using the fine-tuned model for medical question answering

### What is PMC-VQA?

PMC-VQA is a large-scale medical visual question-answering dataset containing:
- 📊 **227,000** VQA pairs
- 🖼️ **149,000** medical images
- 🏥 Multiple medical imaging modalities (CT, MRI, X-ray, etc.)
- ❓ Multiple-choice medical questions

### What is DeepSeek-R1-Distill-Llama-8B?

A reasoning-capable language model:
- 🧠 **8 billion** parameters
- 🎓 Distilled from DeepSeek-R1 (671B parameters)
- 💭 Advanced reasoning with chain-of-thought
- 📝 **16,384** token context length

## 📁 Repository Structure

```
mperform/
├── 🟢 START HERE 👇
│   ├── README_PMC_VQA_SFT.md          ← You are here!
│   └── QUICKSTART_GUIDE.md            ← Step-by-step guide
│
├── 📚 Scripts
│   ├── preprocess_pmc_vqa_for_sft.py  ← Main preprocessing script
│   ├── train_sft_example.py           ← Training script
│   ├── inference_example.py           ← Inference & evaluation
│   └── test_preprocessing.py          ← Quick test script
│
├── 📖 Documentation
│   ├── PMC_VQA_PREPROCESSING_README.md  ← Detailed preprocessing docs
│   └── FILES_SUMMARY.md                 ← Overview of all files
│
├── 📦 Data & Models
│   ├── DeepSeek-R1-Distill-Llama-8B/  ← Base model
│   └── PMC-VQA/                        ← Dataset
│
└── 🎯 Outputs (created during workflow)
    ├── PMC-VQA-Processed/              ← Preprocessed dataset
    └── PMC-VQA-SFT-Output/             ← Fine-tuned model
```

## 🚀 Quick Start (4 Commands)

### 0️⃣ Install & Verify
```bash
pip install -r requirements.txt
python verify_installation.py
```
Installs dependencies and verifies setup (~5 minutes)

### 1️⃣ Test Preprocessing
```bash
python test_preprocessing.py
```
Verifies everything works with 10 samples (~30 seconds)

### 2️⃣ Preprocess Full Dataset
```bash
python preprocess_pmc_vqa_for_sft.py
```
Processes all 227K examples (~30-60 minutes)

### 3️⃣ Fine-Tune Model
```bash
python train_sft_example.py --num_epochs 3
```
Trains the model (~8-12 hours on A100)

**That's it!** Your fine-tuned model will be in `PMC-VQA-SFT-Output/final_model/`

## 📚 Documentation Guide

**Choose your path:**

### 🟢 I want to get started quickly
👉 Read: **[QUICKSTART_GUIDE.md](QUICKSTART_GUIDE.md)**
- Step-by-step instructions
- Common configurations
- Troubleshooting tips

### 🔵 I want to understand preprocessing
👉 Read: **[PMC_VQA_PREPROCESSING_README.md](PMC_VQA_PREPROCESSING_README.md)**
- Dataset format details
- Tokenization process
- Customization options

### 🟡 I want to see all files and their purposes
👉 Read: **[FILES_SUMMARY.md](FILES_SUMMARY.md)**
- Complete file descriptions
- Usage examples
- Dependencies

### 🟠 I want to dive into the code
👉 Start with: **preprocess_pmc_vqa_for_sft.py**
- Well-documented code
- Modular design
- Easy to customize

## 💻 System Requirements

### Minimum (for testing)
- **GPU:** 16 GB VRAM (e.g., V100)
- **RAM:** 32 GB
- **Storage:** 20 GB free

### Recommended (for full training)
- **GPU:** 40 GB VRAM (e.g., A100)
- **RAM:** 64 GB
- **Storage:** 50 GB free

### Software
- **Python:** 3.8+
- **CUDA:** 11.8+ or 12.1+
- **PyTorch:** 2.1.0+

## 📦 Installation

### Quick Install

```bash
# Install all requirements at once
pip install -r requirements.txt
```

### Verify Installation

```bash
# Run verification script
python verify_installation.py
```

### Manual Install (if needed)

```bash
# Core packages
pip install transformers>=4.36.0 datasets>=2.16.0 torch>=2.1.0

# Training essentials
pip install accelerate>=0.25.0 peft>=0.7.0 bitsandbytes>=0.41.0

# Utilities
pip install pandas tqdm
```

📖 **For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md)**

## 🎓 Usage Examples

### Example 1: Quick Prototype (30 minutes)
```bash
# Test on 1000 samples
python preprocess_pmc_vqa_for_sft.py --sample_size 1000 --output_dir PMC-VQA-Test
python train_sft_example.py --dataset_path PMC-VQA-Test --max_samples 1000 --num_epochs 1
python inference_example.py --model_path PMC-VQA-SFT-Output/final_model --interactive
```

### Example 2: Full Production Training (24 hours)
```bash
# Full dataset, 3 epochs
python preprocess_pmc_vqa_for_sft.py --max_length 2048 --add_reasoning
python train_sft_example.py --num_epochs 3 --batch_size 4 --grad_accum 8
python inference_example.py --evaluate --num_samples 100
```

### Example 3: Custom Medical Domain Adaptation
```bash
# Longer context, more epochs
python preprocess_pmc_vqa_for_sft.py --max_length 4096 --add_reasoning
python train_sft_example.py --num_epochs 5 --learning_rate 1e-4
```

## 🔬 Key Features

### Preprocessing (`preprocess_pmc_vqa_for_sft.py`)
- ✅ Automatic chat template formatting
- ✅ Support for multiple-choice questions
- ✅ Customizable prompt engineering
- ✅ Optional reasoning prompts
- ✅ Efficient tokenization
- ✅ Dataset validation with examples

### Training (`train_sft_example.py`)
- ✅ **LoRA** for parameter-efficient fine-tuning (0.88% params)
- ✅ **8-bit quantization** for memory efficiency
- ✅ **Gradient accumulation** for large effective batch sizes
- ✅ **Mixed precision** (BF16) training
- ✅ **Checkpointing** and best model selection
- ✅ **Evaluation** during training

### Inference (`inference_example.py`)
- ✅ Interactive Q&A mode
- ✅ Batch evaluation on test set
- ✅ Customizable generation parameters
- ✅ Demo mode with examples
- ✅ Single-question API

## 📊 Expected Results

### Training Metrics
| Metric | Initial | After 3 Epochs |
|--------|---------|----------------|
| Loss   | ~2.5    | ~1.2-1.5       |
| Perplexity | ~12 | ~3.5-4.5 |

### Performance
- **Accuracy:** Improved medical QA performance
- **Reasoning:** Better explanations for answers
- **Consistency:** More reliable on medical terminology

## 🎯 Use Cases

### 1. Medical Education
Fine-tune on medical textbook QA for student training tools

### 2. Clinical Decision Support
Adapt for specific medical imaging modalities

### 3. Research
Baseline for multimodal medical VQA research

### 4. Benchmarking
Compare text-only vs vision-language models

## 🔧 Customization

### Modify Prompt Format
Edit `PMCVQAPreprocessor.create_conversation()` in `preprocess_pmc_vqa_for_sft.py`

### Adjust Training Hyperparameters
Modify `create_training_args()` in `train_sft_example.py`

### Change LoRA Configuration
Edit `LoraConfig` parameters in `train_sft_example.py`

### Customize Generation
Adjust parameters in `MedicalQAInference.generate_answer()` in `inference_example.py`

## 🐛 Troubleshooting

### Out of Memory?
```bash
# Reduce batch size
python train_sft_example.py --batch_size 1 --grad_accum 32

# Reduce sequence length
python preprocess_pmc_vqa_for_sft.py --max_length 1024
```

### Slow Training?
```bash
# Use gradient checkpointing (already enabled)
# Reduce sequence length
# Use multiple GPUs with torchrun
```

### Model Not Learning?
```bash
# Increase learning rate
python train_sft_example.py --learning_rate 5e-4

# More epochs
python train_sft_example.py --num_epochs 10

# Check data quality in examples.json
```

## 📈 Performance Tips

### For Faster Preprocessing
- Use `--max_length 1024` for shorter sequences
- Run on machine with fast SSD
- Use multiple CPU cores (default: 4)

### For Better Training
- Use larger effective batch size (batch_size × grad_accum)
- Tune learning rate (try 1e-4 to 5e-4)
- Use more epochs for smaller datasets
- Monitor eval loss to avoid overfitting

### For Better Inference
- Use `temperature=0.7` for balanced creativity
- Use `top_p=0.9` for nucleus sampling
- Increase `max_new_tokens` for detailed explanations
- Use `do_sample=False` for deterministic outputs

## 🔬 Advanced Topics

### Multi-GPU Training
```bash
torchrun --nproc_per_node=4 train_sft_example.py
```

### Custom Dataset Format
Modify `load_dataset()` in `preprocess_pmc_vqa_for_sft.py`

### Integration with Vision Models
Combine with CLIP or vision transformers for true multimodal VQA

### Continual Learning
Use the fine-tuned model as base for further domain adaptation

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{pmc-vqa-sft-2025,
  title={PMC-VQA Fine-Tuning Pipeline for DeepSeek-R1-Distill-Llama-8B},
  author={ECE598 Biomedical AI},
  year={2025},
  howpublished={\url{https://github.com/...}}
}
```

And cite the original papers:

```bibtex
@article{zhang2023pmc,
  title={PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering},
  author={Zhang, Xiaoman and others},
  journal={arXiv preprint arXiv:2305.10415},
  year={2023}
}

@article{deepseek2025r1,
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={DeepSeek-AI},
  year={2025}
}
```

## 🤝 Contributing

This is an educational project for ECE598. For improvements:
1. Test thoroughly with `--max_samples`
2. Document changes clearly
3. Update relevant documentation files

## 📄 License

- **Code:** Educational use (ECE598 Course Project)
- **PMC-VQA Dataset:** See original dataset license
- **DeepSeek-R1:** MIT License

## 🎓 Course Information

**Course:** ECE598 - Biomedical AI  
**Institution:** University of Michigan  
**Date:** October 2025  
**Purpose:** Course Project - Medical AI Development

## 🌟 Acknowledgments

- **PMC-VQA Dataset:** RadGenome team
- **DeepSeek-R1:** DeepSeek AI team
- **HuggingFace:** Transformers and PEFT libraries
- **Course Instructors:** ECE598 teaching team

## 📞 Support & Contact

For issues or questions:
1. Check the documentation (QUICKSTART_GUIDE.md, FILES_SUMMARY.md)
2. Review examples.json after preprocessing
3. Test with small samples first (`--sample_size`, `--max_samples`)
4. Check training logs and error messages

## ✅ Next Steps

1. **New users:** Start with [QUICKSTART_GUIDE.md](QUICKSTART_GUIDE.md)
2. **Run test:** `python test_preprocessing.py`
3. **Read examples:** Check `examples.json` after preprocessing
4. **Start training:** Begin with `--max_samples 1000`
5. **Evaluate:** Use `inference_example.py --interactive`

---

**Ready to get started?** Run:
```bash
python test_preprocessing.py
```

**Need help?** Read:
- 🟢 [QUICKSTART_GUIDE.md](QUICKSTART_GUIDE.md) - Get started fast
- 🔵 [PMC_VQA_PREPROCESSING_README.md](PMC_VQA_PREPROCESSING_README.md) - Preprocessing details  
- 🟡 [FILES_SUMMARY.md](FILES_SUMMARY.md) - All files explained

**Happy fine-tuning! 🚀🏥🤖**

