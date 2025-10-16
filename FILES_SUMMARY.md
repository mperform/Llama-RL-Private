# PMC-VQA SFT Pipeline - Files Summary

This document provides an overview of all scripts and documentation created for preprocessing and fine-tuning DeepSeek-R1-Distill-Llama-8B on the PMC-VQA dataset.

## 📂 Created Files

### 1. Main Scripts

#### `preprocess_pmc_vqa_for_sft.py` ⭐
**Purpose:** Main preprocessing script that converts PMC-VQA dataset to DeepSeek-R1 format

**Key Features:**
- Loads PMC-VQA dataset (CSV or HuggingFace format)
- Formats questions in DeepSeek-R1 chat template
- Tokenizes data for SFT training
- Handles multiple choice options
- Optional reasoning prompts
- Saves processed dataset in HuggingFace format

**Usage:**
```bash
python preprocess_pmc_vqa_for_sft.py
python preprocess_pmc_vqa_for_sft.py --max_length 4096 --add_reasoning
python preprocess_pmc_vqa_for_sft.py --sample_size 1000  # For testing
```

**Input:** 
- PMC-VQA dataset (train.csv, test.csv or hf_dataset/)
- DeepSeek-R1-Distill-Llama-8B tokenizer

**Output:**
- Preprocessed dataset with tokenized inputs
- examples.json with sample formatted outputs

---

#### `train_sft_example.py` ⭐
**Purpose:** Complete training script for supervised fine-tuning

**Key Features:**
- 8-bit quantization support (memory efficient)
- LoRA fine-tuning (parameter efficient)
- Gradient accumulation for large effective batch sizes
- Mixed precision training (BF16)
- Checkpoint saving and best model selection
- Evaluation during training

**Usage:**
```bash
# Quick test
python train_sft_example.py --max_samples 1000 --num_epochs 1

# Full training
python train_sft_example.py --num_epochs 3

# Custom settings
python train_sft_example.py \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --grad_accum 16
```

**Input:**
- Preprocessed PMC-VQA dataset
- DeepSeek-R1-Distill-Llama-8B model

**Output:**
- Fine-tuned model with LoRA adapters
- Training checkpoints
- Evaluation results (eval_results.json)

---

#### `inference_example.py` ⭐
**Purpose:** Inference and evaluation script for the fine-tuned model

**Key Features:**
- Load and use fine-tuned model
- Interactive Q&A mode
- Batch evaluation on test set
- Demo mode with example questions
- Customizable generation parameters

**Usage:**
```bash
# Demo mode
python inference_example.py

# Interactive mode
python inference_example.py --interactive

# Evaluate on test set
python inference_example.py --evaluate --num_samples 50

# Single question
python inference_example.py --question "What imaging modality is shown?"
```

**Input:**
- Fine-tuned model
- Optional: preprocessed dataset for evaluation

**Output:**
- Generated answers
- Optional: evaluation_results.json

---

#### `test_preprocessing.py`
**Purpose:** Quick test script to verify preprocessing works correctly

**Key Features:**
- Tests preprocessing on 10 samples
- Displays formatted examples
- Verifies tokenization
- Quick validation before full preprocessing

**Usage:**
```bash
python test_preprocessing.py
```

**Output:**
- Test sample dataset
- Console output with examples

---

### 2. Documentation

#### `QUICKSTART_GUIDE.md` 📖
**Purpose:** Step-by-step guide for the entire pipeline

**Contents:**
- Prerequisites and installation
- 3-step quick start process
- Training monitoring setup
- Using the fine-tuned model
- Troubleshooting guide
- Advanced configuration options

**Best for:** Getting started quickly

---

#### `PMC_VQA_PREPROCESSING_README.md` 📖
**Purpose:** Detailed documentation for preprocessing

**Contents:**
- Dataset structure explanation
- Model information
- All preprocessing options
- Output format details
- Example usage code
- Training integration examples

**Best for:** Understanding preprocessing in depth

---

#### `FILES_SUMMARY.md` 📖
**Purpose:** This file - overview of all created files

**Contents:**
- Description of each file
- Usage examples
- Input/output information
- Workflow guidance

**Best for:** Understanding the project structure

---

## 🔄 Typical Workflow

### For First-Time Users:

```
1. Read QUICKSTART_GUIDE.md
   ↓
2. Run test_preprocessing.py
   ↓
3. Run preprocess_pmc_vqa_for_sft.py
   ↓
4. Run train_sft_example.py (with --max_samples first)
   ↓
5. Run train_sft_example.py (full training)
   ↓
6. Run inference_example.py (test the model)
```

### For Experienced Users:

```
1. Customize preprocess_pmc_vqa_for_sft.py parameters
   ↓
2. Run preprocessing with custom settings
   ↓
3. Customize train_sft_example.py parameters
   ↓
4. Run full training
   ↓
5. Evaluate and iterate
```

---

## 📊 File Dependencies

```
PMC-VQA Dataset (CSV/HF)
    ├── preprocess_pmc_vqa_for_sft.py
    │       ↓
    │   Preprocessed Dataset
    │       ↓
    ├── train_sft_example.py
    │       ↓
    │   Fine-tuned Model
    │       ↓
    └── inference_example.py
            ↓
        Predictions/Evaluations

DeepSeek-R1-Distill-Llama-8B
    ├── Used by all scripts
    └── Provides tokenizer and base model
```

---

## 🎯 Use Cases

### Use Case 1: Quick Prototyping
**Files:** `test_preprocessing.py` → `train_sft_example.py --max_samples 1000`
**Time:** ~30 minutes
**Purpose:** Test the pipeline quickly

### Use Case 2: Full Fine-Tuning
**Files:** `preprocess_pmc_vqa_for_sft.py` → `train_sft_example.py`
**Time:** ~12-24 hours
**Purpose:** Production-ready fine-tuned model

### Use Case 3: Model Evaluation
**Files:** `inference_example.py --evaluate`
**Time:** ~1 hour for 100 samples
**Purpose:** Assess model performance

### Use Case 4: Interactive Demo
**Files:** `inference_example.py --interactive`
**Time:** Real-time
**Purpose:** Live Q&A demonstrations

---

## 🔧 Customization Points

### Preprocessing Customization
**File:** `preprocess_pmc_vqa_for_sft.py`

**Can modify:**
- `PMCVQAPreprocessor.create_conversation()` - Change prompt format
- `PMCVQAPreprocessor.format_multiple_choice()` - Adjust choice formatting
- `--max_length` - Sequence length
- `--include_choices` / `--add_reasoning` - Prompt style

### Training Customization
**File:** `train_sft_example.py`

**Can modify:**
- `LoraConfig` parameters - r, alpha, dropout
- `TrainingArguments` - All training hyperparameters
- `setup_model_and_tokenizer()` - Quantization settings
- Optimizer, scheduler, etc.

### Inference Customization
**File:** `inference_example.py`

**Can modify:**
- `MedicalQAInference.format_question()` - Prompt formatting
- Generation parameters - temperature, top_p, max_tokens
- Evaluation metrics

---

## 📈 Expected Performance

### Preprocessing
- **Time:** ~30-60 minutes for full dataset (~227K examples)
- **Memory:** ~16-32 GB RAM
- **Output size:** ~2-5 GB (tokenized dataset)

### Training (with LoRA + 8-bit)
- **Time:** ~8-12 hours on single A100 GPU
- **Memory:** ~24-40 GB GPU VRAM
- **Trainable params:** ~67M (0.88% of total)

### Inference
- **Speed:** ~50-100 tokens/second (on A100)
- **Memory:** ~10-15 GB GPU VRAM (8-bit mode)
- **Latency:** ~2-5 seconds per question

---

## 🐛 Common Issues and Solutions

### Issue: Import errors
**Files affected:** All Python scripts
**Solution:** 
```bash
pip install transformers datasets torch accelerate peft bitsandbytes pandas tqdm
```

### Issue: Out of memory during preprocessing
**Files affected:** `preprocess_pmc_vqa_for_sft.py`
**Solution:** Use `--sample_size` or reduce `--max_length`

### Issue: Out of memory during training
**Files affected:** `train_sft_example.py`
**Solution:** Reduce `--batch_size`, use `--grad_accum`, ensure 8-bit mode is on

### Issue: Slow training
**Files affected:** `train_sft_example.py`
**Solution:** Increase batch size if memory allows, use multiple GPUs, reduce max_length

### Issue: Model not loading
**Files affected:** `inference_example.py`
**Solution:** Check model path, ensure training completed, verify LoRA adapters exist

---

## 📦 Required Packages

```bash
# Core packages
pip install transformers>=4.36.0
pip install datasets>=2.16.0
pip install torch>=2.1.0

# Training essentials
pip install accelerate>=0.25.0
pip install peft>=0.7.0
pip install bitsandbytes>=0.41.0

# Utilities
pip install pandas>=2.0.0
pip install tqdm>=4.65.0

# Optional: Monitoring
pip install tensorboard  # For TensorBoard logging
pip install wandb        # For Weights & Biases logging
```

---

## 🎓 Learning Resources

### To understand the preprocessing:
1. Read `PMC_VQA_PREPROCESSING_README.md`
2. Inspect `preprocess_pmc_vqa_for_sft.py`
3. Review `examples.json` after running

### To understand the training:
1. Read LoRA paper: https://arxiv.org/abs/2106.09685
2. Review `train_sft_example.py` comments
3. Check HuggingFace Trainer docs

### To understand the model:
1. Read DeepSeek-R1 paper
2. Check `DeepSeek-R1-Distill-Llama-8B/README.md`
3. Explore the chat template in tokenizer_config.json

---

## ✅ Validation Checklist

Before running the full pipeline, verify:

- [ ] All required packages installed
- [ ] DeepSeek-R1-Distill-Llama-8B model downloaded
- [ ] PMC-VQA dataset downloaded
- [ ] `test_preprocessing.py` runs successfully
- [ ] Sufficient disk space (~10 GB for processed data + checkpoints)
- [ ] Sufficient GPU memory (at least 24 GB recommended)
- [ ] CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 📞 Support

If you encounter issues:

1. **Check documentation:**
   - `QUICKSTART_GUIDE.md` for general help
   - `PMC_VQA_PREPROCESSING_README.md` for preprocessing issues
   - This file for file-specific information

2. **Review examples:**
   - `examples.json` for preprocessing output
   - `eval_results.json` for training metrics
   - `evaluation_results.json` for inference results

3. **Test with small samples:**
   - Use `--sample_size 10` for preprocessing
   - Use `--max_samples 100` for training
   - Use `--num_samples 5` for evaluation

4. **Check logs:**
   - Training logs in `PMC-VQA-SFT-Output/`
   - Console output from each script
   - Error messages and stack traces

---

## 🎉 Success Criteria

You've successfully completed the pipeline when:

- ✅ Preprocessing creates `PMC-VQA-Processed/` with examples.json
- ✅ Training creates `PMC-VQA-SFT-Output/final_model/` with model files
- ✅ Evaluation loss decreases during training (from ~2.5 to ~1.5)
- ✅ Inference generates reasonable medical answers
- ✅ Model correctly answers multiple-choice questions

---

## 📝 Notes

- All scripts use absolute paths by default for the server environment
- Scripts are designed to be run from the `mperform/` directory
- LoRA adapters are much smaller than full models (~200 MB vs ~8 GB)
- 8-bit quantization reduces memory by ~50% with minimal quality loss
- The chat template is automatically applied by the tokenizer

---

**Created:** October 2025  
**For:** ECE598 Biomedical AI Course Project  
**Purpose:** Medical VQA Fine-tuning Pipeline

