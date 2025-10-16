# Quick Start Guide: PMC-VQA SFT on DeepSeek-R1-Distill-Llama-8B

This guide provides a step-by-step walkthrough to preprocess the PMC-VQA dataset and fine-tune DeepSeek-R1-Distill-Llama-8B.

## 📋 Prerequisites

### 1. Install Required Packages

```bash
pip install transformers datasets torch accelerate peft bitsandbytes pandas tqdm
```

### 2. Verify Paths

Make sure you have:
- ✅ DeepSeek-R1-Distill-Llama-8B model downloaded
- ✅ PMC-VQA dataset downloaded

```bash
ls DeepSeek-R1-Distill-Llama-8B/
# Should show: config.json, tokenizer files, model files, etc.

ls PMC-VQA/
# Should show: train.csv, test.csv, hf_dataset/, etc.
```

## 🚀 Quick Start (3 Steps)

### Step 1: Test the Preprocessing

First, test with a small sample to make sure everything works:

```bash
python test_preprocessing.py
```

This will:
- Process 10 samples from the dataset
- Show you example formatted outputs
- Verify tokenization works correctly
- Save results to `PMC-VQA-Test-Sample/`

**Expected output:**
```
PMC-VQA Preprocessing Test
======================================================================
Initializing preprocessor...
Loading tokenizer from DeepSeek-R1-Distill-Llama-8B...
Tokenizer loaded. Vocab size: 102400
...
✅ TEST PASSED - Preprocessing completed successfully!
```

### Step 2: Preprocess the Full Dataset

Once the test passes, preprocess the full dataset:

```bash
python preprocess_pmc_vqa_for_sft.py
```

**Options:**
```bash
# Default (recommended for first run)
python preprocess_pmc_vqa_for_sft.py

# With custom settings
python preprocess_pmc_vqa_for_sft.py \
    --max_length 4096 \
    --add_reasoning \
    --output_dir PMC-VQA-Processed-Custom
```

**This will take some time** (~30-60 minutes depending on your hardware) as it processes ~227,000 examples.

**Expected output:**
```
Loading dataset from PMC-VQA...
Loaded HF dataset: DatasetDict({
    train: Dataset({
        features: ['Figure_path', 'Question', 'Answer', 'Choice A', ...],
        num_rows: 176949
    })
    test: Dataset({...})
})

Processing train split...
Preprocessing 176949 examples...
Creating conversations: 100%|████████| 176949/176949
Tokenizing dataset...

✅ Preprocessing complete!
Processed dataset saved to: PMC-VQA-Processed/
```

### Step 3: Train the Model

Now you can fine-tune the model:

#### Option A: Quick Test Training (10 minutes)

Test training on a small sample first:

```bash
python train_sft_example.py \
    --max_samples 1000 \
    --num_epochs 1 \
    --batch_size 2 \
    --grad_accum 4
```

#### Option B: Full Training (Several hours)

For full fine-tuning:

```bash
python train_sft_example.py \
    --num_epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --learning_rate 2e-4
```

**Training arguments explained:**
- `--num_epochs 3`: Train for 3 epochs
- `--batch_size 4`: Process 4 samples per GPU
- `--grad_accum 8`: Accumulate gradients over 8 steps (effective batch size = 4 × 8 = 32)
- `--learning_rate 2e-4`: Learning rate for LoRA

**Expected output:**
```
Loading model with 8-bit quantization...
Configuring LoRA...
trainable params: 67,108,864 || all params: 7,584,301,056 || trainable%: 0.8848

Starting training...
Effective batch size: 32
Number of epochs: 3
Learning rate: 0.0002
LoRA enabled: True
8-bit quantization: True

Training: [########............................] 25% | Step 1000/4000 | Loss: 1.234
...
✅ Training completed successfully!
```

## 📊 Monitoring Training

### Using TensorBoard

To monitor training progress with TensorBoard:

1. Modify `train_sft_example.py` to enable TensorBoard:
   - Change `report_to="none"` to `report_to="tensorboard"`

2. Run training and start TensorBoard in a separate terminal:
   ```bash
   tensorboard --logdir PMC-VQA-SFT-Output/runs
   ```

3. Open browser to `http://localhost:6006`

### Using Weights & Biases (wandb)

1. Install wandb:
   ```bash
   pip install wandb
   wandb login
   ```

2. Modify `train_sft_example.py`:
   - Change `report_to="none"` to `report_to="wandb"`

3. Run training and view results at https://wandb.ai

## 🎯 Using the Fine-Tuned Model

After training completes, use your fine-tuned model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model
model_path = "PMC-VQA-SFT-Output/final_model"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prepare a medical question
conversation = [
    {
        "role": "user",
        "content": """You are a medical expert. Answer the following medical question.

Question: What imaging modality was used in this study?

Options:
A: CT Scan
B: MRI
C: X-ray
D: Ultrasound

Provide the correct answer and explain your reasoning."""
    }
]

# Format with chat template
prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 📁 File Structure

After completing all steps, you'll have:

```
mperform/
├── DeepSeek-R1-Distill-Llama-8B/        # Original model
├── PMC-VQA/                              # Original dataset
│   ├── train.csv
│   ├── test.csv
│   └── hf_dataset/
├── PMC-VQA-Processed/                    # Preprocessed dataset
│   ├── dataset_dict.json
│   ├── train/
│   ├── test/
│   └── examples.json                     # Sample formatted examples
├── PMC-VQA-SFT-Output/                   # Training outputs
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   ├── final_model/                      # Your fine-tuned model!
│   │   ├── config.json
│   │   ├── adapter_model.bin             # LoRA weights
│   │   └── ...
│   └── eval_results.json
├── preprocess_pmc_vqa_for_sft.py        # Main preprocessing script
├── train_sft_example.py                  # Training script
├── test_preprocessing.py                 # Test script
└── PMC_VQA_PREPROCESSING_README.md      # Detailed documentation
```

## 🔧 Advanced Configuration

### Custom Preprocessing Options

```bash
# Longer sequences (needs more GPU memory)
python preprocess_pmc_vqa_for_sft.py --max_length 4096

# Add reasoning prompts (for R1's thinking capability)
python preprocess_pmc_vqa_for_sft.py --add_reasoning

# Without multiple choice options
python preprocess_pmc_vqa_for_sft.py --no_choices

# Custom output directory
python preprocess_pmc_vqa_for_sft.py --output_dir /path/to/custom/output
```

### Custom Training Options

```bash
# Full fine-tuning (no LoRA - requires more GPU memory)
python train_sft_example.py --no_lora --full_precision

# Higher learning rate
python train_sft_example.py --learning_rate 5e-4

# More epochs
python train_sft_example.py --num_epochs 5

# Larger effective batch size
python train_sft_example.py --batch_size 8 --grad_accum 16  # effective = 128
```

### Memory Optimization

If you run out of GPU memory:

1. **Reduce batch size:**
   ```bash
   python train_sft_example.py --batch_size 1 --grad_accum 32
   ```

2. **Reduce sequence length during preprocessing:**
   ```bash
   python preprocess_pmc_vqa_for_sft.py --max_length 1024
   ```

3. **Use 8-bit quantization** (default, but if you disabled it):
   ```bash
   python train_sft_example.py  # Already uses 8-bit by default
   ```

## 🐛 Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:**
- Reduce `--batch_size` to 1 or 2
- Reduce `--max_length` during preprocessing
- Use `--max_samples` for testing with fewer examples

### Issue: Training is very slow

**Solution:**
- Increase `--batch_size` if you have GPU memory
- Use multiple GPUs with `torchrun` or `accelerate`
- Reduce `--max_length` to speed up computation

### Issue: Poor model performance

**Solution:**
- Increase `--num_epochs` (try 5-10)
- Adjust `--learning_rate` (try 1e-4 to 5e-4)
- Use `--add_reasoning` during preprocessing for better reasoning

### Issue: Preprocessing fails

**Solution:**
- Check dataset path is correct
- Ensure required packages are installed: `pip install transformers datasets pandas tqdm`
- Try with `--sample_size 100` first to test

## 📈 Expected Results

### Training Metrics

- **Initial loss**: ~2.5-3.0
- **Final loss**: ~1.0-1.5 (after 3 epochs)
- **Training time**: ~8-12 hours for full dataset on single A100 GPU

### Model Performance

After fine-tuning, the model should be able to:
- ✅ Answer medical questions correctly
- ✅ Explain reasoning for answers
- ✅ Handle multiple choice format
- ✅ Understand medical terminology

## 📚 Additional Resources

- **Main Documentation**: `PMC_VQA_PREPROCESSING_README.md`
- **Dataset Info**: [PMC-VQA on HuggingFace](https://huggingface.co/datasets/RadGenome/PMC-VQA)
- **Model Info**: [DeepSeek-R1 Paper](https://github.com/deepseek-ai/DeepSeek-R1)
- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## ❓ Need Help?

1. Check the detailed documentation in `PMC_VQA_PREPROCESSING_README.md`
2. Inspect `examples.json` to verify preprocessing
3. Start with small samples using `--max_samples` flag
4. Review training logs in `PMC-VQA-SFT-Output/`

## ✅ Checklist

- [ ] Install required packages
- [ ] Verify model and dataset paths
- [ ] Run test preprocessing (`test_preprocessing.py`)
- [ ] Preprocess full dataset (`preprocess_pmc_vqa_for_sft.py`)
- [ ] Test training with small sample (`--max_samples 1000`)
- [ ] Run full training (`train_sft_example.py`)
- [ ] Evaluate fine-tuned model
- [ ] Use model for inference

Good luck with your fine-tuning! 🚀

