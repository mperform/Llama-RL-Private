# PMC-VQA Dataset Preprocessing for DeepSeek-R1-Distill-Llama-8B

This README explains how to preprocess the PMC-VQA dataset for supervised fine-tuning (SFT) on the DeepSeek-R1-Distill-Llama-8B model.

## Overview

The PMC-VQA (PubMed Central Visual Question Answering) dataset contains medical images with associated questions and multiple-choice answers. This preprocessing script converts the dataset into the chat template format expected by DeepSeek-R1-Distill-Llama-8B.

### Dataset Structure

The PMC-VQA dataset has the following columns:
- `Figure_path`: Path/filename of the medical image
- `Question`: The medical question about the image
- `Answer`: The correct answer
- `Choice A`, `Choice B`, `Choice C`, `Choice D`: Multiple choice options
- `Answer_label`: The letter of the correct answer (A, B, C, or D)

### Model Information

**DeepSeek-R1-Distill-Llama-8B** is a reasoning-capable language model distilled from DeepSeek-R1. Key features:
- 8B parameters (Llama-based architecture)
- Max context length: 16,384 tokens
- Special chat template with reasoning capabilities
- Special tokens: `<｜begin▁of▁sentence｜>`, `<｜end▁of▁sentence｜>`, `<｜User｜>`, `<｜Assistant｜>`

## Installation

Make sure you have the required packages installed:

```bash
pip install transformers datasets pandas tqdm torch
```

## Usage

### Basic Usage

Run the preprocessing script with default settings:

```bash
python preprocess_pmc_vqa_for_sft.py
```

This will:
1. Load the PMC-VQA dataset from the default path
2. Convert it to the DeepSeek-R1 chat format
3. Tokenize the data with max length 2048
4. Save the processed dataset to `PMC-VQA-Processed/`

### Advanced Usage

#### Test with a Small Sample

To test the preprocessing on a small sample (e.g., 100 examples):

```bash
python preprocess_pmc_vqa_for_sft.py --sample_size 100 --output_dir PMC-VQA-Sample
```

#### Custom Paths

Specify custom paths for the model and dataset:

```bash
python preprocess_pmc_vqa_for_sft.py \
    --model_path /path/to/DeepSeek-R1-Distill-Llama-8B \
    --dataset_path /path/to/PMC-VQA \
    --output_dir /path/to/output
```

#### Without Multiple Choice Options

To exclude multiple choice options from the prompts:

```bash
python preprocess_pmc_vqa_for_sft.py --no_choices
```

#### With Reasoning Prompts

To add reasoning prompts (leveraging DeepSeek-R1's thinking capabilities):

```bash
python preprocess_pmc_vqa_for_sft.py --add_reasoning
```

#### Custom Max Length

Specify a different maximum sequence length:

```bash
python preprocess_pmc_vqa_for_sft.py --max_length 4096
```

### All Arguments

```
--model_path        Path to DeepSeek-R1-Distill-Llama-8B model directory
--dataset_path      Path to PMC-VQA dataset directory
--output_dir        Output directory for preprocessed dataset
--max_length        Maximum sequence length (default: 2048)
--sample_size       Number of samples for testing (default: use all)
--no_choices        Don't include multiple choice options
--add_reasoning     Add reasoning prompts for thinking
```

## Output Format

The script generates:

1. **Preprocessed Dataset** (`dataset_dict.json` + arrow files)
   - Tokenized inputs with `input_ids`, `attention_mask`, and `labels`
   - Saved in HuggingFace datasets format
   - Ready for SFT training

2. **Examples File** (`examples.json`)
   - Contains 3 examples from each split
   - Shows the formatted text and tokenization
   - Useful for verification and debugging

## Example Output

### Input (CSV row):
```csv
Figure_path: PMC1064097_F1.jpg
Question: What is the uptake pattern in the breast?
Answer: Focal uptake pattern
Choice A: Diffuse uptake pattern
Choice B: Focal uptake pattern
Choice C: No uptake pattern
Choice D: Cannot determine from the information given
Answer_label: B
```

### Output (formatted for DeepSeek-R1):
```
<｜begin▁of▁sentence｜>You are a medical expert. Answer the following medical question based on the image.

Image: PMC1064097_F1.jpg

Question: What is the uptake pattern in the breast?

Options:
A: Diffuse uptake pattern
B: Focal uptake pattern
C: No uptake pattern
D: Cannot determine from the information given

Provide the correct answer and explain your reasoning.<｜User｜><｜Assistant｜>Focal uptake pattern<｜end▁of▁sentence｜>
```

## Using the Preprocessed Dataset

### Loading the Dataset

```python
from datasets import load_from_disk

# Load the preprocessed dataset
dataset = load_from_disk("PMC-VQA-Processed")

# Access train and test splits
train_data = dataset['train']
test_data = dataset['test']

print(f"Train size: {len(train_data)}")
print(f"Test size: {len(test_data)}")

# Access a sample
sample = train_data[0]
print(f"Input IDs shape: {len(sample['input_ids'])}")
print(f"Question: {sample['question']}")
print(f"Answer: {sample['answer']}")
```

### Training with the Dataset

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

# Load model and tokenizer
model_path = "DeepSeek-R1-Distill-Llama-8B"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load preprocessed dataset
dataset = load_from_disk("PMC-VQA-Processed")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./pmc-vqa-sft-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
    fp16=True,  # or bf16=True for better precision
    evaluation_strategy="steps",
    load_best_model_at_end=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)

# Start training
trainer.train()
```

## Dataset Statistics

- **Train set**: ~176,949 examples
- **Test set**: ~50,001 examples
- **Total**: ~227,000 VQA pairs
- **Images**: ~149,000 medical images
- **Modalities**: Various medical imaging types (CT, MRI, X-ray, etc.)

## Notes and Considerations

### Image Handling

Since DeepSeek-R1-Distill-Llama-8B is a **text-only** model, it cannot process images directly. The preprocessing script includes the image filename in the prompt as reference, but the model will rely on the text description in the question.

For true multimodal VQA, you would need a vision-language model. This preprocessing is suitable for:
1. Learning to answer medical questions from text descriptions
2. Baseline experiments for text-based medical QA
3. Comparing with multimodal approaches

### Memory and Performance

- Default max_length of 2048 tokens is sufficient for most examples
- Larger sequences may require more GPU memory during training
- Consider using gradient accumulation to simulate larger batch sizes
- Use `--sample_size` for quick testing before full preprocessing

### Quality Checks

Always inspect the `examples.json` file after preprocessing to verify:
- Proper chat template formatting
- Correct tokenization
- Appropriate sequence lengths
- Preservation of medical information

## Troubleshooting

### ModuleNotFoundError

If you get `ModuleNotFoundError: No module named 'datasets'`:
```bash
pip install datasets transformers
```

### Out of Memory

If you run out of memory during preprocessing:
- Reduce `--max_length` (e.g., to 1024)
- Process in smaller batches by modifying `batch_size` in the code
- Use `--sample_size` to process fewer examples

### Tokenizer Warnings

If you see warnings about `pad_token`, these are expected and handled automatically.

## References

- **PMC-VQA Dataset**: [HuggingFace](https://huggingface.co/datasets/RadGenome/PMC-VQA)
- **DeepSeek-R1**: [Paper](https://github.com/deepseek-ai/DeepSeek-R1)
- **DeepSeek-R1-Distill-Llama-8B**: [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)

## Citation

If you use this preprocessing script or the PMC-VQA dataset, please cite:

```bibtex
@article{zhang2023pmc,
  title={PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering},
  author={Zhang, Xiaoman and others},
  journal={arXiv preprint arXiv:2305.10415},
  year={2023}
}

@article{deepseek2025,
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={DeepSeek-AI},
  year={2025}
}
```

## License

This preprocessing script is provided for educational purposes. Please refer to the original licenses:
- PMC-VQA Dataset: [Dataset License]
- DeepSeek-R1: MIT License

