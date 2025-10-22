import os
import pandas as pd
from datasets import Dataset, DatasetDict
import urllib.request

target_dir = r'D:\Github\Llama-RL-Private\PMC-VQA'
os.makedirs(target_dir, exist_ok=True)

print(f"📦 Downloading PMC-VQA CSV files...")

# Direct download from HF repo
base_url = "https://huggingface.co/datasets/RadGenome/PMC-VQA/resolve/main/"
files = ["train.csv", "train_2.csv", "test.csv", "test_2.csv"]

# Download manually if not present
for fname in files:
    path = os.path.join(target_dir, fname)
    if not os.path.exists(path):
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(base_url + fname, path)
        print(f"✓ Downloaded {fname}")
    else:
        print(f"✓ {fname} already exists")

# Read CSVs and align columns
def load_and_fix_csv(path):
    df = pd.read_csv(path)
    # Ensure consistent column set
    cols = ['Figure_path', 'Question', 'Choice A', 'Choice B', 'Choice C', 'Choice D', 'Answer']
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

train_df = pd.concat([load_and_fix_csv(os.path.join(target_dir, f)) for f in ["train.csv", "train_2.csv"]], ignore_index=True)
test_df = pd.concat([load_and_fix_csv(os.path.join(target_dir, f)) for f in ["test.csv", "test_2.csv"]], ignore_index=True)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

dataset.save_to_disk(os.path.join(target_dir, "hf_dataset"))
print("✅ PMC-VQA dataset processed and saved successfully!")
