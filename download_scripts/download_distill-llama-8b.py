from huggingface_hub import snapshot_download
import os

# Target directory (your Great Lakes scratch path)
target_dir = "/scratch/ece598f25s002_class_root/ece598f25s002_class/mperform/DeepSeek-R1-Distill-Llama-8B"
os.makedirs(target_dir, exist_ok=True)

print(f"📦 Downloading deepseek-ai/DeepSeek-R1-Distill-Llama-8B to: {target_dir}")

# Download the entire model repository
snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    resume_download=True,  # resume if interrupted
)

print("✅ Download complete!")
