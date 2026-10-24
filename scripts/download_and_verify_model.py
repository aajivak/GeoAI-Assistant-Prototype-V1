r"""Script to download Qwen/Qwen2.5-1.5B-Instruct to D:\AI_V2\.cache\huggingface and verify CPU inference."""
import os
import time
import psutil

# Ensure HF_HOME is set before importing transformers
os.environ["HF_HOME"] = r"D:\AI_V2\.cache\huggingface"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR = r"D:\AI_V2\.cache\huggingface"

print("=" * 65)
print(f"DOWNLOADING & VERIFYING {MODEL_ID}")
print(f"Target Cache Directory: {CACHE_DIR}")
print("=" * 65)

proc = psutil.Process()
ram_before_mb = proc.memory_info().rss / (1024 * 1024)
print(f"\nInitial Process RAM: {ram_before_mb:.2f} MB")

# 1. Download/Load Tokenizer
print("\n[1/3] Loading Tokenizer...")
t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
print(f"  --> Tokenizer loaded in {time.perf_counter() - t0:.2f}s")

# 2. Download/Load Model
print("\n[2/3] Loading Model Weights into CPU memory (bfloat16)...")
t0 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model.eval()
load_time = time.perf_counter() - t0

ram_after_mb = proc.memory_info().rss / (1024 * 1024)
model_ram_mb = ram_after_mb - ram_before_mb

print(f"  --> Model loaded in {load_time:.2f}s")
print(f"  --> RAM after loading: {ram_after_mb:.2f} MB")
print(f"  --> Net Model RAM Footprint: {model_ram_mb:.2f} MB (~{model_ram_mb / 1024:.2f} GB)")

# 3. Test CPU Inference
print("\n[3/3] Testing CPU generation and measuring tokens/second...")
test_prompt = (
    "<|im_start|>system\nYou are a GeoAI assistant.<|im_end|>\n"
    "<|im_start|>user\nRecommend a good morning activity in one sentence.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
inputs = tokenizer(test_prompt, return_tensors="pt")
in_tokens = inputs["input_ids"].shape[1]

t_gen_start = time.perf_counter()
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        temperature=0.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
gen_duration = time.perf_counter() - t_gen_start
gen_tokens = outputs[0][in_tokens:]
num_gen = len(gen_tokens)
tok_per_sec = num_gen / max(0.01, gen_duration)

result_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

print(f"  --> Generated {num_gen} tokens in {gen_duration:.2f}s ({tok_per_sec:.2f} tokens/sec)")
print(f"  --> Output: \"{result_text}\"")

# Check on-disk cache size
print("\nChecking D: Drive Cached Files:")
cache_files = []
total_bytes = 0
for root, dirs, files in os.walk(CACHE_DIR):
    for f in files:
        p = os.path.join(root, f)
        sz = os.path.getsize(p)
        total_bytes += sz
        cache_files.append((f, sz))

print(f"  --> Total cached files: {len(cache_files)}")
print(f"  --> Total on-disk size: {total_bytes / (1024 * 1024):.2f} MB ({total_bytes / (1024 * 1024 * 1024):.2f} GB)")

print("\n" + "=" * 65)
print("MODEL VERIFICATION SUCCESSFUL!")
print("=" * 65)
