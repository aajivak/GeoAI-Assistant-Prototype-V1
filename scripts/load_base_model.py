"""Load the unchanged Qwen base model used by GeoAI Assistant Prototype V1.

This repository contains integration documentation, not standalone model weights.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    messages = [
        {"role": "system", "content": "Answer only using the supplied evidence."},
        {"role": "user", "content": (
            "Evidence: outdoor running is not recommended because air quality "
            "is unfavorable. Explain the recommendation briefly."
        )},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ))


if __name__ == "__main__":
    main()
