#!/usr/bin/env python3
"""Quick test for BioMistral chat template fix."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Test prompt
test_prompt = """You are a radiologist writing a professional chest X-ray report.

INPUT DATA:
1. Vision Analysis: The chest X-ray shows bilateral lung opacities.
2. CNN Classification: PNEUMONIA (Confidence: 95.0%)

TASK: Generate a concise, professional radiology report in English.

Generate the radiology report:"""

# Load BioMistral
print("Loading BioMistral...")
tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
model = AutoModelForCausalLM.from_pretrained(
    "BioMistral/BioMistral-7B",
    torch_dtype=torch.float16,
    device_map="auto",
)

print("\n=== TEST 1: Old method (raw text) ===")
inputs_old = tokenizer(test_prompt, return_tensors="pt")
inputs_old = {k: v.to(model.device) for k, v in inputs_old.items()}

with torch.no_grad():
    outputs_old = model.generate(
        **inputs_old,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

result_old = tokenizer.decode(
    outputs_old[0][inputs_old["input_ids"].shape[1] :], skip_special_tokens=True
)
print(f"Output (raw): {result_old[:200]}")

print("\n=== TEST 2: New method (chat template) ===")
formatted_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": test_prompt}],
    tokenize=False,
    add_generation_prompt=True,
)
print(f"Formatted prompt: {formatted_prompt[:150]}...")

inputs_new = tokenizer(formatted_prompt, return_tensors="pt")
inputs_new = {k: v.to(model.device) for k, v in inputs_new.items()}

with torch.no_grad():
    outputs_new = model.generate(
        **inputs_new,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

result_new = tokenizer.decode(
    outputs_new[0][inputs_new["input_ids"].shape[1] :], skip_special_tokens=True
)
print(f"Output (chat): {result_new[:200]}")

print("\n=== COMPARISON ===")
print(f"Old method length: {len(result_old)} chars")
print(f"New method length: {len(result_new)} chars")
print(f"Different: {result_old.strip() != result_new.strip()}")
