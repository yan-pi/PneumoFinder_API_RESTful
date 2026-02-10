# LLaVA-Med Integration

## Overview

LLaVA-Med is a medical vision-language model trained on biomedical image-text pairs. This document describes the integration with PneumoFinder's 2-stage pipeline.

## Model Details

- **Model**: `microsoft/llava-med-v1.5-mistral-7b`
- **Base LLM**: Mistral-7B
- **Vision Encoder**: CLIP ViT
- **Parameters**: ~7B
- **Memory**: ~14GB (requires model splitting on MPS)

## Hardware Requirements

| Platform | RAM Required | Notes |
|----------|--------------|-------|
| CUDA (NVIDIA GPU) | 16GB VRAM | Native support |
| MPS (Apple Silicon) | 24GB unified | Model split across MPS + CPU |
| CPU only | 32GB RAM | Very slow, not recommended |

## Installation

### 1. Clone LLaVA-Med Repository

```bash
cd ~/www/tcc
git clone https://github.com/microsoft/LLaVA-Med.git
```

### 2. Apply MPS Compatibility Patches

The following files need modifications for Apple Silicon (MPS) compatibility:

#### `llava/model/builder.py`

Key changes:
- Use `device_map="auto"` with `max_memory` constraints
- Replace `use_flash_attention_2` with `attn_implementation="eager"`
- Keep vision tower on CPU to save MPS memory

```python
# For MPS (Apple Silicon)
if device == "mps":
    kwargs["device_map"] = "auto"
    kwargs["max_memory"] = {"mps": "10GiB", "cpu": "20GiB"}
```

#### `llava/model/language_model/llava_mistral.py`

Add `cache_position` parameter to `forward()` for transformers >= 4.40:

```python
def forward(
    self,
    ...,
    cache_position: Optional[torch.LongTensor] = None,  # Added
) -> Union[Tuple, CausalLMOutputWithPast]:
```

### 3. Protobuf Compatibility

Set environment variable before loading:

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

## Usage in PneumoFinder

### Pipeline Configuration

```python
from src.core.medical_pipeline import MedicalPipeline

# Use LLaVA-Med for vision analysis
pipeline = MedicalPipeline(
    vision_model='llava-med',  # 'llava-med' or 'llava-llama3'
    use_rag=True,
)
```

### Stage 1: Vision Analysis

```python
cnn_diagnosis = {'prediction': 'PNEUMONIA', 'confidence': 0.95}
result = pipeline.stage1_vision_analysis('xray.jpeg', cnn_diagnosis)

print(result['findings_en'])
# "The chest X-ray shows bilateral infiltrates..."
```

## Performance Comparison

| Model | Latency (Stage 1) | Memory | Medical Accuracy |
|-------|-------------------|--------|-----------------|
| LLaVA-Med (Mistral-7B) | 30-40s | 14GB | High (trained on PubMed) |
| llava-llama3 (Ollama) | 8-15s | 6GB | Moderate |

### Test Results

| Test Case | LLaVA-Med Finding | Ground Truth | Correct |
|-----------|-------------------|--------------|---------|
| case_01_normal_clear | "bilateral clear lung fields" | NORMAL | ✓ |
| case_02_pneumonia_severe | "bilateral infiltrates" | PNEUMONIA | ✓ |

## Known Issues

### 1. Memory Fragmentation

After multiple inferences, MPS memory may fragment. Solution:

```python
pipeline._unload_llava_med()  # Explicit unload
torch.mps.empty_cache()       # Clear MPS cache
```

### 2. Slow First Inference

The first inference takes longer due to model loading (~15s load + 35s inference).
Subsequent inferences reuse the cached model.

### 3. Protobuf Errors

If you see "Descriptors cannot be created directly" errors:

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

## Files Modified in LLaVA-Med

| File | Changes |
|------|---------|
| `llava/model/builder.py` | MPS device_map, max_memory, attn_implementation |
| `llava/model/language_model/llava_mistral.py` | cache_position parameter |

## RAG Integration

LLaVA-Med works with the RAG system for grounded medical responses:

```python
# RAG context is automatically injected into the prompt
rag_context = retriever.build_rag_context(
    diagnosis="pneumonia",
    include_guidelines=True,
)
```

## References

- [LLaVA-Med Paper](https://arxiv.org/abs/2306.00890)
- [LLaVA-Med GitHub](https://github.com/microsoft/LLaVA-Med)
- [Mistral-7B](https://mistral.ai/technology/)
