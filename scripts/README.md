# Model Setup Scripts

Scripts for installing and benchmarking medical AI models on M4 Pro 24GB.

## Quick Start

### 1. Install Models

```bash
# Install all models (recommended order)
uv run scripts/setup_llava_med.py      # ~7GB, medical vision specialist
uv run scripts/setup_biomistral.py     # ~7GB, medical English specialist  
uv run scripts/setup_sabia.py          # ~7GB, Brazilian Portuguese specialist
```

### 2. Benchmark Performance

```bash
# Test all models
uv run scripts/benchmark_models.py --image imgs/person75_bacteria_365.jpeg

# Output: benchmark_results.json with latency, VRAM, tokens/s
```

## Model Details

### LLaVA-Med v1.5
- **Purpose**: Medical image understanding (chest X-rays)
- **Size**: ~7GB (float16), ~4GB (4-bit quantized)
- **Training**: PMC-15M medical images + PubMed captions
- **Backend**: HuggingFace transformers
- **Usage**: Vision encoder for ensemble system

**Installation:**
```bash
uv run scripts/setup_llava_med.py
# Options:
#   --no-4bit      Use float16 instead of 4-bit (higher VRAM)
#   --skip-test    Skip test inference
```

### BioMistral-7B
- **Purpose**: Medical text generation (English)
- **Size**: ~7GB (float16), ~4GB (4-bit quantized)
- **Training**: PubMed abstracts, medical textbooks
- **Backend**: HuggingFace transformers
- **Usage**: Stage 1 medical analysis (before PT-BR translation)

**Installation:**
```bash
uv run scripts/setup_biomistral.py
# Options:
#   --no-4bit      Use float16 instead of 4-bit
#   --skip-test    Skip test inference
```

### Sabiá-7B
- **Purpose**: Brazilian Portuguese native generation
- **Size**: ~7GB
- **Training**: Brazilian Portuguese corpus (Wikipedia, news, books)
- **Backend**: Ollama (preferred) or HuggingFace transformers
- **Usage**: Stage 2 PT-BR translation (medical EN → natural PT-BR)

**Installation:**
```bash
uv run scripts/setup_sabia.py
# Options:
#   --method ollama|transformers|auto
#   --skip-test    Skip test inference

# If using Ollama (recommended):
# 1. Install Ollama: https://ollama.ai
# 2. Run: ollama pull sabia-7b
```

## Benchmarking

The benchmark script tests:
- **Load time**: Model initialization
- **Inference time**: Time to generate response
- **Throughput**: Tokens per second
- **Memory usage**: VRAM/RAM consumption (transformers only)
- **Output quality**: Character length (proxy for verbosity)

**Example output:**
```
✓ llava:7b (ollama)
  Load time: 3.2s
  Inference time: 5.8s
  Speed: 12.3 tokens/s
  Output length: 450 chars

✓ llava-med-v1.5-mistral-7b (transformers)
  Load time: 12.5s
  Inference time: 8.2s
  Speed: 10.1 tokens/s
  Memory: 6800 MB
  Output length: 380 chars
```

## Architecture Integration

These models integrate into the 3-stage pipeline:

```
Stage 1: Vision Ensemble
├── llava:7b (Ollama)
├── llava:13b (Ollama)  
├── llava-llama3 (Ollama)
└── llava-med (transformers) ← NEW medical specialist
    ↓
Stage 2: Medical Text Generation
└── BioMistral-7B ← NEW medical English
    ↓
Stage 3: PT-BR Translation
└── Sabiá-7B ← NEW Brazilian Portuguese
```

## Troubleshooting

### Ollama models not found
```bash
# Check installed models
ollama list

# Reinstall if needed
ollama pull llava:7b
ollama pull llava:13b
ollama pull llava-llama3
ollama pull sabia-7b  # May not exist yet
```

### Transformers VRAM issues
```bash
# Use 4-bit quantization (default)
uv run scripts/setup_llava_med.py

# Force CPU if MPS fails
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Download failures
```bash
# Check HuggingFace cache
ls ~/.cache/huggingface/hub/

# Clear cache if corrupted
rm -rf ~/.cache/huggingface/hub/models--*
```

## Next Steps

After installation:
1. **Run benchmark**: `uv run scripts/benchmark_models.py`
2. **Verify all models work**: Check `benchmark_results.json`
3. **Move to Day 2**: Build RAG infrastructure (`src/rag/`)
4. **Create advanced prompts**: Model-specific templates (`src/prompts/`)

## References

- LLaVA-Med paper: https://arxiv.org/abs/2306.00890
- BioMistral: https://huggingface.co/BioMistral/BioMistral-7B
- Sabiá: https://huggingface.co/maritaca-ai/sabia-7b
- Ollama docs: https://ollama.ai/library
