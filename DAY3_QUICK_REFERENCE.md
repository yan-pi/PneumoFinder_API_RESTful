# Day 3 Quick Reference

## Problem Solved
```
RuntimeError: Invalid buffer size: 13.24 GiB
```

## Root Causes
1. **Missing memory cleanup** between pipeline stages (Stage 2 + Stage 3 = 26GB > 24GB)
2. **MPS buffer limit** (~10GB max allocation, but models are 13GB each)

## Solutions Implemented

### 1. Sequential Model Loading
```python
def _unload_model(self, model_name: str) -> None:
    """Explicitly unload model and free memory."""
    del self._biomistral_model  # or _sabia_model
    del self._biomistral_tokenizer
    gc.collect()
    torch.mps.empty_cache()
```

### 2. Hybrid Device Mapping
```python
AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",  # Let transformers split intelligently
    max_memory={"mps": "10GiB", "cpu": "16GiB"}
)
```

**Result:** Layers 0-22/23 on MPS (GPU), layers 23/24-31 on CPU (70/30 split)

## Test Results
```
Stage 1: 36.0s   (llava-llama3, Ollama)
Stage 2: 41.7s   (BioMistral, MPS+CPU)
Stage 3: 151.8s  (Sabiá, MPS+CPU)
Total:   232.0s  (~3.9 min) ✅
```

## Quick Test
```bash
# Run full pipeline test
uv run python test_memory_fix.py

# Or quick validation
uv run python -c "from src.core.medical_pipeline import MedicalPipeline; print('✅ Works')"
```

## Key Files
- `src/core/medical_pipeline.py` - Core implementation (490 lines)
- `test_memory_fix.py` - Quick test script (116 lines)
- `DAY3_COMPLETE.md` - Full documentation (278 lines)

## Git Commands
```bash
# View Day 3 commits
git log --oneline -4

# Show changes
git diff main...HEAD --stat

# Push when ready
git push origin refactor/enhance-api-to-support-pipeline
```

## Next Steps (Day 4)
1. Add `/analyze` endpoint to Flask API
2. Input validation (image format, CNN diagnosis)
3. Error handling and JSON responses
4. Integration tests
