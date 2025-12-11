# Day 3 Plan: LLM Pipeline Integration

**Goal:** Orchestrate the 3-stage LLM pipeline with RAG context injection

**Status:** READY TO START

---

## Prerequisites (Complete ✅)

- ✅ 4 models installed (LLaVA-Med, BioMistral, Sabiá, llava-llama3)
- ✅ RAG infrastructure working
- ✅ Prompt templates ready
- ✅ Medical knowledge base populated (159 docs)

---

## Tasks

### 1. Create Medical Pipeline Orchestrator
**File:** `src/core/medical_pipeline.py`

**Features:**
- Load all 4 LLMs on demand (lazy loading)
- Orchestrate 3-stage flow:
  1. Vision → English findings
  2. Medical text → Professional English report
  3. Translation → PT-BR report
- Inject RAG context at each stage
- Handle errors & fallbacks

**Estimated:** 2-3 hours

### 2. Integration Testing
**File:** `scripts/test_pipeline.py`

**Tests:**
- Normal X-ray (should NOT hallucinate)
- Pneumonia X-ray (should detect correctly)
- Edge case (subtle findings)

**Validation:**
- ✅ No hallucinations on normal X-rays
- ✅ Correct left/right orientation
- ✅ Natural PT-BR output (not literal translation)
- ✅ 2-3 sentences (not verbose)

**Estimated:** 1-2 hours

### 3. Benchmark on Evaluation Dataset
**Dataset:** `tests/medical_evaluation/` (5 cases × 3 models)

**Metrics:**
- Hallucination rate (target: 0%)
- Anatomical accuracy (target: 100%)
- PT-BR quality (manual review)
- Processing time per image

**Estimated:** 2 hours

---

## Architecture

```
User uploads X-ray image
         ↓
┌─────────────────────────────────────────┐
│   CNN Classification (existing)         │
│   → prediction: PNEUMONIA/NORMAL        │
│   → confidence: 0.89                    │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│   RAG Retriever                         │
│   → Fetch relevant guidelines           │
│   → Build context for prompts           │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│   STAGE 1: Vision Analysis              │
│   Model: LLaVA-Med (ensemble optional)  │
│   Prompt: RAG-enhanced vision prompt    │
│   Output: English findings (2-3 sent.)  │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│   STAGE 2: Medical Text Generation      │
│   Model: BioMistral-7B                  │
│   Prompt: RAG context + vision + CNN    │
│   Output: Professional English report   │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│   STAGE 3: PT-BR Translation            │
│   Model: Sabiá-7B                       │
│   Prompt: Medical terminology + report  │
│   Output: Natural PT-BR report          │
└─────────────────────────────────────────┘
         ↓
    Final PT-BR Report
```

---

## Example Usage

```python
from src.core.medical_pipeline import MedicalPipeline

# Initialize pipeline (loads models)
pipeline = MedicalPipeline(use_rag=True)

# Process X-ray
result = pipeline.analyze_xray(
    image_path="imgs/person75_bacteria_365.jpeg",
    cnn_prediction={"prediction": "PNEUMONIA", "confidence": 0.89}
)

print(result["pt_br_report"])
# Output: "Consolidação em lobo inferior direito com broncogramas 
#          aéreos visíveis. Achados compatíveis com pneumonia lobar."
```

---

## Success Metrics

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Hallucination rate (normal X-rays) | 100% | 0% | Manual review of 5 cases |
| Left/right accuracy | ~50% | 100% | Check anatomical terms |
| PT-BR quality | Poor (literal) | Natural | Native speaker review |
| Verbosity | 7-9 lines | 2-3 lines | Count sentences |
| Processing time | N/A | <60s | Timer |

---

## Files to Create

1. `src/core/medical_pipeline.py` - Main orchestrator
2. `scripts/test_pipeline.py` - Integration tests
3. `scripts/benchmark_pipeline.py` - Full evaluation

---

## Next Steps After Day 3

**Day 4:** API integration (`src/api/app.py`)
**Day 5:** API testing & optimization
**Day 6-7:** WhatsApp bot integration

---

## Quick Commands

```bash
# Test pipeline on single image
uv run python scripts/test_pipeline.py --image imgs/person75_bacteria_365.jpeg

# Benchmark on evaluation dataset
uv run python scripts/benchmark_pipeline.py --output results/

# Check if models are ready
ls ~/.cache/huggingface/hub/ | grep -E "llava-med|biomistral|sabia"
```

---

**Ready to proceed?** All dependencies are installed and tested. Day 3 should take ~6-8 hours total.
