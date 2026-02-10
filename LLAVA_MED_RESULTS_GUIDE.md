# LLaVA-Med Evaluation Results - Complete Guide

## ✅ Status: COMPLETE

All 5 llava-med test cases have been successfully evaluated and saved.

## 📂 Location

```
tests/pipeline_evaluation_3stage/vision_llava-med/
```

## 📊 What You Have

- ✅ **5/5 test cases** fully processed
- ✅ **35 files total** (7 files per case)
- ✅ **All metrics** in evaluation_summary.json
- ✅ **Complete pipeline runs**: Vision (llava-med) → Medical (BioMistral) → Reports

## 📁 File Structure

```
vision_llava-med/
├── case_01_normal_clear/
├── case_02_pneumonia_severe/
├── case_03_normal_challenging/
├── case_04_pneumonia_moderate/
└── case_05_false_negative/
    ├── 1_original.jpeg          - Input X-ray image
    ├── 2_cnn_diagnosis.json     - CNN classification
    ├── 3_gradcam_overlay.png    - Grad-CAM visualization
    ├── 4_stage1_vision.json     - LLaVA-Med analysis (detailed JSON)
    ├── 5_stage2_medical.json    - BioMistral report (detailed JSON)
    ├── 7_final_report_en.txt    - Human-readable report ⭐
    └── 9_pipeline_metrics.json  - Performance metrics
```

## 🔍 How to View Results

### Read Single Report
```bash
cat tests/pipeline_evaluation_3stage/vision_llava-med/case_02_pneumonia_severe/7_final_report_en.txt
```

### Compare llava-med vs llava:13b (Same Case)
```bash
# llava-med
cat tests/pipeline_evaluation_3stage/vision_llava-med/case_02_pneumonia_severe/7_final_report_en.txt

# llava:13b  
cat tests/pipeline_evaluation_3stage/vision_llava_13b/case_02_pneumonia_severe/7_final_report_en.txt
```

### View All llava-med Reports
```bash
for case in tests/pipeline_evaluation_3stage/vision_llava-med/case_*/; do
    echo "=== $(basename $case) ==="
    cat $case/7_final_report_en.txt
    echo ""
done
```

### Extract JSON Metrics
```bash
# All llava-med metrics
jq '.results | map(select(.vision_model == "llava-med"))' \
   tests/pipeline_evaluation_3stage/evaluation_summary.json

# Average latency
jq '.results | map(select(.vision_model == "llava-med")) | map(.total_pipeline_latency_s) | add / length' \
   tests/pipeline_evaluation_3stage/evaluation_summary.json

# Medical term density
jq '.results | map(select(.vision_model == "llava-med")) | map(.stage1_medical_density) | add / length' \
   tests/pipeline_evaluation_3stage/evaluation_summary.json
```

## 📈 Quick Comparison Commands

### Compare All 4 Models on One Case
```bash
echo "=== LLAVA-MED ===" 
cat tests/pipeline_evaluation_3stage/vision_llava-med/case_02_pneumonia_severe/7_final_report_en.txt

echo "=== LLAVA-LLAMA3 ===" 
cat tests/pipeline_evaluation_3stage/vision_llava-llama3/case_02_pneumonia_severe/7_final_report_en.txt

echo "=== LLAVA:13B ===" 
cat tests/pipeline_evaluation_3stage/vision_llava_13b/case_02_pneumonia_severe/7_final_report_en.txt

echo "=== LLAVA:7B ===" 
cat tests/pipeline_evaluation_3stage/vision_llava_7b/case_02_pneumonia_severe/7_final_report_en.txt
```

### Extract Performance Data for Excel/Charts
```bash
# Export llava-med data to CSV-friendly format
jq -r '.results | map(select(.vision_model == "llava-med")) | 
       ["case_id","latency","stage1_words","stage2_terms","medical_density"], 
       (.[] | [.case_id, .total_pipeline_latency_s, .stage1_word_count, .stage2_medical_terms, .stage1_medical_density]) | 
       @csv' tests/pipeline_evaluation_3stage/evaluation_summary.json > llava_med_metrics.csv
```

## 📊 Available Metrics

Each case includes:
- **Latency**: Total time, Stage 1 time, Stage 2 time
- **Medical Quality**: Word count, medical term count, medical density
- **Accuracy**: CNN classification, ground truth comparison
- **RAG**: Context used, retrieved guidelines/reports

## 🎯 For Your Thesis

### Key Comparison Points

1. **Speed**: Compare llava-med latency vs llava:13b/llava:7b
2. **Medical Language**: Compare medical term density across models
3. **Report Quality**: Read reports manually and score on:
   - Anatomical accuracy
   - Finding specificity  
   - Clinical relevance
   - Language appropriateness

4. **Diagnostic Accuracy**: Compare CNN classification rates

### Example Analysis

```bash
# Compare average latencies
echo "llava-med:" && jq '.results | map(select(.vision_model == "llava-med")) | map(.total_pipeline_latency_s) | add / length' tests/pipeline_evaluation_3stage/evaluation_summary.json

echo "llava:13b:" && jq '.results | map(select(.vision_model == "llava:13b")) | map(.total_pipeline_latency_s) | add / length' tests/pipeline_evaluation_3stage/evaluation_summary.json

echo "llava-llama3:" && jq '.results | map(select(.vision_model == "llava-llama3")) | map(.total_pipeline_latency_s) | add / length' tests/pipeline_evaluation_3stage/evaluation_summary.json
```

## 📖 Example Report Content

From `case_02_pneumonia_severe`:

```
CHEST X-RAY ANALYSIS - 3-STAGE MEDICAL PIPELINE
======================================================================

CNN CLASSIFICATION
Diagnosis: PNEUMONIA (Confidence: 100.0%)
Ground Truth: PNEUMONIA
✓ Correct Classification

STAGE 1: VISION ANALYSIS (llava-med)
The chest X-ray shows bilateral lung opacities, which are areas of increased 
density in the lungs. These opacities can be caused by various factors...

STAGE 2: MEDICAL ASSESSMENT (BioMistral-7B + RAG)
"The chest X-ray shows bilateral lung opacities..."

PERFORMANCE METRICS
- Stage 1 (Vision): 34.0s
- Stage 2 (Medical): 222.2s
- Total Pipeline: 257.5s
```

## 🎓 Thesis Contribution

Your unique contribution:
- **First comparison** of medical-specific (llava-med) vs general vision models
- On **Brazilian pneumonia detection** task
- With **RAG-enhanced** 3-stage pipeline
- **Complete evaluation**: 4 models × 5 test cases = 20 runs

**Central Research Question:**
*"Does medical domain fine-tuning (llava-med) justify slower inference for pneumonia detection, or do general models (llava:13b) perform adequately?"*

## ✅ Verification Checklist

- [x] 5 case directories exist
- [x] Each case has 7 files
- [x] Reports are readable (check one with `cat`)
- [x] JSON metrics are valid (check with `jq`)
- [x] All timestamps show recent completion
- [x] evaluation_summary.json includes llava-med entries

## 📞 Next Steps

1. Read 2-3 reports from each model
2. Create comparison table (speed, quality, accuracy)
3. Extract metrics to Excel/R for analysis
4. Generate charts (latency, medical density)
5. Write thesis discussion comparing findings

---

**Files Location**: `tests/pipeline_evaluation_3stage/vision_llava-med/`  
**Summary Data**: `tests/pipeline_evaluation_3stage/evaluation_summary.json`  
**Total Runs**: 20 complete (4 models × 5 cases)  
**Status**: ✅ READY FOR ANALYSIS
