#!/usr/bin/env python3
"""Quick test: 2-stage pipeline without translation."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core.medical_pipeline import MedicalPipeline

print("=" * 80)
print("🧪 TESTING 2-STAGE PIPELINE (NO TRANSLATION)")
print("=" * 80)
print()
print("📋 Configuration:")
print("  - Image: imgs/person75_bacteria_365.jpeg (PNEUMONIA)")
print("  - Vision Model: llava-llama3")
print("  - RAG: Enabled")
print("  - Translation: DISABLED (for speed)")
print()
print("=" * 80)
print()

start = time.time()

# Initialize pipeline WITHOUT translation
pipeline = MedicalPipeline(use_rag=True, vision_model='llava-llama3', enable_translation=False)

# Run analysis
result = pipeline.analyze_xray(
    'imgs/person75_bacteria_365.jpeg',
    {'prediction': 'PNEUMONIA', 'confidence': 0.89}
)

elapsed = time.time() - start

if result['success']:
    print("\n" + "=" * 80)
    print("✅ PIPELINE SUCCESS!")
    print("=" * 80)
    print()
    print(f"⏱️  Total Time: {result['total_latency_s']:.1f}s")
    print()
    print("📊 Stage Breakdown:")
    print(f"  Stage 1 (Vision):  {result['stage1_vision']['latency_s']:.1f}s")
    print(f"  Stage 2 (Medical): {result['stage2_medical']['latency_s']:.1f}s")
    if 'stage3_translation' in result:
        print(f"  Stage 3 (Translation): {result['stage3_translation']['latency_s']:.1f}s")
    else:
        print("  Stage 3 (Translation): SKIPPED ⏭️")
    print()
    print("📄 FINAL REPORT (EN):")
    print("-" * 80)
    print(result['final_report_en'])
    print("-" * 80)
    print()
    print(f"💾 Speed Improvement: ~{150 - result['total_latency_s']:.0f}s saved by skipping translation")
    print()
else:
    print(f"\n❌ FAILED: {result.get('error', 'Unknown')}")
    sys.exit(1)

print("=" * 80)
print("🏁 Test Complete")
print("=" * 80)
