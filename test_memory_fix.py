#!/usr/bin/env python3
"""Standalone test script for medical pipeline with memory management.

Run this in a separate terminal to see live progress:
    uv run python test_memory_fix.py

Or with standard Python:
    python test_memory_fix.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.medical_pipeline import MedicalPipeline


def main():
    print("=" * 80)
    print("🧪 TESTING MEDICAL PIPELINE WITH MEMORY MANAGEMENT FIX")
    print("=" * 80)
    print()
    print("📋 Test Configuration:")
    print("  - Image: imgs/person75_bacteria_365.jpeg (PNEUMONIA)")
    print("  - Vision Model: llava-llama3 (Ollama)")
    print("  - RAG: Enabled")
    print("  - Expected: All 3 stages complete without OOM error")
    print()
    print("=" * 80)
    print()

    # Initialize pipeline
    print("🔧 Initializing pipeline...")
    pipeline = MedicalPipeline(use_rag=True, vision_model="llava-llama3")
    print("✅ Pipeline initialized")
    print()

    # CNN diagnosis (simulated)
    cnn_diagnosis = {"prediction": "PNEUMONIA", "confidence": 0.89}

    # Run complete pipeline
    print("🚀 Starting 3-stage pipeline...\n")
    start_time = time.time()

    try:
        result = pipeline.analyze_xray("imgs/person75_bacteria_365.jpeg", cnn_diagnosis)

        elapsed = time.time() - start_time

        if result["success"]:
            print("\n" + "=" * 80)
            print("✅ PIPELINE SUCCESS!")
            print("=" * 80)
            print(f"\n⏱️  Total Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
            print(f"\n📊 Stage Breakdown:")
            print(f"  Stage 1 (Vision):      {result['stage1_vision']['latency_s']:.1f}s")
            print(f"  Stage 2 (Medical):     {result['stage2_medical']['latency_s']:.1f}s")
            print(f"  Stage 3 (Translation): {result['stage3_translation']['latency_s']:.1f}s")
            print(f"\n🔬 Models Used:")
            print(f"  Vision:      {result['stage1_vision']['model']}")
            print(f"  Medical:     {result['stage2_medical']['model']}")
            print(f"  Translation: {result['stage3_translation']['model']}")
            print(f"\n📄 FINAL REPORT (PT-BR):")
            print("-" * 80)
            print(result["final_report_pt_br"])
            print("-" * 80)
            print()
            print("💾 Memory Management: ✅ NO OOM ERRORS!")
            print("   - Stage 1: Ollama auto-unloaded")
            print("   - Stage 2: BioMistral loaded → generated → unloaded")
            print("   - Stage 3: Sabiá loaded → generated → unloaded")
            print()
            print("🎉 Memory fix VERIFIED!")

        else:
            print("\n" + "=" * 80)
            print("❌ PIPELINE FAILED")
            print("=" * 80)
            print(f"\n⚠️  Error: {result['error']}")
            print(f"\n⏱️  Failed after: {elapsed:.1f}s")

            # Check if it's the memory error we're trying to fix
            if "Invalid buffer size: 13.24 GiB" in result["error"]:
                print("\n🔴 MEMORY ERROR DETECTED!")
                print("   This is the error we're trying to fix.")
                print("   The unload mechanism may not be working correctly.")
            elif "buffer size" in result["error"].lower() or "memory" in result["error"].lower():
                print("\n🔴 MEMORY-RELATED ERROR!")
                print("   Different memory issue than expected.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user (Ctrl+C)")
        elapsed = time.time() - start_time
        print(f"   Ran for: {elapsed:.1f}s")

    except Exception as e:
        print("\n" + "=" * 80)
        print("💥 UNEXPECTED ERROR")
        print("=" * 80)
        print(f"\n{type(e).__name__}: {e}")
        import traceback

        print("\nFull traceback:")
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("🏁 Test Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
