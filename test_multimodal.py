"""
Test script for multimodal pneumonia detection (CNN + Grad-CAM + LLaVA LLM)
"""

from service.pneumonia_service import PneumoniaDetectorService

if __name__ == "__main__":
    print("=" * 80)
    print("MULTIMODAL PNEUMONIA DETECTION TEST")
    print("Testing: CNN (ResNet50) + Grad-CAM + LLaVA LLM")
    print("=" * 80)

    # Initialize detector with LLM enabled
    print("\n1. Initializing PneumoniaDetectorService with LLM...")
    detector = PneumoniaDetectorService("models/pneumonia_model.keras", enable_llm=True)

    # Test images
    test_images = [
        "imgs/person75_bacteria_365.jpeg",
        "imgs/1_normal1.jpeg",
    ]

    for idx, image_path in enumerate(test_images, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {idx}: {image_path}")
        print("=" * 80)

        try:
            (
                class_name,
                confidence,
                llm_description,
                overlay_path,
                heatmap_path,
            ) = detector.diagnose_with_llm_explanation(image_path)

            print(f"\n📊 DIAGNOSIS RESULT:")
            print(f"   Class: {class_name}")
            print(f"   Confidence: {confidence:.1%}")

            print(f"\n🖼️  VISUALIZATION FILES:")
            print(f"   Heatmap: {heatmap_path}")
            print(f"   Overlay: {overlay_path}")

            print(f"\n🤖 LLM CLINICAL DESCRIPTION:")
            print("-" * 80)
            print(llm_description)
            print("-" * 80)

        except Exception as e:
            print(f"\n❌ Error processing {image_path}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print("=" * 80)
