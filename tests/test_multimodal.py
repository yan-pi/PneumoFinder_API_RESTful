"""
Test script for multimodal pneumonia detection (CNN + Grad-CAM + LLaVA LLM)
Uses new functional architecture from src/
"""

from src.core.diagnosis import load_cnn_model, diagnose_from_path
from src.core.visualization import generate_gradcam_visualization
from src.core.clinical_description import generate_clinical_description

if __name__ == "__main__":
    print("=" * 80)
    print("MULTIMODAL PNEUMONIA DETECTION TEST")
    print("Testing: CNN (ResNet50) + Grad-CAM + LLaVA LLM")
    print("=" * 80)

    # Load model once
    print("\n1. Loading CNN model...")
    model = load_cnn_model("models/pneumonia_model.keras")

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
            # 1. CNN Diagnosis
            class_name, confidence = diagnose_from_path(model, image_path)

            print(f"\n📊 DIAGNOSIS RESULT:")
            print(f"   Class: {class_name}")
            print(f"   Confidence: {confidence:.1%}")

            # 2. Generate Grad-CAM visualizations
            heatmap_path, overlay_path = generate_gradcam_visualization(model, image_path)

            print(f"\n🖼️  VISUALIZATION FILES:")
            print(f"   Heatmap: {heatmap_path}")
            print(f"   Overlay: {overlay_path}")

            # 3. Generate LLM clinical description
            llm_description = generate_clinical_description(
                image_path, class_name, confidence, heatmap_path, overlay_path
            )

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
