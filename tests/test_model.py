"""
Test script for CNN model diagnosis with Grad-CAM visualizations
Uses new functional architecture from src/
"""

from src.core.diagnosis import load_cnn_model, diagnose_from_path
from src.core.visualization import generate_gradcam_visualization

if __name__ == "__main__":
    # Load model once
    print("Loading CNN model...")
    model = load_cnn_model("models/best_model.keras")

    # Test images
    test_images = [
        "imgs/person75_bacteria_365.jpeg",
        "imgs/person72_bacteria_352.jpeg",
        "imgs/person74_bacteria_361.jpeg",
    ]

    for image_path in test_images:
        print(f"\nProcessing: {image_path}")

        # Diagnose
        class_name, confidence = diagnose_from_path(model, image_path)
        print(f"Diagnosis: {class_name} ({confidence:.1%})")

        # Generate Grad-CAM visualizations
        heatmap_path, overlay_path = generate_gradcam_visualization(model, image_path)
        print(f"Heatmap saved: {heatmap_path}")
        print(f"Overlay saved: {overlay_path}")
