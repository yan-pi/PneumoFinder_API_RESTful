"""Core diagnosis functions for pneumonia detection."""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from src.utils.image_utils import preprocess_image


def load_cnn_model(model_path: str):
    """
    Load CNN model from path and initialize it.

    Args:
        model_path: Path to .keras model file

    Returns:
        Loaded Keras model
    """
    import tensorflow as tf

    model = load_model(model_path)

    # Force initialization with dummy prediction
    dummy = tf.zeros((1, 224, 224, 3))
    _ = model.predict(dummy, verbose=0)

    return model


def predict_pneumonia(model, image_array: np.ndarray) -> tuple[str, float]:
    """
    Run CNN inference and classify pneumonia.

    Args:
        model: Loaded Keras model
        image_array: Preprocessed image array

    Returns:
        Tuple of (diagnosis, confidence) where diagnosis is 'PNEUMONIA' or 'NORMAL'
    """
    prediction = float(model.predict(image_array, verbose=0)[0][0])
    diagnosis = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return diagnosis, confidence


def diagnose_from_path(model, image_path: str) -> tuple[str, float]:
    """
    Complete diagnosis pipeline from image path.

    Args:
        model: Loaded Keras model
        image_path: Path to X-ray image

    Returns:
        Tuple of (diagnosis, confidence)
    """
    image_array = preprocess_image(image_path)
    return predict_pneumonia(model, image_array)


def diagnose_with_visualization(
    model, base_model, last_conv_layer, image_path: str, output_dir: str = "temp"
) -> tuple[str, float, str, str]:
    """
    Diagnose pneumonia with Grad-CAM visualizations.

    Args:
        model: Full CNN model
        base_model: ResNet base model
        last_conv_layer: Last convolutional layer
        image_path: Path to X-ray image
        output_dir: Directory to save visualizations

    Returns:
        Tuple of (diagnosis, confidence, heatmap_path, overlay_path)
    """
    from PIL import Image

    from src.core.visualization import generate_complete_visualization
    from src.utils.file_utils import generate_output_paths
    from src.utils.image_utils import image_to_array, load_image

    # Load and preprocess image
    pil_image = load_image(image_path)
    image_array = preprocess_image(image_path)

    # Predict
    diagnosis, confidence = predict_pneumonia(model, image_array)

    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Generate visualizations
    heatmap, overlay = generate_complete_visualization(
        base_model, last_conv_layer, image_array, image_bgr
    )

    # Save visualizations
    output_paths = generate_output_paths(image_path, output_dir, ["_heatmap", "_overlay"])
    heatmap_path = output_paths["_heatmap"]
    overlay_path = output_paths["_overlay"]

    cv2.imwrite(heatmap_path, heatmap)
    cv2.imwrite(overlay_path, overlay)

    return diagnosis, confidence, heatmap_path, overlay_path
