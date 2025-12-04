"""Grad-CAM visualization functions for explainable AI."""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def find_last_conv_layer(model):
    """
    Find the last convolutional layer in the model.

    Args:
        model: Keras model

    Returns:
        Last Conv2D layer found in the model
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No convolutional layer found in model")


def find_resnet_base(model):
    """
    Find ResNet base model within the full model.

    Args:
        model: Keras model

    Returns:
        ResNet base model
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "resnet" in layer.name.lower():
            return layer
    raise ValueError("ResNet base model not found")


def generate_gradcam(
    base_model,
    last_conv_layer,
    image_array: np.ndarray,
    n_samples: int = 30,
    noise_level: float = 0.2,
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap using gradient-weighted class activation mapping.

    Uses smooth Grad-CAM with multiple noisy samples for more robust visualization.

    Args:
        base_model: Base model (e.g., ResNet50)
        last_conv_layer: Last convolutional layer for gradients
        image_array: Preprocessed image array (batch of 1)
        n_samples: Number of noisy samples for smoothing
        noise_level: Amount of noise to add for smoothing

    Returns:
        CAM heatmap as numpy array (2D, values 0-1)
    """
    # Create gradient model
    grad_model = Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output],
    )

    cam_accumulator = None

    # Generate multiple samples with noise for smoothing
    for _ in range(n_samples):
        noisy_input = image_array.copy()

        if n_samples > 1:
            noise = np.random.normal(0, noise_level * 25, image_array.shape)
            noisy_input = image_array + noise.astype(np.float32)

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, feature_maps = grad_model(noisy_input, training=False)
            pred = tf.reduce_mean(feature_maps, axis=(1, 2))
            class_score = tf.maximum(pred, 1 - pred)

        grads = tape.gradient(class_score, conv_outputs)
        if grads is None:
            continue

        # Weight the convolutional outputs by gradients
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(weights * conv_outputs[0], axis=-1)

        # Accumulate CAMs
        if cam_accumulator is None:
            cam_accumulator = cam
        else:
            cam_accumulator += cam

    # Average accumulated CAMs
    cam = cam_accumulator / n_samples
    cam = np.maximum(cam, 0)

    # Highlight most relevant regions
    cam = cam**2
    cam = cam / (cam.max() + 1e-8)

    return cam


def resize_and_normalize_cam(cam: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize CAM to target size and normalize to 0-1 range.

    Args:
        cam: CAM heatmap
        target_size: Target size as (width, height)

    Returns:
        Resized and normalized CAM
    """
    width, height = target_size
    cam_resized = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)
    cam_blurred = cv2.GaussianBlur(cam_resized, (3, 3), sigmaX=1.5)

    # Normalize to 0-1
    cam_normalized = (cam_blurred - cam_blurred.min()) / (
        cam_blurred.max() - cam_blurred.min() + 1e-8
    )

    return cam_normalized


def apply_red_heatmap(cam: np.ndarray) -> np.ndarray:
    """
    Apply red-focused heatmap colormap to CAM.

    Args:
        cam: Normalized CAM (values 0-1)

    Returns:
        RGB heatmap image
    """
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_HOT).astype(np.float32)

    # Enhance red channel, reduce blue/green
    red_channel = np.clip(heatmap[:, :, 2] * cam, 0, 255)

    heatmap_adjusted = np.stack(
        [
            heatmap[:, :, 0] * 0.2,  # Blue (reduced)
            heatmap[:, :, 1] * 0.4,  # Green (reduced)
            red_channel,  # Red (enhanced)
        ],
        axis=-1,
    ).astype(np.uint8)

    return heatmap_adjusted


def overlay_heatmap_on_image(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.7
) -> np.ndarray:
    """
    Overlay heatmap on original image.

    Args:
        image: Original image (BGR format)
        heatmap: Heatmap to overlay
        alpha: Transparency of heatmap (0-1, higher = more heatmap visible)

    Returns:
        Overlaid image
    """
    heatmap_resized = cv2.resize(
        heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC
    )
    return cv2.addWeighted(image, 1 - alpha, heatmap_resized, alpha, 0)


def generate_complete_visualization(
    base_model,
    last_conv_layer,
    image_array: np.ndarray,
    original_image: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate complete Grad-CAM visualization (heatmap + overlay).

    Args:
        base_model: Base model for gradients
        last_conv_layer: Last conv layer
        image_array: Preprocessed image array
        original_image: Original image (BGR)
        target_size: Target size for visualization

    Returns:
        Tuple of (heatmap, overlay) as numpy arrays
    """
    # Generate CAM
    cam = generate_gradcam(base_model, last_conv_layer, image_array)

    # Resize and normalize
    cam_normalized = resize_and_normalize_cam(cam, target_size)

    # Create heatmap
    heatmap = apply_red_heatmap(cam_normalized)

    # Create overlay
    overlay = overlay_heatmap_on_image(original_image, heatmap, alpha=0.7)

    return heatmap, overlay
