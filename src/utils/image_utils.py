"""Image processing utilities - pure functions for image transformations."""

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image


def load_image(path: str, target_size: tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Load image from path with target size.

    Args:
        path: Path to image file
        target_size: Target size as (width, height)

    Returns:
        PIL Image object
    """
    return image.load_img(path, target_size=target_size)


def image_to_array(img: Image.Image) -> np.ndarray:
    """
    Convert PIL image to numpy array.

    Args:
        img: PIL Image object

    Returns:
        Numpy array with shape (height, width, channels)
    """
    return image.img_to_array(img)


def add_batch_dimension(arr: np.ndarray) -> np.ndarray:
    """
    Add batch dimension to array.

    Args:
        arr: Array with shape (height, width, channels)

    Returns:
        Array with shape (1, height, width, channels)
    """
    return np.expand_dims(arr, axis=0)


def preprocess_for_resnet(arr: np.ndarray) -> np.ndarray:
    """
    Apply ResNet50 preprocessing to image array.

    Args:
        arr: Image array

    Returns:
        Preprocessed array ready for ResNet50
    """
    return tf.keras.applications.resnet50.preprocess_input(arr.copy())


def normalize_pixels(arr: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to 0-1 range.

    Args:
        arr: Image array with pixel values 0-255

    Returns:
        Normalized array with values 0-1
    """
    return arr / 255.0


def preprocess_image(path: str, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Complete preprocessing pipeline for CNN inference.

    Loads image, converts to array, adds batch dimension, and applies ResNet preprocessing.

    Args:
        path: Path to image file
        target_size: Target size as (width, height)

    Returns:
        Preprocessed array ready for model prediction
    """
    img = load_image(path, target_size)
    arr = image_to_array(img)
    arr = add_batch_dimension(arr)
    arr = preprocess_for_resnet(arr)
    return arr


def preprocess_image_normalized(path: str, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocessing pipeline with simple normalization (0-1 range).

    Used for models that don't need ResNet-specific preprocessing.

    Args:
        path: Path to image file
        target_size: Target size as (width, height)

    Returns:
        Normalized array ready for model prediction
    """
    img = load_image(path, target_size)
    arr = image_to_array(img)
    arr = normalize_pixels(arr)
    arr = add_batch_dimension(arr)
    return arr
