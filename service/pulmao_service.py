import os

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class LungDetector:
    def __init__(self, model_path, img_size=(224, 224)):
        """
        Initialize the lung detector by loading the trained model.

        Parameters:
        - model_path: Path to the saved model file (.keras).
        - img_size: Expected image size (default = (224, 224)).
        """
        self.model = load_model(model_path)
        self.img_size = img_size

    def _preprocess_image(self, image_path):
        """
        Preprocess the image to the format expected by the neural network.

        Parameters:
        - image_path: Path to the image to be processed.

        Returns:
        - Image prepared for prediction (4D tensor).
        """
        img = image.load_img(image_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize pixels
        return np.expand_dims(img_array, axis=0)  # Add batch dimension

    def detect_image(self, image_path):
        """
        Detect whether the image is a lung or not.

        Parameters:
        - image_path: Path to the image to be analyzed.

        Returns:
        - A tuple (predicted_class, confidence) indicating the result.
        """
        processed_image = self._preprocess_image(image_path)
        pred = self.model.predict(processed_image)[0][0]
        class_name = "LUNG" if pred > 0.5 else "NOT A LUNG"
        confidence = pred if pred > 0.5 else 1 - pred
        print(f"{os.path.basename(image_path)}: {class_name} (confidence: {confidence:.2f})")
        return class_name, confidence

    def detect_folder(self, images_folder):
        """
        Detect all images within a folder, indicating whether they are lungs or not.

        Parameters:
        - images_folder: Path to the folder with images to be analyzed.
        """
        for filename in os.listdir(images_folder):
            path = os.path.join(images_folder, filename)
            if os.path.isfile(path):
                processed_image = self._preprocess_image(path)
                pred = self.model.predict(processed_image)[0][0]
                class_name = "LUNG" if pred > 0.5 else "NOT A LUNG"
                confidence = pred if pred > 0.5 else 1 - pred
                print(f"{filename}: {class_name} (confidence: {confidence:.2f})")
