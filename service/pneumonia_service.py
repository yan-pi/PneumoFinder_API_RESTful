# service/pneumonia_service.py
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image

from service.llm_service import LLMService


class PneumoniaDetectorService:
    def __init__(self, model_path, img_size=(224, 224), enable_llm=True):
        self.img_size = img_size
        self.model = load_model(model_path)
        self.enable_llm = enable_llm

        # Initialize LLM service if enabled
        self.llm_service = None
        if enable_llm:
            try:
                self.llm_service = LLMService()
                if self.llm_service.is_available():
                    print("LLM service initialized successfully (llava:7b)")
                else:
                    print(
                        "Warning: LLM service unavailable. Will use fallback descriptions."
                    )
            except Exception as e:
                print(f"Warning: Could not initialize LLM service: {e}")
                self.llm_service = None

        # FORCE INITIALIZATION
        dummy = tf.zeros((1, *img_size, 3))
        _ = self.model.predict(dummy, verbose=0)

        # === FIND RESNET BASE ===
        self.base_model = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model) and "resnet" in layer.name.lower():
                self.base_model = layer
                break
        if not self.base_model:
            raise ValueError("ResNet50 not found!")

        # === LAST CONV ===
        self.last_conv = None
        for layer in reversed(self.base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                self.last_conv = layer
                break
        if not self.last_conv:
            raise ValueError("Conv2D not found!")

        print(f"Grad-CAM target: {self.last_conv.name}")

    def _preprocess(self, image_path):
        img = image.load_img(image_path, target_size=self.img_size)
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = tf.keras.applications.resnet50.preprocess_input(arr.copy())
        return img, arr

    def _gradcam_smooth(self, img_array, n_samples=30, noise_level=0.2):
        grad_model = Model(
            inputs=self.base_model.input, outputs=[self.last_conv.output, self.base_model.output]
        )

        cam_accum = None
        for _ in range(n_samples):
            noisy_input = img_array.copy()
            if n_samples > 1:
                noise = np.random.normal(0, noise_level * 25, img_array.shape)
                noisy_input = img_array + noise.astype(np.float32)

            with tf.GradientTape() as tape:
                conv_out, feature_maps = grad_model(noisy_input, training=False)
                pred = tf.reduce_mean(feature_maps, axis=(1, 2))
                class_score = tf.maximum(pred, 1 - pred)

            grads = tape.gradient(class_score, conv_out)
            if grads is None:
                continue

            weights = tf.reduce_mean(grads, axis=(1, 2))
            cam = tf.reduce_sum(weights * conv_out[0], axis=-1)

            if cam_accum is None:
                cam_accum = cam
            else:
                cam_accum += cam

        cam = cam_accum / n_samples
        cam = np.maximum(cam, 0)

        # Highlight only the most relevant central point
        cam = cam**2
        cam /= cam.max() + 1e-8

        h, w = self.img_size
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        cam = cv2.GaussianBlur(cam, (3, 3), sigmaX=1.5)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def _apply_heatmap_red(self, cam):
        cam_uint8 = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_HOT)
        heatmap = heatmap.astype(np.float32)
        heatmap[:, :, 2] = np.clip(heatmap[:, :, 2] * cam, 0, 255)  # red central
        heatmap[:, :, 0] *= 0.2
        heatmap[:, :, 1] *= 0.4
        return heatmap.astype(np.uint8)

    def _overlay(self, img_bgr, heatmap, alpha=0.68):
        heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]), cv2.INTER_CUBIC)
        return cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)

    def diagnose_image(self, image_path):
        """Simple diagnosis without visualization - returns (class_name, confidence)"""
        _, arr = self._preprocess(image_path)
        pred = float(self.model.predict(arr, verbose=0)[0][0])
        class_name = "PNEUMONIA" if pred > 0.5 else "NORMAL"
        confidence = pred if pred > 0.5 else 1 - pred
        return class_name, np.array(confidence)

    def diagnose_with_explanation(self, image_path, output_folder="relatorios"):
        name = os.path.splitext(os.path.basename(image_path))[0]
        folder = os.path.join(output_folder, name)
        os.makedirs(folder, exist_ok=True)

        img_pil, arr = self._preprocess(image_path)
        pred = float(self.model.predict(arr, verbose=0)[0][0])
        class_name = "PNEUMONIA" if pred > 0.5 else "NORMAL"
        confidence = pred if pred > 0.5 else 1 - pred

        print(f"{name}: {class_name} ({confidence:.1%})")

        cam = self._gradcam_smooth(arr)
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        heatmap = self._apply_heatmap_red(cam)
        overlay = self._overlay(img_bgr, heatmap, alpha=0.7)

        def save_fig(fig, path):
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1)
            plt.close(fig)

        # 01 Original
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_pil)
        ax.set_title(
            f"Original\n{class_name} ({confidence:.1%})", fontsize=14, pad=15, weight="bold"
        )
        ax.axis("off")
        save_fig(fig, f"{folder}/01_original.png")

        # 02 Grad-CAM
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cam, cmap="hot", vmin=0, vmax=1)
        ax.set_title("Grad-CAM (Pulmonary Focus)", fontsize=14, pad=15, weight="bold")
        ax.axis("off")
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("Activation", rotation=270, labelpad=15)
        save_fig(fig, f"{folder}/02_gradcam.png")

        # 03 Overlay
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title("Pneumonia Regions", fontsize=14, pad=15, weight="bold")
        ax.axis("off")
        save_fig(fig, f"{folder}/03_overlay.png")

        # 04 Confidence
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(
            ["NORMAL", "PNEUMONIA"],
            [1 - pred, pred],
            color=["#2ca02c", "#d62728"],
            edgecolor="black",
        )
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 0.03,
                f"{h:.1%}",
                ha="center",
                fontweight="bold",
                fontsize=12,
            )
        ax.set_ylim(0, 1)
        ax.set_title("Confidence", fontsize=14, weight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        save_fig(fig, f"{folder}/04_confidence.png")

        print(f"Report saved: {folder}/")
        return class_name, confidence, folder

    def diagnose_with_llm_explanation(self, image_path, temp_folder="temp"):
        """
        Diagnose pneumonia with LLM-generated clinical description.
        Returns: (class_name, confidence, llm_description, overlay_path, heatmap_path)
        """
        # Create temp folder if not exists
        os.makedirs(temp_folder, exist_ok=True)

        # Preprocess and predict
        img_pil, arr = self._preprocess(image_path)
        pred = float(self.model.predict(arr, verbose=0)[0][0])
        class_name = "PNEUMONIA" if pred > 0.5 else "NORMAL"
        confidence = pred if pred > 0.5 else 1 - pred

        print(f"Diagnosis: {class_name} ({confidence:.1%})")

        # Generate Grad-CAM
        cam = self._gradcam_smooth(arr)
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        heatmap = self._apply_heatmap_red(cam)
        overlay = self._overlay(img_bgr, heatmap, alpha=0.7)

        # Save heatmap and overlay to temp folder
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        heatmap_path = os.path.join(temp_folder, f"{base_name}_heatmap.png")
        overlay_path = os.path.join(temp_folder, f"{base_name}_overlay.png")

        cv2.imwrite(heatmap_path, heatmap)
        cv2.imwrite(overlay_path, overlay)

        # Generate LLM description
        llm_description = None
        if self.llm_service:
            try:
                print("Generating LLM clinical description...")
                llm_description = self.llm_service.generate_clinical_description(
                    class_name=class_name,
                    confidence=confidence,
                    original_image_path=image_path,
                    overlay_image_path=overlay_path,
                )
                print("LLM description generated successfully")
            except Exception as e:
                print(f"Error generating LLM description: {e}")
                llm_description = None

        # Fallback if LLM fails or disabled
        if not llm_description:
            if class_name == "PNEUMONIA":
                llm_description = (
                    f"The CNN model detected signs of pneumonia with {confidence*100:.1f}% confidence. "
                    f"The Grad-CAM visualization highlights regions of the lung that influenced this diagnosis. "
                    f"Further clinical evaluation and additional imaging may be warranted.\n\n"
                    f"⚠️ Disclaimer: This AI analysis is for educational purposes only. "
                    f"Always consult qualified healthcare professionals for medical decisions."
                )
            else:
                llm_description = (
                    f"The CNN model indicates normal lung appearance with {confidence*100:.1f}% confidence. "
                    f"No significant abnormalities were detected in the analyzed regions.\n\n"
                    f"⚠️ Disclaimer: This AI analysis is for educational purposes only. "
                    f"Always consult qualified healthcare professionals for medical decisions."
                )

        return class_name, confidence, llm_description, overlay_path, heatmap_path
