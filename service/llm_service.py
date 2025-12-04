# service/llm_service.py
import base64
import os
from io import BytesIO

import requests
from PIL import Image


class LLMService:
    """
    Service for generating clinical descriptions using LLaVA multimodal LLM via Ollama.
    Integrates CNN diagnosis + Grad-CAM visualizations with language model reasoning.
    """

    def __init__(
        self,
        model_name="llava:7b",
        ollama_host="http://localhost:11434",
        max_tokens=500,
        temperature=0.3,
    ):
        """
        Initialize LLM service with Ollama configuration.

        Parameters:
        - model_name: Name of the Ollama model to use (default: llava:7b)
        - ollama_host: Ollama API endpoint
        - max_tokens: Maximum tokens for LLM generation
        - temperature: Sampling temperature (lower = more focused)
        """
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_url = f"{ollama_host}/api/generate"

        # Load prompts
        self.medical_prompt_template = self._load_prompt_template()

    def _load_prompt_template(self):
        """Load medical analysis prompt template"""
        prompt_path = os.path.join("prompts", "medical_analysis.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path) as f:
                return f.read()
        else:
            # Default prompt if file doesn't exist
            return """You are a medical AI assistant analyzing chest X-ray images for pneumonia detection.

Context:
- CNN Diagnosis: {class_name}
- Model Confidence: {confidence}%
- Visual Analysis: The heatmap overlay shows regions where the AI model focused its attention during diagnosis

Task:
Provide a clinical description of the findings in 2-3 sentences:
1. Describe what the highlighted regions reveal about potential lung abnormalities
2. Interpret the diagnosis using appropriate medical terminology
3. Mention the confidence level and any clinical considerations

Important: Base your response on the visual evidence shown. Use professional medical language suitable for healthcare providers."""

    def _encode_image_to_base64(self, image_path):
        """Convert image file to base64 string for API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _encode_pil_to_base64(self, pil_image):
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate_clinical_description(
        self, class_name, confidence, original_image_path, overlay_image_path=None
    ):
        """
        Generate clinical description using multimodal LLM.

        Parameters:
        - class_name: CNN diagnosis result (e.g., "PNEUMONIA" or "NORMAL")
        - confidence: Model confidence score (0-1)
        - original_image_path: Path to original X-ray image
        - overlay_image_path: Optional path to Grad-CAM overlay image

        Returns:
        - description: Clinical description text
        """
        try:
            # Format prompt with CNN results
            prompt = self.medical_prompt_template.format(
                class_name=class_name, confidence=f"{confidence*100:.1f}"
            )

            # Use overlay image if available, otherwise original
            image_path = overlay_image_path if overlay_image_path else original_image_path
            image_b64 = self._encode_image_to_base64(image_path)

            # Call Ollama API
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }

            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            description = result.get("response", "").strip()

            # Add medical disclaimer
            disclaimer = (
                "\n\n⚠️ Disclaimer: This AI analysis is for educational purposes only. "
                "Always consult qualified healthcare professionals for medical decisions."
            )

            # Add confidence note if low
            if confidence < 0.7:
                confidence_note = (
                    "\n\nNote: Model confidence is moderate. Manual review recommended."
                )
                description += confidence_note

            return description + disclaimer

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return self._generate_fallback_description(class_name, confidence)
        except Exception as e:
            print(f"Error generating LLM description: {e}")
            return self._generate_fallback_description(class_name, confidence)

    def _generate_fallback_description(self, class_name, confidence):
        """Generate simple description when LLM is unavailable"""
        if class_name == "PNEUMONIA":
            description = (
                f"The CNN model detected signs of pneumonia with {confidence*100:.1f}% confidence. "
                f"The Grad-CAM visualization highlights regions of the lung that influenced this diagnosis. "
                f"Further clinical evaluation and additional imaging may be warranted."
            )
        else:
            description = (
                f"The CNN model indicates normal lung appearance with {confidence*100:.1f}% confidence. "
                f"No significant abnormalities were detected in the analyzed regions."
            )

        disclaimer = (
            "\n\n⚠️ Disclaimer: This AI analysis is for educational purposes only. "
            "Always consult qualified healthcare professionals for medical decisions."
        )
        return description + disclaimer

    def is_available(self):
        """Check if Ollama service is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            return any(self.model_name in model.get("name", "") for model in models)
        except Exception:
            return False
