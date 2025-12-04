"""LLM integration for clinical description generation."""

import base64
import os

import requests


def load_medical_prompt_template(prompt_path: str = "prompts/medical_analysis.txt") -> str:
    """
    Load medical analysis prompt template from file.

    Args:
        prompt_path: Path to prompt template file

    Returns:
        Prompt template string
    """
    if os.path.exists(prompt_path):
        with open(prompt_path) as f:
            return f.read()

    # Fallback prompt if file doesn't exist
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


def encode_image_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 string.

    Args:
        image_path: Path to image file

    Returns:
        Base64-encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def call_ollama_api(
    prompt: str,
    image_base64: str,
    model_name: str = "llava:7b",
    ollama_host: str = "http://localhost:11434",
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> str | None:
    """
    Call Ollama API for multimodal LLM inference.

    Args:
        prompt: Text prompt for the LLM
        image_base64: Base64-encoded image
        model_name: Name of Ollama model
        ollama_host: Ollama API host
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text response or None if error
    """
    api_url = f"{ollama_host}/api/generate"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in Ollama API call: {e}")
        return None


def check_ollama_availability(
    ollama_host: str = "http://localhost:11434", model_name: str = "llava:7b"
) -> bool:
    """
    Check if Ollama service is running and model is available.

    Args:
        ollama_host: Ollama API host
        model_name: Name of model to check

    Returns:
        True if service and model are available
    """
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        return any(model_name in model.get("name", "") for model in models)
    except Exception:
        return False


def generate_fallback_description(diagnosis: str, confidence: float) -> str:
    """
    Generate simple fallback description when LLM is unavailable.

    Args:
        diagnosis: Diagnosis result (PNEUMONIA or NORMAL)
        confidence: Model confidence (0-1)

    Returns:
        Fallback description text
    """
    if diagnosis == "PNEUMONIA":
        description = (
            f"The CNN model detected signs of pneumonia with {confidence * 100:.1f}% confidence. "
            "The Grad-CAM visualization highlights regions of the lung that influenced this diagnosis. "
            "Further clinical evaluation and additional imaging may be warranted."
        )
    else:
        description = (
            f"The CNN model indicates normal lung appearance with {confidence * 100:.1f}% confidence. "
            "No significant abnormalities were detected in the analyzed regions."
        )

    return description


def add_medical_disclaimer(description: str, confidence: float) -> str:
    """
    Add medical disclaimer to description.

    Args:
        description: Clinical description text
        confidence: Model confidence (0-1)

    Returns:
        Description with disclaimer appended
    """
    disclaimer = (
        "\n\n⚠️ Disclaimer: This AI analysis is for educational purposes only. "
        "Always consult qualified healthcare professionals for medical decisions."
    )

    # Add confidence note if low
    if confidence < 0.7:
        confidence_note = "\n\nNote: Model confidence is moderate. Manual review recommended."
        description += confidence_note

    return description + disclaimer


def generate_clinical_description(
    diagnosis: str,
    confidence: float,
    original_image_path: str,
    overlay_image_path: str | None = None,
    prompt_template_path: str = "prompts/medical_analysis.txt",
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "llava:7b",
) -> str:
    """
    Generate clinical description using LLM or fallback.

    Args:
        diagnosis: Diagnosis result (PNEUMONIA or NORMAL)
        confidence: Model confidence (0-1)
        original_image_path: Path to original X-ray image
        overlay_image_path: Optional path to Grad-CAM overlay
        prompt_template_path: Path to prompt template
        ollama_host: Ollama API host
        ollama_model: Ollama model name

    Returns:
        Clinical description text with disclaimer
    """
    # Try LLM first
    try:
        # Load prompt template
        template = load_medical_prompt_template(prompt_template_path)
        prompt = template.format(class_name=diagnosis, confidence=f"{confidence * 100:.1f}")

        # Use overlay if available, otherwise original
        image_path = overlay_image_path if overlay_image_path else original_image_path
        image_b64 = encode_image_to_base64(image_path)

        # Call LLM
        llm_response = call_ollama_api(
            prompt, image_b64, model_name=ollama_model, ollama_host=ollama_host
        )

        if llm_response:
            return add_medical_disclaimer(llm_response, confidence)

    except Exception as e:
        print(f"Error generating LLM description: {e}")

    # Fallback if LLM fails
    fallback = generate_fallback_description(diagnosis, confidence)
    return add_medical_disclaimer(fallback, confidence)
