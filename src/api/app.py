"""PneumoFinder API - Refactored with functional approach."""

import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from src.core.clinical_description import generate_clinical_description
from src.core.diagnosis import (
    diagnose_from_path,
    diagnose_with_visualization,
    load_cnn_model,
)
from src.core.visualization import find_last_conv_layer, find_resnet_base
from src.utils.config import config
from src.utils.file_utils import cleanup_file, ensure_directory, save_uploaded_file

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models at startup
print("Loading CNN model...")
cnn_model = load_cnn_model(config.cnn_model_path)
print("Finding ResNet base and last conv layer...")
resnet_base = find_resnet_base(cnn_model)
last_conv_layer = find_last_conv_layer(resnet_base)
print(f"✓ Models loaded. Grad-CAM target: {last_conv_layer.name}")


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "PneumoFinder API"})


@app.route("/diagnose", methods=["POST"])
def diagnose_pneumonia():
    """
    Simple pneumonia diagnosis without visualization.

    POST /diagnose
    Form data: image (file)

    Returns: {"diagnosis": "PNEUMONIA" or "NORMAL", "confidence": 0.87}
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    temp_path = None
    try:
        # Save uploaded file
        temp_path = save_uploaded_file(request.files["image"], config.temp_dir)

        # Diagnose
        diagnosis, confidence = diagnose_from_path(cnn_model, temp_path)

        return jsonify({"diagnosis": diagnosis, "confidence": round(confidence, 2)})

    except Exception as e:
        print(f"Error in /diagnose: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_path:
            cleanup_file(temp_path)


@app.route("/diagnose/complete", methods=["POST"])
def complete_diagnosis():
    """
    Alias for /diagnose (backward compatibility).

    POST /diagnose/complete
    Form data: image (file)

    Returns: {"diagnosis": "PNEUMONIA" or "NORMAL", "confidence": 0.87}
    """
    return diagnose_pneumonia()


@app.route("/diagnose/explained", methods=["POST"])
def diagnose_with_explanation():
    """
    Multimodal diagnosis with Grad-CAM visualizations and LLM explanation.

    POST /diagnose/explained
    Form data: image (file)

    Returns: {
        "diagnosis": "PNEUMONIA",
        "confidence": 0.87,
        "description": "Clinical description from LLM...",
        "heatmap_url": "/static/temp/filename_heatmap.png",
        "overlay_url": "/static/temp/filename_overlay.png"
    }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    temp_path = None
    try:
        # Save uploaded file
        temp_path = save_uploaded_file(request.files["image"], config.temp_dir)

        # Diagnose with visualization
        diagnosis, confidence, heatmap_path, overlay_path = diagnose_with_visualization(
            cnn_model, resnet_base, last_conv_layer, temp_path, config.temp_dir
        )

        # Generate clinical description
        description = generate_clinical_description(
            diagnosis,
            confidence,
            temp_path,
            overlay_path,
            config.medical_prompt_path,
            config.ollama_host,
            config.ollama_model,
        )

        # Generate URLs for visualizations
        heatmap_url = f"/static/temp/{os.path.basename(heatmap_path)}"
        overlay_url = f"/static/temp/{os.path.basename(overlay_path)}"

        return jsonify(
            {
                "diagnosis": diagnosis,
                "confidence": round(confidence, 2),
                "description": description,
                "heatmap_url": heatmap_url,
                "overlay_url": overlay_url,
            }
        )

    except Exception as e:
        print(f"Error in /diagnose/explained: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_path:
            cleanup_file(temp_path)


@app.route("/static/temp/<filename>")
def serve_temp_file(filename):
    """Serve heatmap/overlay images from temp folder."""
    return send_from_directory(config.temp_dir, filename)


# Backward compatibility - old Portuguese endpoints
@app.route("/diagnosticar_pneumonia", methods=["POST"])
def diagnosticar_pneumonia_legacy():
    """Legacy endpoint - redirects to /diagnose."""
    return diagnose_pneumonia()


@app.route("/diagnostico_completo", methods=["POST"])
def diagnostico_completo_legacy():
    """Legacy endpoint - redirects to /diagnose/complete."""
    return complete_diagnosis()


@app.route("/diagnosticar_com_descricao", methods=["POST"])
def diagnosticar_com_descricao_legacy():
    """Legacy endpoint - redirects to /diagnose/explained."""
    return diagnose_with_explanation()


if __name__ == "__main__":
    # Ensure directories exist
    ensure_directory(config.temp_dir)
    ensure_directory(config.reports_dir)

    # Run app
    print(f"\n🚀 Starting PneumoFinder API on port {config.api_port}...")
    print(f"   - Simple diagnosis: POST /diagnose")
    print(f"   - With explanation: POST /diagnose/explained")
    print(f"   - Health check: GET /health\n")

    app.run(debug=config.debug_mode, port=config.api_port)
