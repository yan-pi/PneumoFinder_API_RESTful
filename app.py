import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from service import pneumonia_service as pf

load_dotenv()


app = Flask(__name__)
CORS(app)

# Initialize pneumonia detector
pneumonia_detector = pf.PneumoniaDetectorService("models/pneumonia_model.keras")


@app.route("/diagnosticar_pneumonia", methods=["POST"])
def diagnose_pneumonia():
    if "imagem" not in request.files:
        return jsonify({"error": "No image sent."}), 400

    image = request.files["imagem"]
    temp_path = os.path.join("temp", image.filename)
    image.save(temp_path)

    try:
        class_name, confidence = pneumonia_detector.diagnose_image(temp_path)
        os.remove(temp_path)
        response = {"class": str(class_name), "confidence": float(round(confidence.item(), 2))}
        print("Response generated for frontend:", response)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/diagnostico_completo", methods=["POST"])
def complete_diagnosis():
    """Complete diagnosis - same as /diagnosticar_pneumonia for now"""
    if "imagem" not in request.files:
        return jsonify({"error": "No image sent."}), 400

    image = request.files["imagem"]
    temp_path = os.path.join("temp", image.filename)
    image.save(temp_path)

    try:
        class_name, confidence = pneumonia_detector.diagnose_image(temp_path)
        os.remove(temp_path)
        response = {"class": str(class_name), "confidence": float(round(confidence.item(), 2))}
        print("Complete response for frontend:", response)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/diagnosticar_com_descricao", methods=["POST"])
def diagnose_with_llm_description():
    """
    Diagnose pneumonia with LLM-generated clinical description.
    Returns: class, confidence, description, heatmap_url, overlay_url
    """
    if "imagem" not in request.files:
        return jsonify({"error": "No image sent."}), 400

    image = request.files["imagem"]
    temp_path = os.path.join("temp", image.filename)
    image.save(temp_path)

    try:
        (
            class_name,
            confidence,
            llm_description,
            overlay_path,
            heatmap_path,
        ) = pneumonia_detector.diagnose_with_llm_explanation(temp_path)

        # Generate URLs for heatmap and overlay
        heatmap_url = f"/static/temp/{os.path.basename(heatmap_path)}"
        overlay_url = f"/static/temp/{os.path.basename(overlay_path)}"

        # Handle confidence conversion (may be numpy array or float)
        confidence_value = float(confidence.item()) if hasattr(confidence, "item") else float(confidence)

        response = {
            "class": str(class_name),
            "confidence": round(confidence_value, 2),
            "description": llm_description,
            "heatmap_url": heatmap_url,
            "overlay_url": overlay_url,
        }

        # Clean up original temp file (keep heatmap/overlay for serving)
        os.remove(temp_path)

        print(f"Multimodal response generated: {class_name} ({confidence:.1%})")
        return jsonify(response)
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500


@app.route("/static/temp/<filename>")
def serve_temp_file(filename):
    """Serve heatmap/overlay images from temp folder"""
    return send_from_directory("temp", filename)


if __name__ == "__main__":
    os.makedirs("temp", exist_ok=True)
    app.run(debug=True, port=5001)
