import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from service import pneumonia_service as pf
from service import pulmao_service as pulm

load_dotenv()


app = Flask(__name__)
CORS(app)

# Initialize both detectors
pneumonia_detector = pf.PneumoniaDetectorService("models/pneumonia_model.keras")
lung_detector = pulm.LungDetector("models/pulmao_model.keras")


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


@app.route("/verificar_pulmao", methods=["POST"])
def verify_lung():
    if "imagem" not in request.files:
        return jsonify({"error": "No image sent."}), 400

    image = request.files["imagem"]
    temp_path = os.path.join("temp", image.filename)
    image.save(temp_path)

    try:
        class_name, confidence = lung_detector.detect_image(temp_path)
        os.remove(temp_path)
        response = {"class": str(class_name), "confidence": float(round(confidence.item(), 2))}
        print("Response generated for frontend (lung):", response)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/diagnostico_completo", methods=["POST"])
def complete_diagnosis():
    if "imagem" not in request.files:
        return jsonify({"error": "No image sent."}), 400

    image = request.files["imagem"]
    temp_path = os.path.join("temp", image.filename)
    image.save(temp_path)

    try:
        # First: check if it's a lung
        lung_class, lung_confidence = lung_detector.detect_image(temp_path)

        if lung_class != "LUNG":
            os.remove(temp_path)
            return jsonify(
                {
                    "lung_class": "NOT A LUNG",
                    "confidence": float(round(lung_confidence.item(), 2)),
                }
            )

        # Second: diagnose pneumonia
        pneumonia_class, pneumonia_confidence = pneumonia_detector.diagnose_image(temp_path)
        os.remove(temp_path)

        response = {
            "lung_class": "LUNG",
            "lung_confidence": float(round(lung_confidence.item(), 2)),
            "pneumonia_class": pneumonia_class,
            "pneumonia_confidence": float(round(pneumonia_confidence.item(), 2)),
        }
        print("Complete response for frontend:", response)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("temp", exist_ok=True)
    app.run(debug=True, port=5001)
