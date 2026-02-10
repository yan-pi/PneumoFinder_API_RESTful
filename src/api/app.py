"""PneumoFinder API - Refactored with functional approach."""

import hashlib
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
from src.core.medical_pipeline import MedicalPipeline
from src.core.visualization import find_last_conv_layer, find_resnet_base
from src.db import (
    get_clinical_description,
    get_diagnosis,
    get_recent_diagnoses,
    get_visualization,
    init_database,
    save_clinical_description,
    save_diagnosis,
    save_visualization,
    search_similar_cases,
)
from src.utils.config import config
from src.utils.file_utils import cleanup_file, ensure_directory, save_uploaded_file

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize database
print("Initializing database...")
init_database()

# Load models at startup
print("Loading CNN model...")
cnn_model = load_cnn_model(config.cnn_model_path)
print("Finding ResNet base and last conv layer...")
resnet_base = find_resnet_base(cnn_model)
last_conv_layer = find_last_conv_layer(resnet_base)
print(f"✓ Models loaded. Grad-CAM target: {last_conv_layer.name}")

# Initialize medical pipeline (lazy-loaded models)
print("Initializing medical pipeline...")
medical_pipeline = MedicalPipeline(use_rag=True, vision_model="llava-llama3")
print("✓ Medical pipeline initialized (2-stage: Vision + Medical)")


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

    Returns: {"diagnosis": "PNEUMONIA" or "NORMAL", "confidence": 0.87, "diagnosis_id": 123}
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    temp_path = None
    try:
        # Save uploaded file
        image_file = request.files["image"]
        temp_path = save_uploaded_file(image_file, config.temp_dir)

        # Compute image hash for deduplication
        with open(temp_path, "rb") as f:
            image_hash = hashlib.sha256(f.read()).hexdigest()

        # Diagnose
        diagnosis, confidence = diagnose_from_path(cnn_model, temp_path)

        # Save to database
        diagnosis_id = save_diagnosis(
            image_hash=image_hash,
            diagnosis=diagnosis,
            confidence=confidence,
            metadata={"endpoint": "/diagnose", "filename": image_file.filename},
        )

        return jsonify(
            {
                "diagnosis": diagnosis,
                "confidence": round(confidence, 2),
                "diagnosis_id": diagnosis_id,
            }
        )

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
        "diagnosis_id": 123,
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
        image_file = request.files["image"]
        temp_path = save_uploaded_file(image_file, config.temp_dir)

        # Compute image hash for deduplication
        with open(temp_path, "rb") as f:
            image_hash = hashlib.sha256(f.read()).hexdigest()

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

        # Save to database
        diagnosis_id = save_diagnosis(
            image_hash=image_hash,
            diagnosis=diagnosis,
            confidence=confidence,
            metadata={"endpoint": "/diagnose/explained", "filename": image_file.filename},
        )

        # Save visualizations as BLOBs
        with open(heatmap_path, "rb") as f:
            heatmap_bytes = f.read()
        with open(overlay_path, "rb") as f:
            overlay_bytes = f.read()
        save_visualization(diagnosis_id, heatmap_bytes, overlay_bytes)

        # Save clinical description to database and vector store
        save_clinical_description(diagnosis_id, description, diagnosis, confidence)

        # Generate URLs for visualizations
        heatmap_url = f"/static/temp/{os.path.basename(heatmap_path)}"
        overlay_url = f"/static/temp/{os.path.basename(overlay_path)}"

        return jsonify(
            {
                "diagnosis": diagnosis,
                "confidence": round(confidence, 2),
                "diagnosis_id": diagnosis_id,
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


@app.route("/api/diagnoses/<int:diagnosis_id>", methods=["GET"])
def get_diagnosis_by_id(diagnosis_id):
    """
    Retrieve a diagnosis by ID with all associated data.

    GET /api/diagnoses/123

    Returns: {
        "id": 123,
        "diagnosis": "PNEUMONIA",
        "confidence": 0.87,
        "created_at": "2024-01-15T10:30:00",
        "metadata": {...},
        "description": "Clinical description...",
        "has_visualizations": true
    }
    """
    try:
        # Get diagnosis
        diagnosis_data = get_diagnosis(diagnosis_id)
        if not diagnosis_data:
            return jsonify({"error": "Diagnosis not found"}), 404

        # Get clinical description
        description = get_clinical_description(diagnosis_id)
        if description:
            diagnosis_data["description"] = description

        # Check if visualizations exist
        visualization = get_visualization(diagnosis_id)
        diagnosis_data["has_visualizations"] = visualization is not None

        return jsonify(diagnosis_data)

    except Exception as e:
        print(f"Error in /api/diagnoses/{diagnosis_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/search/similar", methods=["POST"])
def search_similar_diagnoses():
    """
    Semantic search for similar past diagnoses.

    POST /api/search/similar
    JSON body: {"query": "patient with bilateral infiltrates", "top_k": 5}

    Returns: {
        "results": [
            {
                "diagnosis_id": 123,
                "diagnosis": "PNEUMONIA",
                "confidence": 0.87,
                "similarity": 0.71,
                "description": "Clinical description...",
                "created_at": "2024-01-15T10:30:00"
            },
            ...
        ]
    }
    """
    try:
        # Validate request
        if not request.json or "query" not in request.json:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        query = request.json["query"]
        top_k = request.json.get("top_k", 5)

        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1 or top_k > 50:
            return jsonify({"error": "top_k must be between 1 and 50"}), 400

        # Search similar cases
        results = search_similar_cases(query, top_k=top_k)

        return jsonify({"query": query, "top_k": top_k, "results": results})

    except Exception as e:
        print(f"Error in /api/search/similar: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/diagnoses/recent", methods=["GET"])
def get_recent_diagnoses_list():
    """
    Get recent diagnoses.

    GET /api/diagnoses/recent?limit=10

    Returns: {
        "diagnoses": [
            {
                "id": 123,
                "diagnosis": "PNEUMONIA",
                "confidence": 0.87,
                "created_at": "2024-01-15T10:30:00"
            },
            ...
        ]
    }
    """
    try:
        limit = request.args.get("limit", default=10, type=int)

        # Validate limit
        if limit < 1 or limit > 100:
            return jsonify({"error": "limit must be between 1 and 100"}), 400

        diagnoses = get_recent_diagnoses(limit=limit)

        return jsonify({"count": len(diagnoses), "diagnoses": diagnoses})

    except Exception as e:
        print(f"Error in /api/diagnoses/recent: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze_xray():
    """
    Complete medical analysis with 2-stage LLM pipeline (Vision + Medical).

    POST /analyze
    Form data:
        - image (file): X-ray image

    Returns: {
        "success": true,
        "diagnosis_id": 123,
        "cnn_diagnosis": {
            "prediction": "PNEUMONIA",
            "confidence": 0.89
        },
        "stage1_vision": {
            "findings_en": "Bilateral opacities...",
            "model": "llava-llama3",
            "latency_s": 65.3
        },
        "stage2_medical": {
            "report_en": "The chest X-ray reveals...",
            "model": "BioMistral-7B",
            "latency_s": 61.8
        },
        "final_report_en": "The chest X-ray reveals...",
        "total_latency_s": 128.1,
        "timestamp": "2024-01-15 10:30:00"
    }
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400

    temp_path = None
    try:
        # Save uploaded file
        image_file = request.files["image"]
        temp_path = save_uploaded_file(image_file, config.temp_dir)

        # Validate image format
        allowed_extensions = {".jpg", ".jpeg", ".png"}
        filename = image_file.filename or "unknown.jpg"
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in allowed_extensions:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Invalid image format. Allowed: {', '.join(allowed_extensions)}",
                    }
                ),
                400,
            )

        # Compute image hash for deduplication
        with open(temp_path, "rb") as f:
            image_hash = hashlib.sha256(f.read()).hexdigest()

        # Stage 0: CNN diagnosis
        print(f"[/analyze] Running CNN diagnosis on {image_file.filename}...")
        diagnosis, confidence = diagnose_from_path(cnn_model, temp_path)

        cnn_diagnosis = {"prediction": diagnosis, "confidence": float(confidence)}

        # Save CNN diagnosis to database
        diagnosis_id = save_diagnosis(
            image_hash=image_hash,
            diagnosis=diagnosis,
            confidence=confidence,
            metadata={"endpoint": "/analyze", "filename": image_file.filename},
        )

        # Run medical pipeline (Stages 1-2)
        print("[/analyze] Running 2-stage medical pipeline...")
        result = medical_pipeline.analyze_xray(temp_path, cnn_diagnosis)

        if not result["success"]:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Pipeline failed: {result.get('error', 'Unknown error')}",
                    }
                ),
                500,
            )

        # Add diagnosis_id to result
        result["diagnosis_id"] = diagnosis_id

        # Save clinical description to database
        final_report = result.get("final_report_en")
        if final_report:
            save_clinical_description(diagnosis_id, final_report, diagnosis, confidence)

        print(
            f"[/analyze] ✅ Analysis complete in {result['total_latency_s']:.1f}s "
            f"(diagnosis_id: {diagnosis_id})"
        )

        return jsonify(result)

    except Exception as e:
        print(f"Error in /analyze: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if temp_path:
            cleanup_file(temp_path)


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

    # Run app
    print(f"\nStarting PneumoFinder API on port {config.api_port}...")
    print(f"   - Medical analysis: POST /analyze (2-stage pipeline)")
    print(f"   - Simple diagnosis: POST /diagnose")
    print(f"   - With explanation: POST /diagnose/explained")
    print(f"   - Health check: GET /health")
    print(f"   - Get diagnosis: GET /api/diagnoses/<id>")
    print(f"   - Search similar: POST /api/search/similar")
    print(f"   - Recent diagnoses: GET /api/diagnoses/recent\n")

    app.run(debug=config.debug_mode, port=config.api_port)
