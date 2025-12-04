"""WhatsApp bot - refactored to use new functional approach."""

import os
import time

import requests
from dotenv import load_dotenv
from flask import Flask, request
from requests.auth import HTTPBasicAuth
from twilio.twiml.messaging_response import MessagingResponse

from src.core.diagnosis import diagnose_from_path, load_cnn_model
from src.utils.config import config
from src.utils.file_utils import cleanup_file, ensure_directory

load_dotenv()

# Initialize bot app
app = Flask(__name__)

# Load model
print("Loading CNN model for WhatsApp bot...")
cnn_model = load_cnn_model(config.cnn_model_path)
print("✓ Model loaded")

# Ensure upload directory exists
ensure_directory(config.uploads_dir)

# Twilio credentials
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")


@app.route("/webhook", methods=["POST"])
def webhook():
    """Handle incoming WhatsApp messages with X-ray images."""
    resp = MessagingResponse()
    num_media = int(request.form.get("NumMedia", 0))

    # No media sent
    if num_media == 0:
        resp.message("Hi! Please send a lung X-ray image for analysis. 🫁")
        return str(resp)

    # Download image from Twilio
    media_url = request.form.get("MediaUrl0")

    try:
        image_response = requests.get(media_url, auth=HTTPBasicAuth(account_sid, auth_token))
        image_response.raise_for_status()
    except Exception as e:
        print(f"Error downloading image: {e}")
        resp.message("Error downloading the image. Please try again.")
        return str(resp)

    # Save image
    timestamp = int(time.time())
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(config.uploads_dir, filename)

    with open(filepath, "wb") as f:
        f.write(image_response.content)

    # Diagnose
    try:
        diagnosis, confidence = diagnose_from_path(cnn_model, filepath)

        # Format message
        if diagnosis == "PNEUMONIA":
            message = (
                f"🚨 **Diagnosis**: Pneumonia detected with {confidence * 100:.1f}% confidence."
            )
        else:
            message = (
                f"✅ **Diagnosis**: No signs of pneumonia. Confidence: {confidence * 100:.1f}%."
            )

        print(f"WhatsApp diagnosis: {diagnosis} ({confidence:.1%})")
        resp.message(message)

    except Exception as e:
        print(f"Error during processing: {e}")
        resp.message("I had an error processing the image. Please try again later.")

    finally:
        # Clean up uploaded file
        cleanup_file(filepath)

    return str(resp)


if __name__ == "__main__":
    print("\n🤖 Starting PneumoFinder WhatsApp Bot...")
    print("   - Webhook endpoint: POST /webhook\n")
    app.run(debug=True)
