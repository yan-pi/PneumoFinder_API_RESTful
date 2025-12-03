import os
import time

import requests
from flask import Flask, request
from pneumonia_service import PneumoniaDetectorService
from pulmao_service import LungDetector
from requests.auth import HTTPBasicAuth
from twilio.twiml.messaging_response import MessagingResponse

# Initialize the detectors with the paths to the saved models
lung_detector = LungDetector("../models/pulmao_model.keras")
pneumonia_detector = PneumoniaDetectorService("../models/pneumonia_model.keras")

app = Flask(__name__)

IMAGES_FOLDER = "imgs_pulmoes"
os.makedirs(IMAGES_FOLDER, exist_ok=True)

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")


@app.route("/webhook", methods=["POST"])
def webhook():
    resp = MessagingResponse()
    num_media = int(request.form.get("NumMedia", 0))

    if num_media == 0:
        resp.message("Hi! Please send a lung X-ray image for analysis. 🫁")
        return str(resp)

    media_url = request.form.get("MediaUrl0")

    try:
        image_response = requests.get(media_url, auth=HTTPBasicAuth(account_sid, auth_token))
        image_response.raise_for_status()
    except Exception as e:
        print("SID:", account_sid)
        print("TOKEN:", auth_token)
        print("Error downloading image:", e)
        resp.message("Error downloading the image. Please try again.")
        return str(resp)

    timestamp = int(time.time())
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(IMAGES_FOLDER, filename)

    with open(filepath, "wb") as f:
        f.write(image_response.content)

    try:
        # Step 1: verify if it's a lung
        lung_class, lung_confidence = lung_detector.detect_image(filepath)

        if lung_class != "LUNG":
            message = f"⛔ The sent image does not appear to be a lung X-ray.\n❌ Confidence: {lung_confidence * 100:.1f}% "
            resp.message(message)
            os.remove(filepath)
            return str(resp)

        # Step 2: diagnose pneumonia
        pneumonia_class, pneumonia_confidence = pneumonia_detector.diagnose_image(filepath)
        os.remove(filepath)

        if pneumonia_class == "PNEUMONIA":
            message = (
                f"🆘 The image is of a lung.\n"
                f"🚨 **Diagnosis**: Pneumonia detected with confidence of {pneumonia_confidence * 100:.1f}%."
            )
        else:
            message = (
                f"✅ The image is of a lung.\n"
                f"🎉 **Diagnosis**: No signs of pneumonia. Confidence: {pneumonia_confidence * 100:.1f}%."
            )

        print(message)
        resp.message(message)

    except Exception as e:
        print("Error during processing:", e)
        resp.message("I had an error processing the image. Please try again later.")

    return str(resp)


if __name__ == "__main__":
    app.run(debug=True)
