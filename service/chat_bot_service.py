import os
import time

import requests
from flask import Flask, request
from pneumonia_service import DetectorDePneumoniaService
from pulmao_service import DetectorDePulmao
from requests.auth import HTTPBasicAuth
from twilio.twiml.messaging_response import MessagingResponse

# Inicializa os detectores com os caminhos dos modelos salvos
detector_pulmao = DetectorDePulmao("../models/pulmao_model.keras")
detector_pneumonia = DetectorDePneumoniaService("../models/pneumonia_model.keras")

app = Flask(__name__)

PASTA_IMAGENS = "imgs_pulmoes"
os.makedirs(PASTA_IMAGENS, exist_ok=True)

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")


@app.route("/webhook", methods=["POST"])
def webhook():
    resp = MessagingResponse()
    num_media = int(request.form.get("NumMedia", 0))

    if num_media == 0:
        resp.message("Oi! Por favor, envie uma imagem de raio-x do pulmão para análise. 🫁")
        return str(resp)

    media_url = request.form.get("MediaUrl0")

    try:
        image_response = requests.get(media_url, auth=HTTPBasicAuth(account_sid, auth_token))
        image_response.raise_for_status()
    except Exception as e:
        print("SID:", account_sid)
        print("TOKEN:", auth_token)
        print("Erro ao baixar imagem:", e)
        resp.message("Erro ao baixar a imagem. Tente novamente, por favor.")
        return str(resp)

    timestamp = int(time.time())
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(PASTA_IMAGENS, filename)

    with open(filepath, "wb") as f:
        f.write(image_response.content)

    try:
        # Passo 1: verificar se é pulmão
        classe_pulmao, confianca_pulmao = detector_pulmao.detectar_imagem(filepath)

        if classe_pulmao != "PULMÃO":
            mensagem = f"⛔ A imagem enviada não parece ser um raio-x de pulmão.\n❌ Confiança: {confianca_pulmao * 100:.1f}% "
            resp.message(mensagem)
            os.remove(filepath)
            return str(resp)

        # Passo 2: diagnosticar pneumonia
        classe_pneumonia, confianca_pneumonia = detector_pneumonia.diagnosticar_imagem(filepath)
        os.remove(filepath)

        if classe_pneumonia == "PNEUMONIA":
            mensagem = (
                f"🆘 A imagem é de um pulmão.\n"
                f"🚨 **Diagnóstico**: Pneumonia detectada com confiança de {confianca_pneumonia * 100:.1f}%."
            )
        else:
            mensagem = (
                f"✅ A imagem é de um pulmão.\n"
                f"🎉 **Diagnóstico**: Não há sinais de pneumonia. Confiança: {confianca_pneumonia * 100:.1f}%."
            )

        print(mensagem)
        resp.message(mensagem)

    except Exception as e:
        print("Erro durante o processamento:", e)
        resp.message("Tive um erro ao processar a imagem. Tente novamente mais tarde, tá bom?")

    return str(resp)


if __name__ == "__main__":
    app.run(debug=True)
