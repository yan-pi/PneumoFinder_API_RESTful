"""Entry point for WhatsApp bot."""

if __name__ == "__main__":
    from src.bots.whatsapp_bot import app

    print("\n🤖 Starting PneumoFinder WhatsApp Bot...")
    print("   - Webhook endpoint: POST /webhook\n")
    app.run(debug=True)
