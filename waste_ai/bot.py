"""Run the real Telegram photo classifier after a model has been trained.

Never put bot tokens in source code, logs, or a Codex prompt.
"""
from __future__ import annotations
import asyncio
from io import BytesIO
import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from waste_ai.config import MAX_IMAGE_BYTES
from waste_ai.inference import WasteClassifier, format_result

log = logging.getLogger("waste_ai.bot")
CLASSIFIER = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Hello! I'm a Waste Classification AI bot.\n\n"
        "Send a photo or image file of one waste item. I'll predict Recyclable or "
        "Non-Recyclable and return a confidence score.\n\n"
        "This is an academic demo trained from provisional source-folder mappings. "
        "It does not establish local recycling acceptance.\n"
        "Use /help to learn more.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "📷 How to use:\n"
        "1. Take a clear photo of ONE waste item.\n"
        "2. Send it here as a Telegram photo or PNG/JPEG image document.\n"
        "3. Receive a Recyclable or Non-Recyclable prediction and confidence automatically.\n\n"
        "Commands: /start, /help, /about.\n"
        "Your uploaded photo is processed in memory and not saved by this bot."
    )


async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "ITM 360 — Group 4 binary waste classification research prototype.\n"
        "The bot predicts Recyclable or Non-Recyclable directly from a two-output model.\n"
        "Dataset targets were mapped from source folders and are not human-verified.\n"
        "The deployed model is selected using validation performance, not the test set.\n"
        "Photo predictions are research estimates; local recycling acceptance varies."
    )


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    try:
        if msg.photo:
            obj = msg.photo[-1]
        elif msg.document and msg.document.mime_type in ("image/jpeg", "image/png", "image/webp"):
            obj = msg.document
        else:
            await msg.reply_text("Please send an image as a photo, or a JPG/PNG image document.")
            return
        if obj.file_size is not None and obj.file_size > MAX_IMAGE_BYTES:
            await msg.reply_text("Please send an image smaller than 10 MB.")
            return
        tfile = await obj.get_file()
        data = BytesIO()
        await tfile.download_to_memory(out=data)
        if data.tell() > MAX_IMAGE_BYTES:
            await msg.reply_text("Please send an image smaller than 10 MB.")
            return
        prediction = await asyncio.to_thread(CLASSIFIER.predict_bytes, data.getvalue())
        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
        await msg.reply_text(format_result(prediction, threshold))
    except ValueError as e:
        await msg.reply_text(f"Sorry, I couldn't process that image: {e}")
    except Exception:
        log.exception("Image processing failed")
        await msg.reply_text("Sorry, prediction failed. Please retry with a clear image.")


async def handle_non_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Send a waste photo to classify it, or use /help.")


def main():
    global CLASSIFIER
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Quiet Telegram/httpx logger: never log request URLs with a token in them.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token or token == "replace_with_your_real_token":
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in private .env. Do not publish your token.")
    CLASSIFIER = WasteClassifier()  # fail fast if model missing, before connecting to Telegram
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_non_image))
    log.info("Waste Classification Telegram bot is running (model: %s)", CLASSIFIER.model_path.stem)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
