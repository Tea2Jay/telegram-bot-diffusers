import asyncio
from fileinput import FileInput
import json
import logging
import os
from telegram import Update, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    filters,
    MessageHandler,
    InlineQueryHandler,
)
import torch
from diffuse import Diffuser, lms


config = open("./config.json")
config = json.load(config)

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
diffuser = Diffuser(model_id, lms, torch.float16)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="I'm a diffuser, please pass a text to diffues!",
    )


async def diffuse(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args)
    image = diffuser.diffuse(prompt)
    user = update.message.chat.username
    image_fake_id = update.message.id
    os.makedirs(f"./{user}", exist_ok=True)
    image_path = f"./{user}/{image_fake_id}_{prompt}.png"
    image.save(image_path)
    # tmp = FileInput("astronaut_rides_horse.png")
    await context.bot.send_photo(
        chat_id=update.effective_chat.id,
        photo=open(image_path, "rb"),
    )


if __name__ == "__main__":
    application = ApplicationBuilder().token(config["TOKEN"]).build()

    start_handler = CommandHandler("start", start)
    diffuse_handler = CommandHandler("diffuse", diffuse)

    application.add_handler(start_handler)
    application.add_handler(diffuse_handler)

    application.run_polling()
