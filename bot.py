import os
import asyncio
import threading
from collections import defaultdict, deque

import discord
from flask import Flask
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

MODEL = "openai/gpt-oss-120b"  # Groq-hosted open-weight 120B model
SYSTEM_PROMPT = (
    "You are a helpful, friendly AI assistant living in a Discord server. "
    "Keep replies concise unless the user asks for detail. "
    "Use Discord-flavored markdown (e.g. **bold**, `code`) where useful."
)
MAX_HISTORY = 10          # messages remembered per channel
MAX_REPLY_CHARS = 1900    # Discord's hard limit is 2000

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

intents = discord.Intents.default()
intents.message_content = True  # required to read message text
bot = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(bot)

# per-channel short-term memory: channel_id -> deque of {"role", "content"}
history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

# --- Keep-alive web server ----------------------------------------------
# Render's free tier (and some Railway setups) put a web service to sleep
# after ~15 min of no HTTP traffic. This tiny Flask app gives UptimeRobot
# something to ping every few minutes so the service never goes idle.
keep_alive_app = Flask(__name__)


@keep_alive_app.route("/")
def home():
    return "Bot is alive!"


def run_keep_alive():
    port = int(os.environ.get("PORT", 8080))
    keep_alive_app.run(host="0.0.0.0", port=port)


def ask_groq(channel_id: int, user_message: str) -> str:
    history[channel_id].append({"role": "user", "content": user_message})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history[channel_id])

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=800,
    )
    reply = response.choices[0].message.content
    history[channel_id].append({"role": "assistant", "content": reply})
    return reply


@bot.event
async def on_ready():
    await tree.sync()  # registers slash commands with Discord (global — can take up to ~1hr to first appear everywhere)
    print(f"Logged in as {bot.user} (id: {bot.user.id})")


# --- Slash commands -------------------------------------------------------

@tree.command(name="ask", description="Ask the AI assistant something")
async def ask_command(interaction: discord.Interaction, question: str):
    await interaction.response.defer()  # shows "thinking..." while Groq responds
    try:
        loop = asyncio.get_event_loop()
        reply = await loop.run_in_executor(None, ask_groq, interaction.channel_id, question)
    except Exception as e:
        reply = f"Sorry, I ran into an error: `{e}`"

    # Slash command replies also cap at 2000 chars; send first chunk as the
    # reply, any overflow as follow-up messages.
    chunks = [reply[i:i + MAX_REPLY_CHARS] for i in range(0, len(reply), MAX_REPLY_CHARS)] or [""]
    await interaction.followup.send(chunks[0])
    for chunk in chunks[1:]:
        await interaction.followup.send(chunk)


@tree.command(name="reset", description="Clear the AI's memory of this channel's conversation")
async def reset_command(interaction: discord.Interaction):
    history[interaction.channel_id].clear()
    await interaction.response.send_message("Conversation history cleared for this channel.", ephemeral=True)


@tree.command(name="help", description="Show what this bot can do")
async def help_command(interaction: discord.Interaction):
    await interaction.response.send_message(
        "**SAIChatbot commands:**\n"
        "`/ask <question>` — ask me anything\n"
        "`/reset` — clear this channel's conversation memory\n"
        "`/help` — show this message\n\n"
        "You can also just @mention me or DM me directly instead of using commands.",
        ephemeral=True,
    )


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user in message.mentions

    # Only respond to DMs or when explicitly @mentioned in a server
    if not (is_dm or is_mentioned):
        return

    content = message.content.replace(f"<@{bot.user.id}>", "").strip()
    if not content:
        content = "Hello!"

    async with message.channel.typing():
        try:
            loop = asyncio.get_event_loop()
            reply = await loop.run_in_executor(None, ask_groq, message.channel.id, content)
        except Exception as e:
            reply = f"Sorry, I ran into an error: `{e}`"

    for i in range(0, len(reply), MAX_REPLY_CHARS):
        await message.channel.send(reply[i:i + MAX_REPLY_CHARS])


if __name__ == "__main__":
    threading.Thread(target=run_keep_alive, daemon=True).start()
    bot.run(DISCORD_TOKEN)
