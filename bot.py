import os
import time
import asyncio
import threading
from datetime import datetime, timezone
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
EMBED_DESC_LIMIT = 4096   # Discord's embed description limit

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


async def generate_embed(channel_id: int, question: str) -> discord.Embed:
    """Calls Groq, times it, and packages the reply into an embed styled
    like: description = reply, footer = 'Groq • <model> • <ping>ms',
    with Discord auto-appending ' • Today at HH:MM' from embed.timestamp."""
    loop = asyncio.get_event_loop()
    start = time.monotonic()
    try:
        reply = await loop.run_in_executor(None, ask_groq, channel_id, question)
    except Exception as e:
        reply = f"Sorry, I ran into an error: `{e}`"
    ping_ms = int((time.monotonic() - start) * 1000)

    embed = discord.Embed(description=reply[:EMBED_DESC_LIMIT], color=discord.Color.blurple())
    embed.set_footer(text=f"Groq • {MODEL} • {ping_ms}ms")
    embed.timestamp = datetime.now(timezone.utc)
    return embed


class ResponseView(discord.ui.View):
    """Regenerate / thumbs up / thumbs down buttons attached to a reply."""

    def __init__(self, question: str, channel_id: int):
        super().__init__(timeout=600)  # buttons stop responding after 10 min idle
        self.question = question
        self.channel_id = channel_id

    @discord.ui.button(label="Regenerate", style=discord.ButtonStyle.primary, emoji="🔄")
    async def regenerate(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        new_embed = await generate_embed(self.channel_id, self.question)
        await interaction.edit_original_response(embed=new_embed, view=self)

    @discord.ui.button(style=discord.ButtonStyle.success, emoji="👍")
    async def thumbs_up(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("Thanks for the feedback! 👍", ephemeral=True)

    @discord.ui.button(style=discord.ButtonStyle.danger, emoji="👎")
    async def thumbs_down(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("Thanks — noted. 👎", ephemeral=True)


@bot.event
async def on_ready():
    await tree.sync()  # registers slash commands with Discord (can take up to ~1hr globally)
    print(f"Logged in as {bot.user} (id: {bot.user.id})")


# --- Slash commands -------------------------------------------------------

@tree.command(name="ask", description="Ask the AI assistant something")
async def ask_command(interaction: discord.Interaction, question: str):
    await interaction.response.defer()  # shows "thinking..." while Groq responds
    embed = await generate_embed(interaction.channel_id, question)
    view = ResponseView(question, interaction.channel_id)
    await interaction.followup.send(embed=embed, view=view)


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

    if not (is_dm or is_mentioned):
        return

    content = message.content.replace(f"<@{bot.user.id}>", "").strip()
    if not content:
        content = "Hello!"

    async with message.channel.typing():
        embed = await generate_embed(message.channel.id, content)

    view = ResponseView(content, message.channel.id)
    await message.channel.send(embed=embed, view=view)


if __name__ == "__main__":
    threading.Thread(target=run_keep_alive, daemon=True).start()
    bot.run(DISCORD_TOKEN)
