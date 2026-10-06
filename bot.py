import os
import time
import random
import asyncio
import threading
from datetime import datetime, timezone
from collections import defaultdict, deque

import discord
from discord.ext import commands
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

# --- XP / leveling config --------------------------------------------------
PREFIXES = [".", ">"]      # supported command prefixes, e.g. .rank or >rank
XP_MIN, XP_MAX = 15, 25    # XP awarded per eligible message
XP_COOLDOWN_SECONDS = 60   # per-user cooldown so spamming doesn't farm XP

groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

intents = discord.Intents.default()
intents.message_content = True  # required to read message text
intents.members = True  # required to see the full member list and resolve names reliably
bot = commands.Bot(command_prefix=PREFIXES, intents=intents, help_command=None)

# per-channel short-term memory: channel_id -> deque of {"role", "content"}
history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

# --- XP storage (in-memory — resets on restart/redeploy, see README) ------
# (guild_id, user_id) -> total lifetime XP
user_xp = defaultdict(int)
# (guild_id, user_id) -> unix timestamp of last XP award, for cooldown
last_xp_time = defaultdict(float)


def calculate_level(total_xp: int):
    """Turns cumulative XP into (level, xp_into_level, xp_needed_for_level)."""
    level = 0
    xp_needed = 100
    remaining = total_xp
    while remaining >= xp_needed:
        remaining -= xp_needed
        level += 1
        xp_needed = 100 + level * 50
    return level, remaining, xp_needed


def progress_bar(current: int, total: int, length: int = 16) -> str:
    filled = int(length * current / total) if total else 0
    return "█" * filled + "░" * (length - filled)


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

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=800,
    )
    reply = response.choices[0].message.content
    history[channel_id].append({"role": "assistant", "content": reply})
    return reply


async def generate_embed(channel_id: int, question: str) -> discord.Embed:
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
        super().__init__(timeout=600)
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
    await bot.tree.sync()  # registers slash commands (can take up to ~1hr globally)
    print(f"Logged in as {bot.user} (id: {bot.user.id})")


# --- Slash commands ---------------------------------------------------------

@bot.tree.command(name="ask", description="Ask the AI assistant something")
async def ask_command(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    embed = await generate_embed(interaction.channel_id, question)
    view = ResponseView(question, interaction.channel_id)
    await interaction.followup.send(embed=embed, view=view)


@bot.tree.command(name="reset", description="Clear the AI's memory of this channel's conversation")
async def reset_command(interaction: discord.Interaction):
    history[interaction.channel_id].clear()
    await interaction.response.send_message("Conversation history cleared for this channel.", ephemeral=True)


@bot.tree.command(name="help", description="Show what this bot can do")
async def help_command(interaction: discord.Interaction):
    await interaction.response.send_message(
        "**SAIChatbot commands:**\n"
        "`/ask <question>` — ask me anything\n"
        "`/reset` — clear this channel's conversation memory\n"
        f"`{PREFIXES[0]}rank` / `{PREFIXES[1]}rank` — check your level & XP\n"
        f"`{PREFIXES[0]}leaderboard` — see the top XP earners in this server\n"
        "`/help` — show this message\n\n"
        "You can also just @mention me or DM me directly instead of using commands.",
        ephemeral=True,
    )


# --- Prefix commands (e.g. .rank or >rank) ----------------------------------

@bot.command(name="ping")
async def ping_command(ctx: commands.Context):
    await ctx.send(f"🏓 Pong! `{round(bot.latency * 1000)}ms`")


@bot.command(name="rank", aliases=["level", "xp"])
async def rank_command(ctx: commands.Context, member: discord.Member = None):
    member = member or ctx.author
    total = user_xp[(ctx.guild.id, member.id)]
    level, into_level, needed = calculate_level(total)

    embed = discord.Embed(color=discord.Color.blurple())
    embed.set_author(name=f"{member.display_name}'s Rank", icon_url=member.display_avatar.url)
    embed.add_field(name="Level", value=str(level), inline=True)
    embed.add_field(name="Total XP", value=str(total), inline=True)
    embed.add_field(
        name="Progress",
        value=f"`{progress_bar(into_level, needed)}` {into_level}/{needed} XP",
        inline=False,
    )
    await ctx.send(embed=embed)


@bot.command(name="leaderboard", aliases=["lb", "top"])
async def leaderboard_command(ctx: commands.Context):
    # Every non-bot member of the server, defaulting to 0 XP if they haven't earned any yet
    all_scores = [
        (member.id, user_xp.get((ctx.guild.id, member.id), 0))
        for member in ctx.guild.members
        if not member.bot
    ]
    all_scores.sort(key=lambda pair: pair[1], reverse=True)
    top = all_scores[:10]

    if not top:
        await ctx.send("No members found to rank.")
        return

    lines = []
    for i, (uid, xp) in enumerate(top, start=1):
        level, _, _ = calculate_level(xp)
        lines.append(f"**{i}.** <@{uid}> — Level {level} ({xp} XP)")

    embed = discord.Embed(
        title=f"🏆 {ctx.guild.name} Leaderboard",
        description="\n".join(lines),
        color=discord.Color.gold(),
    )
    await ctx.send(embed=embed)


# --- Message handling: XP awarding + AI replies + command dispatch ---------

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Award XP for any non-bot message in a server (not DMs), with a
    # per-user cooldown so spam doesn't farm XP.
    if message.guild is not None:
        key = (message.guild.id, message.author.id)
        now = time.time()
        if now - last_xp_time[key] >= XP_COOLDOWN_SECONDS:
            last_xp_time[key] = now
            before_level, _, _ = calculate_level(user_xp[key])
            user_xp[key] += random.randint(XP_MIN, XP_MAX)
            after_level, _, _ = calculate_level(user_xp[key])
            if after_level > before_level:
                await message.channel.send(
                    f"🎉 {message.author.mention} just leveled up to **Level {after_level}**!"
                )

    # Let discord.py handle prefix commands (.rank, >leaderboard, etc.)
    await bot.process_commands(message)

    # Prefix commands are handled above; don't also treat them as AI prompts
    if message.content.startswith(tuple(PREFIXES)):
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
