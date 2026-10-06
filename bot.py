import os
import time
import json
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
DEFAULT_PREFIX = "."

# --- Persistence (JSON files) -----------------------------------------
# NOTE: Render's free tier has an ephemeral filesystem — these files
# survive ordinary restarts/crashes, but are WIPED on every redeploy
# (new code push). For true cross-redeploy persistence you'd need a real
# database (e.g. a free Supabase/Postgres instance) instead of local JSON.
LEVELS_FILE = "levels.json"
CONFIG_FILE = "config.json"


def load_data(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_data(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


# levels_data[guild_id][user_id] = {"xp": int, "level": int}
levels_data = load_data(LEVELS_FILE, {})
# config_data[guild_id] = {"prefix": str, "level_channel_id": int|None}
config_data = load_data(CONFIG_FILE, {})


def get_prefix(bot_instance, message):
    if not message.guild:
        return DEFAULT_PREFIX
    return config_data.get(str(message.guild.id), {}).get("prefix", DEFAULT_PREFIX)


groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

intents = discord.Intents.default()
intents.message_content = True  # required to read message text
intents.members = True          # required to list every member for the leaderboard & resolve names
bot = commands.Bot(command_prefix=get_prefix, intents=intents, help_command=None)

# per-channel short-term memory: channel_id -> deque of {"role", "content"}
history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
# per-user XP cooldown tracker: user_id -> last award timestamp
xp_cooldowns = {}


def xp_for_level(level: int) -> int:
    """XP required to go from `level` to `level + 1`."""
    return 5 * (level ** 2) + (50 * level) + 100


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
    await bot.tree.sync()
    print(f"Logged in as {bot.user} (id: {bot.user.id})")


# --- Shared builders (used by both prefix and slash versions) --------------

async def build_rank_embed(member: discord.Member) -> discord.Embed:
    guild_id, user_id = str(member.guild.id), str(member.id)
    stats = levels_data.get(guild_id, {}).get(user_id, {"xp": 0, "level": 1})
    needed = xp_for_level(stats["level"])

    embed = discord.Embed(color=discord.Color.blurple())
    embed.set_author(name=f"{member.display_name}'s Rank", icon_url=member.display_avatar.url)
    embed.add_field(name="Level", value=str(stats["level"]), inline=True)
    embed.add_field(name="XP", value=f"{stats['xp']} / {needed}", inline=True)
    embed.add_field(
        name="Progress",
        value=f"`{progress_bar(stats['xp'], needed)}`",
        inline=False,
    )
    return embed


async def build_leaderboard_embed(guild: discord.Guild) -> discord.Embed:
    guild_levels = levels_data.get(str(guild.id), {})
    # Every non-bot member, defaulting to level 1 / 0 XP if they haven't earned any yet
    all_scores = [
        (member.id, guild_levels.get(str(member.id), {"xp": 0, "level": 1}))
        for member in guild.members
        if not member.bot
    ]
    all_scores.sort(key=lambda pair: (pair[1]["level"], pair[1]["xp"]), reverse=True)
    top = all_scores[:10]

    lines = [
        f"**{i}.** <@{uid}> — Level {stats['level']} ({stats['xp']} XP)"
        for i, (uid, stats) in enumerate(top, start=1)
    ] or ["No members found to rank."]

    return discord.Embed(
        title=f"🏆 {guild.name} Leaderboard",
        description="\n".join(lines),
        color=discord.Color.gold(),
    )


def build_help_text(guild: discord.Guild | None) -> str:
    prefix = get_prefix(bot, type("obj", (), {"guild": guild})()) if guild else DEFAULT_PREFIX
    return (
        "**SAIChatbot commands** (work as both `/slash` and prefix — "
        f"`{prefix}`):\n"
        "`ask <question>` — ask me anything\n"
        "`reset` — clear this channel's conversation memory\n"
        "`rank [@user]` — check your (or someone's) level & XP\n"
        "`leaderboard` — see every member ranked by XP\n"
        "`ping` — check the bot's latency\n"
        "`setprefix <new_prefix>` — change the command prefix (Admin only)\n"
        "`setlevelchannel #channel` — set where level-ups are announced (Admin only)\n"
        "`help` — show this message\n\n"
        "You can also just @mention me or DM me directly instead of using commands."
    )


# --- Slash commands ----------------------------------------------------------

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
    await interaction.response.send_message(build_help_text(interaction.guild), ephemeral=True)


@bot.tree.command(name="ping", description="Check the bot's latency")
async def ping_slash(interaction: discord.Interaction):
    await interaction.response.send_message(f"🏓 Pong! `{round(bot.latency * 1000)}ms`")


@bot.tree.command(name="rank", description="Check your level & XP")
async def rank_slash(interaction: discord.Interaction, member: discord.Member = None):
    member = member or interaction.user
    await interaction.response.send_message(embed=await build_rank_embed(member))


@bot.tree.command(name="leaderboard", description="See the top XP earners in this server")
async def leaderboard_slash(interaction: discord.Interaction):
    await interaction.response.send_message(embed=await build_leaderboard_embed(interaction.guild))


@bot.tree.command(name="setprefix", description="Set the command prefix for this server (Admin only)")
@discord.app_commands.checks.has_permissions(administrator=True)
async def setprefix_slash(interaction: discord.Interaction, new_prefix: str):
    guild_id = str(interaction.guild_id)
    config_data.setdefault(guild_id, {})["prefix"] = new_prefix
    save_data(CONFIG_FILE, config_data)
    await interaction.response.send_message(f"Prefix updated to `{new_prefix}`", ephemeral=True)


@bot.tree.command(name="setlevelchannel", description="Set the channel for level-up announcements (Admin only)")
@discord.app_commands.checks.has_permissions(administrator=True)
async def setlevelchannel_slash(interaction: discord.Interaction, channel: discord.TextChannel):
    guild_id = str(interaction.guild_id)
    config_data.setdefault(guild_id, {})["level_channel_id"] = channel.id
    save_data(CONFIG_FILE, config_data)
    await interaction.response.send_message(f"Level-up announcements will now be sent in {channel.mention}", ephemeral=True)


@setprefix_slash.error
@setlevelchannel_slash.error
async def admin_slash_error(interaction: discord.Interaction, error):
    if isinstance(error, discord.app_commands.MissingPermissions):
        await interaction.response.send_message("You need **Administrator** permissions to use this command.", ephemeral=True)
    else:
        raise error


# --- Prefix commands (e.g. .rank, or whatever this server's prefix is) ------

@bot.command(name="ping")
async def ping_command(ctx: commands.Context):
    await ctx.send(f"🏓 Pong! `{round(bot.latency * 1000)}ms`")


@bot.command(name="rank", aliases=["level", "xp"])
async def rank_command(ctx: commands.Context, member: discord.Member = None):
    member = member or ctx.author
    await ctx.send(embed=await build_rank_embed(member))


@bot.command(name="leaderboard", aliases=["lb", "top"])
async def leaderboard_command(ctx: commands.Context):
    await ctx.send(embed=await build_leaderboard_embed(ctx.guild))


@bot.command(name="help")
async def help_prefix(ctx: commands.Context):
    await ctx.send(build_help_text(ctx.guild))


@bot.command(name="ask")
async def ask_prefix(ctx: commands.Context, *, question: str):
    async with ctx.typing():
        embed = await generate_embed(ctx.channel.id, question)
    await ctx.send(embed=embed, view=ResponseView(question, ctx.channel.id))


@bot.command(name="reset")
async def reset_prefix(ctx: commands.Context):
    history[ctx.channel.id].clear()
    await ctx.send("Conversation history cleared for this channel.")


@bot.command(name="setprefix")
@commands.has_permissions(administrator=True)
async def setprefix_prefix(ctx: commands.Context, new_prefix: str):
    guild_id = str(ctx.guild.id)
    config_data.setdefault(guild_id, {})["prefix"] = new_prefix
    save_data(CONFIG_FILE, config_data)
    await ctx.send(f"Prefix updated to `{new_prefix}`")


@bot.command(name="setlevelchannel")
@commands.has_permissions(administrator=True)
async def setlevelchannel_prefix(ctx: commands.Context, channel: discord.TextChannel):
    guild_id = str(ctx.guild.id)
    config_data.setdefault(guild_id, {})["level_channel_id"] = channel.id
    save_data(CONFIG_FILE, config_data)
    await ctx.send(f"Level-up announcements will now be sent in {channel.mention}")


@setprefix_prefix.error
@setlevelchannel_prefix.error
async def admin_prefix_error(ctx: commands.Context, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("You need **Administrator** permissions to use this command.")
    else:
        raise error


# --- Message handling: XP awarding + AI replies + command dispatch ---------

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # --- XP / leveling ---
    if message.guild:
        guild_id, user_id = str(message.guild.id), str(message.author.id)
        now = time.time()

        if user_id not in xp_cooldowns or (now - xp_cooldowns[user_id]) > 60:
            xp_cooldowns[user_id] = now
            levels_data.setdefault(guild_id, {}).setdefault(user_id, {"xp": 0, "level": 1})
            stats = levels_data[guild_id][user_id]
            stats["xp"] += 15
            needed = xp_for_level(stats["level"])

            if stats["xp"] >= needed:
                stats["level"] += 1
                target_channel = message.channel
                lvl_chan_id = config_data.get(guild_id, {}).get("level_channel_id")
                if lvl_chan_id:
                    configured_chan = bot.get_channel(lvl_chan_id)
                    if configured_chan:
                        target_channel = configured_chan
                await target_channel.send(
                    f"🎉 Congratulations {message.author.mention}! You've reached **Level {stats['level']}**!"
                )
            save_data(LEVELS_FILE, levels_data)

    # Let discord.py handle prefix commands (dynamic per-guild prefix)
    await bot.process_commands(message)

    # Don't also treat a prefix command as an AI prompt
    current_prefix = get_prefix(bot, message)
    if message.content.startswith(current_prefix):
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
