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
DEFAULT_PREFIX = "!"

# --- Persistence (JSON Files) -------------------------------------------
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

# Data structure: levels[guild_id][user_id] = {"xp": 0, "level": 1}
levels_data = load_data(LEVELS_FILE, {})
# Data structure: config[guild_id] = {"prefix": "!", "level_channel_id": null}
config_data = load_data(CONFIG_FILE, {})

def get_prefix(bot_instance, message):
    if not message.guild:
        return DEFAULT_PREFIX
    guild_id = str(message.guild.id)
    return config_data.get(guild_id, {}).get("prefix", DEFAULT_PREFIX)

# --- Client Setup --------------------------------------------------------
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

intents = discord.Intents.default()
intents.message_content = True  # required to read message text
bot = commands.Bot(command_prefix=get_prefix, intents=intents)

# Short-term memory: channel_id -> deque of {"role", "content"}
history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
# Cooldown tracker for XP gain (user_id -> last_xp_time)
xp_cooldowns = {}

# --- Keep-alive web server ----------------------------------------------
keep_alive_app = Flask(__name__)

@keep_alive_app.route("/")
def home():
    return "Bot is alive!"

def run_keep_alive():
    port = int(os.environ.get("PORT", 8080))
    keep_alive_app.run(host="0.0.0.0", port=port)

# --- Helper Functions ----------------------------------------------------
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

def xp_for_level(level: int) -> int:
    """XP required to reach the next level."""
    return 5 * (level ** 2) + (50 * level) + 100

# --- Views ---------------------------------------------------------------
class ResponseView(discord.ui.View):
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

# --- Event Handlers ------------------------------------------------------
@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"Logged in as {bot.user} (id: {bot.user.id})")

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # --- Leveling Logic ---
    if message.guild:
        guild_id = str(message.guild.id)
        user_id = str(message.author.id)
        now = time.time()

        # 60-second cooldown on XP per user to prevent spamming
        if user_id not in xp_cooldowns or (now - xp_cooldowns[user_id]) > 60:
            xp_cooldowns[user_id] = now
            if guild_id not in levels_data:
                levels_data[guild_id] = {}
            if user_id not in levels_data[guild_id]:
                levels_data[guild_id][user_id] = {"xp": 0, "level": 1}

            user_stats = levels_data[guild_id][user_id]
            user_stats["xp"] += 15  # XP added per message
            needed_xp = xp_for_level(user_stats["level"])

            if user_stats["xp"] >= needed_xp:
                user_stats["level"] += 1
                save_data(LEVELS_FILE, levels_data)

                # Find level announcement channel
                target_channel = message.channel
                lvl_chan_id = config_data.get(guild_id, {}).get("level_channel_id")
                if lvl_chan_id:
                    configured_chan = bot.get_channel(lvl_chan_id)
                    if configured_chan:
                        target_channel = configured_chan

                await target_channel.send(
                    f"🎉 Congratulations {message.author.mention}! You've reached **Level {user_stats['level']}**!"
                )
            else:
                save_data(LEVELS_FILE, levels_data)

    # Process Prefix Commands
    await bot.process_commands(message)

    # --- AI Chat Processing (DMs & Mentions) ---
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user in message.mentions

    if is_dm or is_mentioned:
        # Ignore if message starts with prefix to prevent interference with prefix commands
        current_prefix = get_prefix(bot, message)
        if message.content.startswith(current_prefix):
            return

        content = message.content.replace(f"<@{bot.user.id}>", "").strip()
        if not content:
            content = "Hello!"

        async with message.channel.typing():
            embed = await generate_embed(message.channel.id, content)

        view = ResponseView(content, message.channel.id)
        await message.channel.send(embed=embed, view=view)

# --- Slash Commands ------------------------------------------------------
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

@bot.tree.command(name="rank", description="Check your current level and XP")
async def rank_command(interaction: discord.Interaction, target: discord.Member = None):
    target_user = target or interaction.user
    guild_id = str(interaction.guild_id)
    user_id = str(target_user.id)

    stats = levels_data.get(guild_id, {}).get(user_id, {"xp": 0, "level": 1})
    needed_xp = xp_for_level(stats["level"])

    embed = discord.Embed(title=f"📊 Rank for {target_user.display_name}", color=discord.Color.gold())
    embed.set_thumbnail(url=target_user.display_avatar.url)
    embed.add_field(name="Level", value=str(stats["level"]), inline=True)
    embed.add_field(name="XP", value=f"{stats['xp']} / {needed_xp}", inline=True)
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="leaderboard", description="View the server's top leveled members")
async def leaderboard_command(interaction: discord.Interaction):
    guild_id = str(interaction.guild_id)
    guild_levels = levels_data.get(guild_id, {})

    if not guild_levels:
        await interaction.response.send_message("No activity recorded yet!", ephemeral=True)
        return

    sorted_users = sorted(guild_levels.items(), key=lambda x: (x[1]["level"], x[1]["xp"]), reverse=True)[:10]

    description = ""
    for i, (u_id, stats) in enumerate(sorted_users, 1):
        member = interaction.guild.get_member(int(u_id))
        name = member.display_name if member else f"User {u_id}"
        description += f"**#{i} {name}** — Level {stats['level']} ({stats['xp']} XP)\n"

    embed = discord.Embed(title=f"🏆 {interaction.guild.name} Leaderboard", description=description, color=discord.Color.gold())
    await interaction.response.send_message(embed=embed)

# --- Admin Slash Commands ------------------------------------------------
@bot.tree.command(name="setprefix", description="Set the prefix for traditional commands (Admin only)")
@discord.app_commands.checks.has_permissions(administrator=True)
async def setprefix_command(interaction: discord.Interaction, new_prefix: str):
    guild_id = str(interaction.guild_id)
    if guild_id not in config_data:
        config_data[guild_id] = {}
    
    config_data[guild_id]["prefix"] = new_prefix
    save_data(CONFIG_FILE, config_data)
    await interaction.response.send_message(f"Prefix updated to `{new_prefix}`", ephemeral=True)

@bot.tree.command(name="setlevelchannel", description="Set channel where level-up messages are sent (Admin only)")
@discord.app_commands.checks.has_permissions(administrator=True)
async def setlevelchannel_command(interaction: discord.Interaction, channel: discord.TextChannel):
    guild_id = str(interaction.guild_id)
    if guild_id not in config_data:
        config_data[guild_id] = {}

    config_data[guild_id]["level_channel_id"] = channel.id
    save_data(CONFIG_FILE, config_data)
    await interaction.response.send_message(f"Level-up announcements will now be sent in {channel.mention}", ephemeral=True)

# --- Prefix Commands -----------------------------------------------------
@bot.command(name="setprefix")
@commands.has_permissions(administrator=True)
async def prefix_setprefix(ctx, new_prefix: str):
    guild_id = str(ctx.guild.id)
    if guild_id not in config_data:
        config_data[guild_id] = {}

    config_data[guild_id]["prefix"] = new_prefix
    save_data(CONFIG_FILE, config_data)
    await ctx.send(f"Prefix updated to `{new_prefix}`")

@bot.command(name="setlevelchannel")
@commands.has_permissions(administrator=True)
async def prefix_setlevelchannel(ctx, channel: discord.TextChannel):
    guild_id = str(ctx.guild.id)
    if guild_id not in config_data:
        config_data[guild_id] = {}

    config_data[guild_id]["level_channel_id"] = channel.id
    save_data(CONFIG_FILE, config_data)
    await ctx.send(f"Level-up announcements directed to {channel.mention}")

# Permissions Error Handler
@setprefix_command.error
@setlevelchannel_command.error
@prefix_setprefix.error
@prefix_setlevelchannel.error
async def admin_command_error(ctx_or_interaction, error):
    msg = "You need **Administrator** permissions to use this command."
    if isinstance(ctx_or_interaction, discord.Interaction):
        await ctx_or_interaction.response.send_message(msg, ephemeral=True)
    else:
        await ctx_or_interaction.send(msg)

if __name__ == "__main__":
    threading.Thread(target=run_keep_alive, daemon=True).start()
    bot.run(DISCORD_TOKEN)
