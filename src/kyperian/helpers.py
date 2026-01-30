import random
import tiktoken
import json
import os
import requests
from pathlib import Path
from .llm import LLM
from rich.markdown import Markdown

class TokenCounter:
    def __init__(self, model="gpt-4o"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("o200k_base")
    
    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def count_conversation_tokens(self, conversation: list) -> int:
        total_tokens = 0
        for message in conversation:
            if isinstance(message, dict) and "content" in message:
                total_tokens += self.count_tokens(message["content"])
            elif isinstance(message, str):
                total_tokens += self.count_tokens(message)
        return total_tokens
    
    def format_token_count(self, token_count: int) -> str:
        if token_count >= 1000:
            return f"{token_count / 1000:.1f}k tokens"
        return f"{token_count} tokens"

def get_timeout_message(elapsed_time):
    """Get appropriate message based on elapsed time"""
    
    if elapsed_time < 10:
        return "[yellow]  Retrieving data...[/yellow]"
    else:
         # After retrieving data, randomly pick from these messages every 10 seconds
        messages = [
                "[yellow]  Cogitating...[/yellow]",
                "[yellow]  Deep dive...[/yellow]",
                "[yellow]  Percolating...[/yellow]",
                "[yellow]  Synthesizing...[/yellow]",
                "[yellow]  Triangulating...[/yellow]",
                "[yellow]  Crystallizing...[/yellow]",
                "[yellow]  Distilling...[/yellow]",
                "[yellow]  Calibrating...[/yellow]",
                "[yellow]  Optimizing...[/yellow]",
                "[yellow]  Finalizing...[/yellow]",
                "[yellow]  Polishing...[/yellow]",
                "[yellow]  Contemplating...[/yellow]",
                "[yellow]  Deliberating...[/yellow]",
                "[yellow]  Ruminating...[/yellow]",
                "[yellow]  Pondering...[/yellow]",
                "[yellow]  Mulling over...[/yellow]",
                "[yellow]  Reflecting...[/yellow]",
                "[yellow]  Meditating...[/yellow]",
                "[yellow]  Concentrating...[/yellow]",
                "[yellow]  Focusing...[/yellow]",
                "[yellow]  Absorbing...[/yellow]",
                "[yellow]  Digesting...[/yellow]",
                "[yellow]  Assimilating...[/yellow]",
                "[yellow]  Integrating...[/yellow]",
                "[yellow]  Harmonizing...[/yellow]",
                "[yellow]  Balancing...[/yellow]",
                "[yellow]  Aligning...[/yellow]",
                "[yellow]  Orchestrating...[/yellow]",
                "[yellow]  Weaving...[/yellow]",
                "[yellow]  Crafting...[/yellow]",
                "[yellow]  Sculpting...[/yellow]",
                "[yellow]  Refining...[/yellow]",
        ]
        # Change message every 10 seconds after initial retrieval
        message_index = int((elapsed_time - 10) // 10) % len(messages)
        return messages[message_index]

def show_help(console):
    console.print("\n[bright_cyan]Available Commands:[/bright_cyan]")
    console.print("  [white]/key API_KEY[/white]              Set up your API key")
    console.print("  [white]/feed[/white]               Show recent high-scoring questions from the community")
    console.print("  [white]/clear[/white]              Clear conversation history and free up context")
    console.print("  [white]/compact[/white]            Clear conversation history but keep a summary in context.")
    console.print("                      Optional: /compact [instructions for summarization]")
    console.print("  [white]/exit (quit)[/white]        Exit the REPL")
    console.print("  [white]/help[/white]               Show help and available commands")
    console.print()


def handle_help_command(console):
    show_help(console)
    return True


def handle_clear_command(conversation, console):
    conversation.clear()
    console.print("[green]Conversation history cleared.[/green]")
    return True


def handle_compact_command(prompt, conversation, agent, console):
    if len(conversation) == 0:
        console.print("[red]No conversation history to compact.[/red]")
        return True
    
    console.print("Let us compact the conversation to reduce tokens")
    conversation = agent.compact(conversation)
    tokens = TokenCounter().count_conversation_tokens(conversation)
    console.print(f"[green]✓ Conversation condensed to {tokens} tokens. You can continue asking more questions now.[/green]")
    console.print()
    return True 


def handle_exit_command(console):
    console.print("\nGoodbye!")
    import sys
    sys.exit(0)


def get_config_dir():
    """Get or create the config directory"""
    home = Path.home()
    config_dir = home / ".kyperian"
    config_dir.mkdir(exist_ok=True)
    return config_dir


def get_config_file():
    """Get the config file path"""
    return get_config_dir() / "config.json"


def load_config():
    """Load configuration from file"""
    config_file = get_config_file()
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_config(config):
    """Save configuration to file"""
    config_file = get_config_file()
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f)
        return True
    except IOError:
        return False


def get_api_key():
    """Get the stored API key"""
    config = load_config()
    return config.get("api_key")


def set_api_key(api_key):
    """Set and save the API key"""
    config = load_config()
    config["api_key"] = api_key
    return save_config(config)


def handle_key_command(prompt, agent, console):
    """Handle the /key command"""
    parts = prompt.strip().split(None, 1)
    if len(parts) < 2:
        console.print("[red]Usage: /key API_KEY[/red]")
        return True
    
    api_key = parts[1]
    if set_api_key(api_key):
        # Update the current agent's API key immediately
        agent.api_key = api_key
        console.print("[green]API key saved and activated.[/green]")
    else:
        console.print("[red]Failed to save API key.[/red]")
    return True


def handle_feed_command(console):
    """Handle the /feed command - show example queries and tips"""
    console.print(f"\n[bright_cyan]KYPERIAN - Example Queries:[/bright_cyan]\n")
    
    examples = [
        ("Stock Analysis", [
            "What happened to AAPL stock today?",
            "Analyze NVDA's technical indicators",
            "Compare MSFT and GOOGL performance this month",
        ]),
        ("SEC Filings", [
            "What are the key risks in Tesla's latest 10-K?",
            "Summarize Apple's revenue segments from their annual report",
            "Find any mention of AI investments in Microsoft's 10-Q",
        ]),
        ("Market Overview", [
            "Which tech stocks are down the most today?",
            "Show me stocks with unusual volume today",
            "What's moving the market today?",
        ]),
        ("Options & Trading", [
            "What's the options flow on SPY today?",
            "Find stocks with high implied volatility",
            "Analyze gamma exposure on major indices",
        ]),
    ]
    
    for category, queries in examples:
        console.print(f"[yellow]{category}:[/yellow]")
        for query in queries:
            console.print(f"  • {query}")
        console.print()
    
    console.print("[dim]Try any of these queries or ask your own question![/dim]")
    return True


def handle_command(prompt, conversation, agent, console):
    if prompt.strip() == "/help":
        return handle_help_command(console)
    
    if prompt.strip() == "/feed":
        return handle_feed_command(console)
    
    if prompt.strip() == "/clear":
        return handle_clear_command(conversation, console)
    
    if prompt.strip().startswith("/compact"):
        return handle_compact_command(prompt, conversation, agent, console)
    
    if prompt.strip().startswith("/key"):
        return handle_key_command(prompt, agent, console)
    
    if prompt.strip() in ["/exit", "/quit"]:
        handle_exit_command(console)

    
    return False