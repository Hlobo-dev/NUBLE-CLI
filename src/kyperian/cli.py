import sys
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from kyperian.manager import Manager
from kyperian import console

# Check if services are available for status display
try:
    from kyperian.services import get_services
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False

def display_application_banner():
    banner_text = """
██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗ ██╗ █████╗ ███╗   ██╗
██║ ██╔╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗██║██╔══██╗████╗  ██║
█████╔╝  ╚████╔╝ ██████╔╝█████╗  ██████╔╝██║███████║██╔██╗ ██║
██╔═██╗   ╚██╔╝  ██╔═══╝ ██╔══╝  ██╔══██╗██║██╔══██║██║╚██╗██║
██║  ██╗   ██║   ██║     ███████╗██║  ██║██║██║  ██║██║ ╚████║
╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝
"""
    
    # Create gradient effect similar to Gemini CLI
    lines = banner_text.strip().split('\n')
    gradient_colors = ['bright_blue', 'blue', 'cyan', 'bright_cyan', 'magenta', 'bright_magenta']
    
    styled_lines = []
    for i, line in enumerate(lines):
        color = gradient_colors[i % len(gradient_colors)]
        styled_lines.append(Text(line, style=f"bold {color}"))
    
    # Combine all lines
    full_banner = Text()
    full_banner.append("\n\n")
    for line in styled_lines:
        full_banner.append(line)
        full_banner.append('\n')
    
    # Add subtitle with gradient
    subtitle = Text("Institutional-Grade AI Investment Research", style="bold bright_magenta")
    full_banner.append('\n')
    full_banner.append(subtitle)
    
    # Print banner without border, left-aligned like Gemini CLI
    console.print(full_banner)


def show_system_status():
    """Display system status and capabilities."""
    if not SERVICES_AVAILABLE:
        console.print("[yellow]Services not fully initialized[/yellow]")
        return
    
    try:
        services = get_services()
        status = services.get_system_status()
        
        # Build status table
        table = Table(title="🔧 KYPERIAN System Status", border_style="bright_cyan")
        table.add_column("Component", style="white")
        table.add_column("Status", style="white")
        table.add_column("Details", style="dim")
        
        components = status.get('components', {})
        
        for name, info in components.items():
            status_icon = "✅" if info.get('status') == 'available' else "❌" if info.get('status') == 'error' else "⚪"
            status_text = f"{status_icon} {info.get('status', 'unknown')}"
            details = info.get('details', '-')
            table.add_row(name.replace('_', ' ').title(), status_text, str(details)[:40])
        
        console.print(table)
        console.print()
        
        # Show capabilities
        console.print("[bold bright_cyan]Available Commands:[/bold bright_cyan]")
        console.print("  [white]AAPL[/white]          - Quick quote")
        console.print("  [white]predict TSLA[/white]  - ML prediction")
        console.print("  [white]RSI for AMD[/white]   - Technical analysis")
        console.print("  [white]NVDA 10-K[/white]     - SEC filings")
        console.print("  [white]/status[/white]       - System status")
        console.print("  [white]/help[/white]         - Help")
        console.print()
        
    except Exception as e:
        console.print(f"[red]Could not get system status: {e}[/red]")


def handle_quick_command(cmd: str, manager: Manager) -> bool:
    """
    Handle quick commands that start with /
    Returns True if command was handled.
    """
    cmd_lower = cmd.lower().strip()
    
    if cmd_lower == '/status':
        show_system_status()
        return True
    
    if cmd_lower in ('/quit', '/exit', '/q'):
        console.print("\n[bright_cyan]Goodbye![/bright_cyan]")
        sys.exit(0)
    
    if cmd_lower == '/version':
        console.print("[bright_cyan]KYPERIAN[/bright_cyan] v2.0.0 - Unified Edition")
        console.print("[dim]ML Models: MLP, LSTM, Transformer, N-BEATS, Ensemble[/dim]")
        console.print("[dim]Data Sources: Polygon.io, SEC EDGAR[/dim]")
        return True
    
    if cmd_lower == '/clear':
        console.clear()
        display_application_banner()
        return True
    
    return False  # Not a quick command

def interactive_shell():
    display_application_banner()
    
    # Tips section for user guidance
    console.print("\n[dim white]Tips for getting started:[/dim white]")
    console.print("[white]1. Quick: [/white][bright_cyan]AAPL[/bright_cyan][white] → instant quote[/white]")
    console.print("[white]2. ML: [/white][bright_cyan]predict TSLA[/bright_cyan][white] → AI prediction[/white]")
    console.print("[white]3. Technical: [/white][bright_cyan]RSI for AMD[/bright_cyan][white] → indicators[/white]")
    console.print("[white]4. Research: [/white][bright_cyan]Should I buy NVDA?[/bright_cyan][white] → deep analysis[/white]")
    console.print("[white]5. Commands: [/white][bright_cyan]/status[/bright_cyan][white], [/white][bright_cyan]/help[/bright_cyan][white], [/white][bright_cyan]/clear[/bright_cyan]\n")
    
    # Use free agent by default
    selected_agent = Manager()
    
    print("\nType your queries below. Press Ctrl+C to exit.\n")
    messages = []
    try:
        while True:
            # Display prompt for user input
            console.print("[bright_cyan]> [/bright_cyan]", end="")
            user_input_text = input()
            
            if not user_input_text.strip():
                console.print("[yellow]Please enter a query.[/yellow]\n")
                continue
            
            # Handle quick commands first
            if user_input_text.startswith('/'):
                if handle_quick_command(user_input_text, selected_agent):
                    continue
            
            # Process normal query
            messages.append({"role": "user", "content": user_input_text})
            selected_agent.process_prompt(user_input_text, messages)
                
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)

def main():
    interactive_shell()

if __name__ == '__main__':
    main()