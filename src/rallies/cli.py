import sys
from rich.text import Text
from rallies.manager import Manager
from rallies import console

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

def interactive_shell():
    display_application_banner()
    
    # Tips section for user guidance
    console.print("\n[dim white]Tips for getting started:[/dim white]")
    console.print("[white]1. Ask questions about stocks, analyze trends, or get market insights.[/white]")
    console.print("[white]2. Be specific for the best results.[/white]")
    console.print("[white]3. Type /help for more information.[/white]\n")
    
    # Use free agent by default
    selected_agent = Manager()
    
    print("\nType your queries below. Press Ctrl+C to exit.\n")
    messages = []
    try:
        while True:
            # Display prompt for user input
            console.print("[bright_cyan]> [/bright_cyan]", end="")
            user_input_text = input()
            
            if user_input_text.strip():
                messages.append({"role": "user", "content": user_input_text})
                selected_agent.process_prompt(user_input_text, messages)
            else:
                console.print("[yellow]Please enter a query.[/yellow]\n")
                
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)

def main():
    interactive_shell()

if __name__ == '__main__':
    main()