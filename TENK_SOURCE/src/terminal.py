import json
import sys
import tty
import termios
import time
import asyncio
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live

console = Console()

class WaitingDots:
    def __init__(self):
        self.task = None
        self.stopped = False

    async def _animate(self):
        dots = 0
        while not self.stopped:
            dots = (dots % 3) + 1
            sys.stdout.write(f"\r{'.' * dots}{' ' * (3 - dots)}")
            sys.stdout.flush()
            await asyncio.sleep(0.4)

    def start(self):
        self.stopped = False
        self.task = asyncio.create_task(self._animate())

    def stop(self):
        self.stopped = True
        if self.task:
            self.task.cancel()
            self.task = None
        sys.stdout.write("\r   \r")
        sys.stdout.flush()

waiting_dots = WaitingDots()

def display_banner():
    console.print("""
[bold bright_green]
▀█▀ █▀▀ █▄ █ █▄▀
 █  ██▄ █ ▀█ █ █
[/]
[dim]Talk to SEC filings with AI[/]
[dim]--------------------------------[/]
[dim]Built with ❤️ by [link=https://rallies.ai]Rallies.ai[/link][/]
[dim]────────────────────────────────[/]
""")

def animate_loading(done_flag):
    dots = 0
    while not done_flag():
        dots = (dots % 3) + 1
        sys.stdout.write(f"\r\033[37m⏺\033[0m\033[2m Loading up some data{'.' * dots}{' ' * (3 - dots)}\033[0m")
        sys.stdout.flush()
        time.sleep(0.4)

def start_waiting():
    waiting_dots.start()

def stop_waiting():
    waiting_dots.stop()

def show_ready():
    sys.stdout.write("\r\033[32m✓\033[0m Ready, ask anything                        \n")
    sys.stdout.flush()

def show_working():
    sys.stdout.write("\r\033[32m✓\033[0m Ready, let me get to work                  \n")
    sys.stdout.flush()

def format_tool_args(args_json: str) -> str:
    try:
        args = json.loads(args_json)
        return " ".join(f"{k}={v}" for k, v in args.items())
    except:
        return args_json[:50]

def print_tool_call(tool_name: str, tool_args: str):
    console.print(f"\n[bright_cyan]⏺[/bright_cyan] [bold]{tool_name}[/bold]")
    if tool_args:
        console.print(f"  [dim]⎿[/dim] [dim]{tool_args}[/dim]")

def print_tool_output(output: str):
    output = str(output)[:100].replace('\n', ' ')
    console.print(f"  [dim]⎿[/dim] [green dim]{output}{'...' if len(str(output)) > 100 else ''}[/green dim]")

def print_download_success(filepath: str):
    console.print(f"\n[green]✓[/green] Downloaded: [white]{filepath}[/white]")

def print_no_files():
    console.print(f"  [dim yellow]No files found in response[/dim yellow]")

def print_export_success(filepath: str):
    console.print(f"\n[green]✓[/green] Exported to [white]{filepath}[/white]")

def print_error(msg: str):
    console.print(f"[red]{msg}[/red]")

def print_dim(msg: str):
    console.print(f"[dim]{msg}[/dim]")

def print_dim_yellow(msg: str):
    console.print(f"[dim yellow]{msg}[/dim yellow]")

def print_prompt():
    console.print("[bright_cyan]> [/bright_cyan]", end="")

def print_bye():
    console.print("\n[dim]Bye![/dim]")

def create_live_panel():
    return Live(console=console, refresh_per_second=10)

def update_live_panel(live, text: str):
    live.update(Panel(Markdown(text), border_style="bright_cyan"))

def get_input_with_export(answer_text: str, has_table_fn) -> str:
    console.print()
    show_excel = has_table_fn(answer_text)
    if show_excel:
        console.print("[dim]Export:[/dim] [bright_cyan][1][/bright_cyan] PDF  [bright_cyan][2][/bright_cyan] DOCX  [bright_cyan][3][/bright_cyan] Excel", justify="right")
    else:
        console.print("[dim]Export:[/dim] [bright_cyan][1][/bright_cyan] PDF  [bright_cyan][2][/bright_cyan] DOCX", justify="right")
    console.print("[bright_cyan]> [/bright_cyan]", end="")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        first_char = sys.stdin.read(1)

        if first_char == '1':
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ('export', 'pdf')
        elif first_char == '2':
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ('export', 'docx')
        elif first_char == '3' and show_excel:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ('export', 'xlsx')
        elif first_char == '\x03':
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            raise KeyboardInterrupt
        elif first_char == '\x04':
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            raise EOFError
        else:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            sys.stdout.write(first_char)
            sys.stdout.flush()
            rest = input()
            return ('input', first_char + rest)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
