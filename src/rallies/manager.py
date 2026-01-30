import os
from . import console
from .agent.agent import Agent
from .agent.prompts import agent_prompt
from rich.spinner import Spinner
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
import time
import threading
from .helpers import get_timeout_message, TokenCounter, handle_command, get_api_key

class Manager:
    def __init__(self):
        self.tier = "free"
        api_key = get_api_key()
        self.agent = Agent(api_key=api_key)
        self.system_prompt = agent_prompt
        self.token_counter = TokenCounter()

    def execute_plan(
        self, planning_live, planning_content, item, prompt
    ):
        """Execute agent action with progressive timeout messages"""
        result = None
        start_time = time.time()

        # Flag to track if the action is complete
        action_complete = threading.Event()
        def run_action():
            nonlocal result
            try:
                result = self.agent.action(prompt, item["title"], item["description"])
            except Exception as e:
                result = str(e)
            finally:
                action_complete.set()

        # Start the action in a separate thread
        action_thread = threading.Thread(target=run_action)
        action_thread.start()

        # Update the display while waiting
        while not action_complete.is_set():
            elapsed_time = time.time() - start_time
            elapsed_seconds = int(elapsed_time)
            timeout_message = get_timeout_message(elapsed_time)
            
            # Add elapsed time in brackets to the timeout message
            timeout_message_with_time = f"{timeout_message} ({elapsed_seconds}s)"

            # Update the last item in planning_content with current timeout message
            planning_content[-1] = timeout_message_with_time
            planning_live.update(
                Panel("\n".join(planning_content), title="Planning", style="magenta")
            )

            # Wait 1 second before checking again to update the timer
            time.sleep(1.0)

        # Wait for the thread to complete
        action_thread.join()

        return result

    def process_prompt(self, prompt: str, conversation: list) -> str:
        # Handle commands using helpers
        if handle_command(prompt, conversation, self.agent, console):
            return ""
        
        # Check for API keys - either OpenAI or Anthropic works now
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            console.print("[red]⚠ No API key found. Please set either:[/red]")
            console.print("[dim white]  • ANTHROPIC_API_KEY for Claude (recommended)[/dim white]")
            console.print("[dim white]  • OPENAI_API_KEY for OpenAI[/dim white]")
            console.print()
            console.print("[dim white]Add to .env file: ANTHROPIC_API_KEY=sk-ant-...[/dim white]")
            console.print()
            exit()

        # Show initial planning spinner
        console.print()
        plan_spinner = Spinner(
            "dots", text="[bright_magenta]Planning...[/bright_magenta]"
        )
        with Live(plan_spinner, console=console, refresh_per_second=10):
            pass  # Initial planning display

        # Planning pane content that streams live
        planning_content = []
        with Live(console=console, refresh_per_second=10) as planning_live:
            while True:
                # Get plan from the agent
                plan = self.agent.run(conversation)
                if len(plan) == 0:
                    break

                # Add to conversation
                conversation.append({"role": "assistant", "content": str(plan)})

                # Process each plan item
                for item in plan:
                    # Add description and update planning pane immediately
                    planning_content.append(
                        f"[bright_green]●[/bright_green] [white]{item['description']}[/white]"
                    )
                    planning_live.update(
                        Panel(
                            "\n".join(planning_content),
                            title="Planning",
                            style="magenta",
                        )
                    )

                    # Add initial spinner message
                    planning_content.append("[yellow] Retrieving data... (0s)[/yellow]")
                    planning_live.update(
                        Panel(
                            "\n".join(planning_content),
                            title="Planning",
                            style="magenta",
                        )
                    )

                    # Execute with progressive timeout display
                    result = self.execute_plan(
                        planning_live, planning_content, item, prompt
                    )

                    if "[red]⚠" in result:
                        # Clear the planning pane and show only the error message
                        planning_live.stop()
                        console.print(result)
                        console.print()
                        console.print(f"[dim white]Contact us at [/dim white][link=mailto:support@rallies.ai][white]support@rallies.ai[/white][/link] [dim white]in case of any issues[/dim white]", justify="right")
                        return ""

                    # Add to conversation
                    conversation.append(
                        {
                            "role": "user",
                            "content": f"{item['title']} - {item['description']}",
                        }
                    )
                    conversation.append({"role": "user", "content": str(result), "type": "data"})

                    # Get summary and add to conversation
                    summary = self.agent.summarize(conversation)
                    conversation.append({"role": "user", "content": str(summary)})

                    # Replace spinner with summary and add new line
                    planning_content[-1] = (
                        f"[white]└─[/white] [bright_black]{summary}[/bright_black]"
                    )
                    planning_content.append("")  # Add empty line for spacing
                    planning_live.update(
                        Panel(
                            "\n".join(planning_content),
                            title="Planning",
                            style="magenta",
                        )
                    )

        # Stream the answer as markdown in answer pane
        answer_text = ""
        with Live(console=console, refresh_per_second=10) as live:
            for chunk in self.agent.answer(prompt, conversation):
                answer_text += chunk
                markdown_answer = Markdown(answer_text)
                live.update(
                    Panel(markdown_answer, title="Answer", border_style="bright_cyan")
                )
        conversation.append({"role": "assistant", "content": answer_text})

        # build footer with usage and token info
        tokens = self.token_counter.count_conversation_tokens(conversation)

        # remove large amounts of raw data to reduce token usage
        conversation = [item for item in conversation if "type" not in item or item["type"] != "data"]
        
        usage_info = ""
        if hasattr(self.agent, 'last_usage') and hasattr(self.agent, 'last_limit'):
            usage_info = f"[dim white]Usage left: [/dim white][pink]{self.agent.last_limit - self.agent.last_usage}[/pink] | "
        
        console.print(f"{usage_info}[dim white]Tokens used: [/dim white][white]{tokens:,}[/white] | [dim white]with [/dim white][magenta]♥[/magenta] [dim white]by [/dim white][link=https://rallies.ai][dim white]rallies.ai[/dim white][/link]", justify="right")

        return answer_text 
