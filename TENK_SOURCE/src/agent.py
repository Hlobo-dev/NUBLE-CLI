import asyncio
import os
import yaml
from agents import Agent, Runner, CodeInterpreterTool, WebSearchTool
from openai.types.responses import ResponseTextDeltaEvent

from src.tools import load_filing, search, list_loaded, check_available, get_stock_price
from src.export import handle_export, has_table
from src.prompts import get_system_instructions
from src.utils import extract_files_from_result, summarize_conversation, strip_citations
from src import terminal as term

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

code_interpreter = CodeInterpreterTool(tool_config={
    "type": "code_interpreter",
    "container": {"type": "auto"}
})

sec_agent = Agent(
    name="SEC Filing Analyst",
    instructions=get_system_instructions(),
    model=config["model"]["name"],
    tools=[load_filing, search, list_loaded, check_available, get_stock_price, code_interpreter, WebSearchTool()],
)


async def ask_streaming(query, conversation=None):
    input_list = conversation + [{"role": "user", "content": query}] if conversation else query
    result = Runner.run_streamed(sec_agent, input_list, max_turns=config["model"]["max_turns"])

    first_event = True
    async for event in result.stream_events():
        if first_event:
            first_event = False
            yield "first_event"
        if event.type == "run_item_stream_event":
            item = event.item
            if item.type == "tool_call_item" and hasattr(item, "raw_item"):
                raw = item.raw_item
                tool_name = getattr(raw, "name", None) or getattr(raw, "type", "code_interpreter")
                tool_args = ""
                if hasattr(raw, "arguments"):
                    tool_args = term.format_tool_args(raw.arguments)
                elif hasattr(raw, "code"):
                    tool_args = raw.code[:80] + "..." if len(raw.code) > 80 else raw.code
                term.print_tool_call(tool_name, tool_args)
            elif item.type == "tool_call_output_item":
                term.print_tool_output(item.output)
        elif event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            yield event.data.delta

    downloaded_files = extract_files_from_result(result)
    if downloaded_files:
        for f in downloaded_files:
            term.print_download_success(f)
    else:
        term.print_no_files()

    yield result


async def run_query(query, conversation=None, oneshot=False):
    if oneshot:
        query = query + "\n\n[Do not ask follow-up questions or prompt for more - just answer and stop.]"
    answer_text = ""
    result = None
    term.console.print()
    term.start_waiting()
    waiting_stopped = False
    live = None
    try:
        async for chunk in ask_streaming(query, conversation):
            if chunk == "first_event":
                term.stop_waiting()
                waiting_stopped = True
            elif isinstance(chunk, str):
                if not live:
                    live = term.create_live_panel()
                    live.__enter__()
                answer_text += chunk
                term.update_live_panel(live, strip_citations(answer_text))
            else:
                result = chunk
    finally:
        if not waiting_stopped:
            term.stop_waiting()
        if live:
            live.__exit__(None, None, None)
    return strip_citations(answer_text), result


async def run_oneshot(query):
    try:
        await run_query(query, oneshot=True)
    except Exception as e:
        if "context" in str(e).lower() or "token" in str(e).lower():
            term.print_error("Context too large")
        else:
            raise
    term.console.print()


async def run_interactive():
    conversation = None
    pending_query = None

    while True:
        if pending_query is None:
            term.print_prompt()
            try:
                query = input()
            except (EOFError, KeyboardInterrupt):
                term.print_bye()
                break
        else:
            query = pending_query
            pending_query = None

        if not query.strip():
            continue

        if query.strip().lower() in ['exit', 'quit', 'q']:
            term.print_bye()
            break

        term.console.print()

        if conversation:
            conversation = summarize_conversation(conversation)

        try:
            answer_text, result = await run_query(query, conversation)
        except Exception as e:
            if "context" in str(e).lower() or "token" in str(e).lower():
                term.print_error("Context too large, clearing conversation...")
                conversation = None
                continue
            raise

        if result:
            conversation = result.to_input_list()

        try:
            user_input = term.get_input_with_export(answer_text, has_table)
            if user_input[0] == 'export':
                filepath = handle_export(user_input[1], answer_text)
                if filepath:
                    term.print_export_success(filepath)
            elif user_input[0] == 'input' and user_input[1]:
                pending_query = user_input[1]
        except (EOFError, KeyboardInterrupt):
            term.print_bye()
            break

        term.console.print()
