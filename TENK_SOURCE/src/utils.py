import os
import re
import json
import yaml
from openai import OpenAI

from src import terminal as term

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

openai_client = OpenAI()

exports_dir = config["exports"]["dir"]
if not os.path.isabs(exports_dir):
    exports_dir = os.path.join(os.getcwd(), exports_dir)
os.makedirs(exports_dir, exist_ok=True)

CITATION_PATTERN = re.compile(r'\s*(?:cite)?turn\d+search\d+(?:turn\d+search\d+)*\s*')

def strip_citations(text: str) -> str:
    return CITATION_PATTERN.sub('', text)


def download_container_file(container_id: str, file_id: str, filename: str) -> str:
    filepath = os.path.join(exports_dir, filename)
    try:
        resp = openai_client.containers.files.content.retrieve(file_id=file_id, container_id=container_id)
        with open(filepath, "wb") as f:
            f.write(resp.content)
        return filepath
    except Exception as e:
        term.print_error(f"Download error: {e}")
        return None


def extract_files_from_result(result) -> list[str]:
    downloaded = []
    container_ids = set()

    for item in result.new_items:
        raw = getattr(item, "raw_item", None)
        if not raw:
            continue
        if getattr(raw, "type", "") == "code_interpreter_call":
            cid = getattr(raw, "container_id", None)
            if cid:
                container_ids.add(cid)

    for container_id in container_ids:
        try:
            files = openai_client.containers.files.list(container_id)
            for f in files:
                if f.path and f.path.endswith((".xlsx", ".csv", ".png", ".pdf")):
                    filename = os.path.basename(f.path)
                    filepath = download_container_file(container_id, f.id, filename)
                    if filepath:
                        downloaded.append(filepath)
        except Exception as e:
            term.print_error(f"Container list error: {e}")

    return downloaded


def estimate_tokens(conversation: list) -> int:
    if not conversation:
        return 0
    text = json.dumps(conversation)
    return len(text) // 4


def summarize_conversation(conversation: list) -> list:
    if not conversation:
        return conversation

    tokens = estimate_tokens(conversation)
    if tokens <= 80000:
        return conversation

    term.print_dim_yellow(f"Summarizing conversation ({tokens} tokens)...")

    conv_text = ""
    for item in conversation:
        if isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
        else:
            role = getattr(item, "role", "")
            content = getattr(item, "content", "")

        if isinstance(content, list):
            content = " ".join(str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content)

        if content and role in ["user", "assistant"]:
            conv_text += f"{role.upper()}: {content[:2000]}\n\n"

    summary_prompt = f"""Summarize this conversation between a user and an SEC filings analyst.
Preserve ALL key information:
- Tickers and companies discussed
- Specific financial data points and numbers found
- Key findings and insights
- Questions asked and answered
- Any files created or exports made

Be comprehensive but concise. This summary will replace the conversation history.

CONVERSATION:
{conv_text[:100000]}"""

    try:
        response = openai_client.chat.completions.create(
            model=config["model"]["name"],
            messages=[{"role": "user", "content": summary_prompt}],
            max_completion_tokens=4000
        )
        summary = response.choices[0].message.content
        term.print_dim_yellow(f"Compressed to ~{len(summary)//4} tokens")
        return [{"role": "assistant", "content": f"[Previous conversation summary]\n{summary}"}]
    except Exception as e:
        term.print_error(f"Summarization failed: {e}")
        return conversation[-20:]
