import sys
import asyncio
from src.db import show_ready_message
from src.agent import run_interactive, run_oneshot

def main():
    has_query = len(sys.argv) > 1
    show_ready_message(has_query)
    if has_query:
        query = " ".join(sys.argv[1:])
        asyncio.run(run_oneshot(query))
    else:
        asyncio.run(run_interactive())

if __name__ == "__main__":
    main()
