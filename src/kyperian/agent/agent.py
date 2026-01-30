import os
import json
import subprocess
import tempfile
import requests
from .prompts import agent_prompt, answer_prompt, summary_prompt, compact_prompt, action_prompt
from ..llm import LLM, token_tracker


class Agent:
    def __init__(self, api_key=None):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.api_key = api_key
        self.llm = LLM()
        
    @property
    def token_tracker(self):
        """Access to the global token tracker for usage stats."""
        return token_tracker
        
    def parse_messages(self, messages: list) -> list:
         parsed_messages = []
         for message in messages:
             if isinstance(message, dict) and "role" in message and "content" in message:
                 parsed_messages.append({
                     "role": message["role"],
                     "content": message["content"]
                 })
         return parsed_messages

    def run(self, messages: str) -> str:
        message = []
        message.append({"role": "developer", "content": agent_prompt})
        message.extend(self.parse_messages(messages))
        response = self.llm.prompt(message, requires_json = True)
        return response
    
    def action(self, question, title, description):
        """
        Execute a research action using Claude directly with web search.
        Uses Polygon API for market data and Claude for analysis.
        """
        try:
            # Build a research prompt for Claude
            action_request = action_prompt.format(
                question=question,
                title=title,
                description=description
            )
            
            messages = [
                {"role": "developer", "content": action_request},
                {"role": "user", "content": f"Research task: {title}\n\nDetails: {description}\n\nOriginal question: {question}"}
            ]
            
            # Try to get real market data from Polygon if we have a ticker
            market_data = self._get_market_data(question, title, description)
            if market_data:
                messages.append({
                    "role": "user", 
                    "content": f"Here is real-time market data to help with your analysis:\n\n{market_data}"
                })
            
            # Get response from Claude
            response = self.llm.prompt(messages)
            return response
            
        except Exception as e:
            return f"[red]⚠ Error:[/red] {str(e)}"
    
    def _get_market_data(self, question, title, description):
        """
        Fetch real market data from Polygon API if available.
        """
        polygon_key = os.getenv("POLYGON_API_KEY")
        if not polygon_key:
            return None
        
        # Try to extract ticker from the question/title/description
        import re
        text = f"{question} {title} {description}".upper()
        
        # Common stock ticker patterns
        ticker_patterns = [
            r'\b(AAPL|MSFT|GOOGL|GOOG|AMZN|META|NVDA|TSLA|AMD|INTC|NFLX|DIS|JPM|BAC|WMT|V|MA|HD|PG|JNJ|UNH|XOM|CVX|PFE|ABBV|MRK|KO|PEP|COST|AVGO|ADBE|CRM|ORCL|CSCO|ACN|TXN|QCOM|NOW|IBM|INTU|SNOW|PLTR|COIN|SQ|PYPL|ROKU|SHOP|ZM|DDOG|NET|CRWD|ZS|OKTA|MDB|U|RBLX|ABNB|UBER|LYFT|DASH|SNAP|PINS|TWTR|SPOT|SOFI|HOOD|RIVN|LCID|NIO|XPEV|LI|F|GM|BABA|JD|PDD|BIDU|NKE|LULU|TGT|LOW|TJX|SBUX|MCD|CMG|DPZ|YUM|MAR|HLT|BA|LMT|RTX|GE|CAT|DE|MMM|HON|UPS|FDX)\b',
        ]
        
        tickers = []
        for pattern in ticker_patterns:
            matches = re.findall(pattern, text)
            tickers.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        tickers = [t for t in tickers if not (t in seen or seen.add(t))]
        
        if not tickers:
            return None
        
        # Get data for up to 3 tickers
        market_info = []
        for ticker in tickers[:3]:
            try:
                # Get previous day's data
                url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?apiKey={polygon_key}"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("results"):
                        result = data["results"][0]
                        change = ((result["c"] - result["o"]) / result["o"]) * 100 if result["o"] else 0
                        market_info.append(
                            f"**{ticker}**: Open ${result['o']:.2f}, Close ${result['c']:.2f}, "
                            f"High ${result['h']:.2f}, Low ${result['l']:.2f}, "
                            f"Volume {result['v']:,.0f}, Change {change:+.2f}%"
                        )
                
                # Get ticker details
                details_url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={polygon_key}"
                resp = requests.get(details_url, timeout=10)
                if resp.status_code == 200:
                    details = resp.json().get("results", {})
                    if details.get("name"):
                        market_info.append(f"Company: {details.get('name')}")
                    if details.get("market_cap"):
                        market_info.append(f"Market Cap: ${details.get('market_cap'):,.0f}")
                        
            except Exception:
                continue
        
        return "\n".join(market_info) if market_info else None
    
    def summarize(self, messages):
        message = []
        message.append({"role": "developer", "content": summary_prompt})
        message.extend(self.parse_messages(messages))
        summary = self.llm.prompt(message)
        return summary
    
    def answer(self, question, messages):
        message = []
        answer_prompt_formatted = answer_prompt.replace("--question--", question)
        message.append({"role": "developer", "content": answer_prompt_formatted})
        message.extend(self.parse_messages(messages))
        for chunk in self.llm.prompt_stream(message):
            yield chunk

    def compact(self, messages):
        message = []
        message.append({"role": "developer", "content": compact_prompt})
        message.extend(self.parse_messages(messages))
        summary = self.llm.prompt(message)

        messages.clear()
        messages.append({"role": "user", "content": summary})
        return messages