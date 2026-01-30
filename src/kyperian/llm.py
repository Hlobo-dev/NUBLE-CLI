import os
import json
from functools import wraps
from datetime import datetime, timedelta

# Try Anthropic first (Claude), fallback to OpenAI
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class TokenTracker:
    """Tracks Claude API token usage across the session."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_tracker()
        return cls._instance
    
    def _init_tracker(self):
        self.session_start = datetime.now()
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_requests = 0
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0
    
    def add_usage(self, input_tokens=0, output_tokens=0, cache_read=0, cache_creation=0):
        """Add token usage from an API call."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_read_tokens += cache_read
        self.cache_creation_tokens += cache_creation
        self.total_requests += 1
    
    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens
    
    @property
    def session_duration(self):
        return datetime.now() - self.session_start
    
    def get_cost_estimate(self):
        """Estimate cost based on Claude Sonnet pricing (as of 2025)."""
        # Claude Sonnet 4 pricing: $3/M input, $15/M output
        input_cost = (self.input_tokens / 1_000_000) * 3.0
        output_cost = (self.output_tokens / 1_000_000) * 15.0
        return input_cost + output_cost
    
    def get_summary(self):
        """Get formatted usage summary."""
        return {
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "requests": self.total_requests,
            "cost_estimate": f"${self.get_cost_estimate():.4f}",
            "session_minutes": int(self.session_duration.total_seconds() / 60)
        }
    
    def reset(self):
        """Reset the tracker."""
        self._init_tracker()


def retry_json_decode(max_retries=3):
    def decorator(func):
        @wraps(func)
        def wrapper(self, messages, model=None, requires_json=False):
            if not requires_json:
                return func(self, messages, model, requires_json)
            
            for attempt in range(max_retries):
                try:
                    return func(self, messages, model, requires_json)
                except json.JSONDecodeError:
                    if attempt == max_retries - 1:
                        return []
                    continue
            
        return wrapper
    return decorator


# Global token tracker instance
token_tracker = TokenTracker()


class LLM:
    """
    LLM wrapper that uses Claude (Anthropic) or OpenAI.
    Prefers Claude Opus 4.5 if ANTHROPIC_API_KEY is set.
    """
    
    def __init__(self):
        self.use_anthropic = False
        self.client = None
        
        # Prefer Anthropic/Claude if available
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if ANTHROPIC_AVAILABLE and anthropic_key:
            self.client = Anthropic(api_key=anthropic_key)
            self.use_anthropic = True
            self.default_model = "claude-sonnet-4-20250514"  # Fast and capable
        elif OPENAI_AVAILABLE and openai_key:
            self.client = OpenAI(api_key=openai_key)
            self.use_anthropic = False
            self.default_model = "gpt-4.1"
        else:
            raise ValueError(
                "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env"
            )

    def _convert_messages_for_anthropic(self, messages):
        """Convert OpenAI-style messages to Anthropic format"""
        system_prompt = ""
        converted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role in ("developer", "system"):
                system_prompt += content + "\n"
            elif role == "assistant":
                converted.append({"role": "assistant", "content": content})
            else:  # user
                converted.append({"role": "user", "content": content})
        
        # Ensure messages alternate between user and assistant
        # If two consecutive user messages, merge them
        cleaned = []
        for msg in converted:
            if cleaned and cleaned[-1]["role"] == msg["role"]:
                cleaned[-1]["content"] += "\n\n" + msg["content"]
            else:
                cleaned.append(msg)
        
        # Ensure first message is from user
        if cleaned and cleaned[0]["role"] != "user":
            cleaned.insert(0, {"role": "user", "content": "Continue from previous context."})
        
        return system_prompt.strip(), cleaned

    @retry_json_decode()
    def prompt(self, messages, model=None, requires_json=False):
        model = model or self.default_model
        
        if self.use_anthropic:
            system_prompt, converted_msgs = self._convert_messages_for_anthropic(messages)
            
            # Ensure we have at least one message
            if not converted_msgs:
                converted_msgs = [{"role": "user", "content": "Hello"}]
            
            response = self.client.messages.create(
                model=model if "claude" in model else self.default_model,
                max_tokens=4096,
                system=system_prompt or "You are a helpful AI assistant.",
                messages=converted_msgs,
            )
            
            # Track token usage
            if hasattr(response, 'usage'):
                token_tracker.add_usage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    cache_read=getattr(response.usage, 'cache_read_input_tokens', 0) or 0,
                    cache_creation=getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
                )
            
            response_text = response.content[0].text
        else:
            # OpenAI path
            response = self.client.responses.create(
                model=model,
                input=messages
            )
            response_text = response.output_text
        
        if requires_json:
            # Try to extract JSON from response
            response_text = response_text.strip()
            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            response_text = json.loads(response_text)
        
        return response_text
    
    def prompt_stream(self, messages, model=None):
        model = model or self.default_model
        
        if self.use_anthropic:
            system_prompt, converted_msgs = self._convert_messages_for_anthropic(messages)
            
            if not converted_msgs:
                converted_msgs = [{"role": "user", "content": "Hello"}]
            
            with self.client.messages.stream(
                model=model if "claude" in model else self.default_model,
                max_tokens=4096,
                system=system_prompt or "You are a helpful AI assistant.",
                messages=converted_msgs,
            ) as stream:
                for text in stream.text_stream:
                    yield text
                
                # Get final message for token tracking
                final_message = stream.get_final_message()
                if hasattr(final_message, 'usage'):
                    token_tracker.add_usage(
                        input_tokens=final_message.usage.input_tokens,
                        output_tokens=final_message.usage.output_tokens,
                        cache_read=getattr(final_message.usage, 'cache_read_input_tokens', 0) or 0,
                        cache_creation=getattr(final_message.usage, 'cache_creation_input_tokens', 0) or 0
                    )
        else:
            # OpenAI path
            response = self.client.responses.create(
                model=model,
                input=messages,
                stream=True
            )
            for event in response:
                if event.type == "response.output_text.delta":
                    yield event.delta