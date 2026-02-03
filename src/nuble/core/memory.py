"""
NUBLE Memory System
===================

Provides persistent memory for:
- Conversation history
- User preferences
- Prediction tracking and accuracy
- Learning from feedback
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class UserPreferences:
    """User preferences and profile."""
    user_id: str = "default"
    risk_tolerance: str = "moderate"  # conservative, moderate, aggressive
    preferred_assets: List[str] = field(default_factory=list)
    watchlist: List[str] = field(default_factory=list)
    portfolio: Dict[str, float] = field(default_factory=dict)  # symbol -> shares
    alerts: List[Dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserPreferences':
        return cls(**data)


@dataclass
class PredictionRecord:
    """Record of a prediction for accuracy tracking."""
    id: str
    symbol: str
    timestamp: str
    prediction_type: str  # 'direction', 'price', 'signal'
    predicted_value: Any
    confidence: float
    model: str
    actual_value: Optional[Any] = None
    resolved: bool = False
    resolved_at: Optional[str] = None
    correct: Optional[bool] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Message:
    """A single message in conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)


class ConversationMemory:
    """
    Manages conversation history with sliding window.
    
    Features:
    - Automatic summarization of old messages
    - Symbol and topic tracking
    - Context retrieval for follow-up questions
    """
    
    def __init__(
        self,
        max_messages: int = 50,
        summary_threshold: int = 30,
        persist_path: Optional[str] = None
    ):
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.persist_path = Path(persist_path) if persist_path else Path.home() / '.nuble' / 'conversations'
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        self.conversations: Dict[str, List[Message]] = {}
        self.summaries: Dict[str, str] = {}
        self.active_symbols: Dict[str, List[str]] = defaultdict(list)
        self.topics: Dict[str, List[str]] = defaultdict(list)
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Dict = None
    ):
        """Add a message to conversation."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        self.conversations[conversation_id].append(message)
        
        # Extract symbols from user messages
        if role == 'user':
            symbols = self._extract_symbols(content)
            if symbols:
                self.active_symbols[conversation_id].extend(symbols)
                self.active_symbols[conversation_id] = list(set(self.active_symbols[conversation_id]))[-10:]
        
        # Manage conversation length
        if len(self.conversations[conversation_id]) > self.max_messages:
            self._summarize_old_messages(conversation_id)
        
        # Auto-persist
        self._persist(conversation_id)
    
    def get_messages(
        self,
        conversation_id: str,
        limit: int = None,
        include_summary: bool = True
    ) -> List[Dict]:
        """Get messages for a conversation."""
        messages = self.conversations.get(conversation_id, [])
        
        result = []
        
        # Include summary if exists
        if include_summary and conversation_id in self.summaries:
            result.append({
                'role': 'system',
                'content': f"Previous conversation summary: {self.summaries[conversation_id]}"
            })
        
        # Add messages
        messages_to_include = messages[-limit:] if limit else messages
        result.extend([{'role': m.role, 'content': m.content} for m in messages_to_include])
        
        return result
    
    def get_context(self, conversation_id: str) -> Dict:
        """Get full context for a conversation."""
        return {
            'messages': self.get_messages(conversation_id, limit=20),
            'summary': self.summaries.get(conversation_id),
            'active_symbols': self.active_symbols.get(conversation_id, []),
            'topics': self.topics.get(conversation_id, []),
            'message_count': len(self.conversations.get(conversation_id, []))
        }
    
    def get_recent_symbols(self, conversation_id: str, limit: int = 5) -> List[str]:
        """Get recently discussed symbols."""
        return self.active_symbols.get(conversation_id, [])[-limit:]
    
    def clear(self, conversation_id: str):
        """Clear a conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
        if conversation_id in self.summaries:
            del self.summaries[conversation_id]
        if conversation_id in self.active_symbols:
            del self.active_symbols[conversation_id]
        
        # Remove persisted file
        file_path = self.persist_path / f"{conversation_id}.json"
        if file_path.exists():
            file_path.unlink()
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text."""
        import re
        
        # Common patterns
        patterns = [
            r'\$([A-Z]{1,5})\b',  # $AAPL
            r'\b([A-Z]{2,5})\b',  # AAPL
        ]
        
        symbols = set()
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            symbols.update(matches)
        
        # Filter common words
        common = {'I', 'A', 'THE', 'AND', 'OR', 'FOR', 'TO', 'IN', 'IS', 'IT', 'MY', 'BE', 'ARE', 'DO', 'OF', 'BUY', 'SELL'}
        symbols -= common
        
        # Known valid symbols
        valid = {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'NFLX', 
                 'DIS', 'JPM', 'BAC', 'WMT', 'SPY', 'QQQ', 'BTC', 'ETH', 'GLD', 'SLV'}
        
        matched = symbols & valid
        return list(matched) if matched else list(symbols)[:5]
    
    def _summarize_old_messages(self, conversation_id: str):
        """Summarize old messages to manage context length."""
        messages = self.conversations[conversation_id]
        
        if len(messages) <= self.summary_threshold:
            return
        
        # Take older messages to summarize
        to_summarize = messages[:self.summary_threshold // 2]
        
        # Create simple summary
        symbols = set()
        topics = []
        
        for msg in to_summarize:
            syms = self._extract_symbols(msg.content)
            symbols.update(syms)
        
        summary = f"Previous discussion covered: {', '.join(symbols) if symbols else 'general topics'}"
        
        # Update state
        self.summaries[conversation_id] = summary
        self.conversations[conversation_id] = messages[self.summary_threshold // 2:]
    
    def _persist(self, conversation_id: str):
        """Persist conversation to disk."""
        try:
            data = {
                'messages': [asdict(m) for m in self.conversations.get(conversation_id, [])],
                'summary': self.summaries.get(conversation_id),
                'symbols': self.active_symbols.get(conversation_id, []),
                'topics': self.topics.get(conversation_id, [])
            }
            
            file_path = self.persist_path / f"{conversation_id}.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist conversation: {e}")
    
    def load(self, conversation_id: str) -> bool:
        """Load conversation from disk."""
        file_path = self.persist_path / f"{conversation_id}.json"
        
        if not file_path.exists():
            return False
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.conversations[conversation_id] = [
                Message(**m) for m in data.get('messages', [])
            ]
            self.summaries[conversation_id] = data.get('summary')
            self.active_symbols[conversation_id] = data.get('symbols', [])
            self.topics[conversation_id] = data.get('topics', [])
            
            return True
        except Exception as e:
            logger.warning(f"Failed to load conversation: {e}")
            return False


class PredictionTracker:
    """
    Tracks predictions and measures accuracy over time.
    
    This allows the system to:
    - Learn which models perform best
    - Adjust confidence calibration
    - Provide accuracy statistics to users
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = Path(persist_path) if persist_path else Path.home() / '.nuble' / 'predictions'
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        self.predictions: Dict[str, PredictionRecord] = {}
        self.model_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total': 0, 'correct': 0, 'incorrect': 0, 'pending': 0
        })
        
        self._load()
    
    def record_prediction(
        self,
        symbol: str,
        prediction_type: str,
        predicted_value: Any,
        confidence: float,
        model: str = "unknown"
    ) -> str:
        """Record a new prediction."""
        import uuid
        
        prediction_id = str(uuid.uuid4())[:8]
        
        record = PredictionRecord(
            id=prediction_id,
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            prediction_type=prediction_type,
            predicted_value=predicted_value,
            confidence=confidence,
            model=model
        )
        
        self.predictions[prediction_id] = record
        self.model_stats[model]['total'] += 1
        self.model_stats[model]['pending'] += 1
        
        self._persist()
        
        return prediction_id
    
    def resolve_prediction(
        self,
        prediction_id: str,
        actual_value: Any
    ) -> Optional[bool]:
        """Resolve a prediction with actual outcome."""
        if prediction_id not in self.predictions:
            return None
        
        record = self.predictions[prediction_id]
        record.actual_value = actual_value
        record.resolved = True
        record.resolved_at = datetime.now().isoformat()
        
        # Determine if correct
        if record.prediction_type == 'direction':
            # Direction: bullish/bearish matches actual movement
            if isinstance(record.predicted_value, str):
                actual_direction = 'bullish' if actual_value > 0 else 'bearish' if actual_value < 0 else 'neutral'
                record.correct = record.predicted_value.lower() == actual_direction
            else:
                record.correct = (record.predicted_value > 0) == (actual_value > 0)
        else:
            # Price: within 5% tolerance
            tolerance = 0.05
            diff = abs(record.predicted_value - actual_value) / abs(record.predicted_value) if record.predicted_value != 0 else 1
            record.correct = diff <= tolerance
        
        # Update stats
        model = record.model
        self.model_stats[model]['pending'] -= 1
        if record.correct:
            self.model_stats[model]['correct'] += 1
        else:
            self.model_stats[model]['incorrect'] += 1
        
        self._persist()
        
        return record.correct
    
    def get_accuracy(self, model: str = None) -> Dict:
        """Get accuracy statistics."""
        if model:
            stats = self.model_stats.get(model, {})
            resolved = stats.get('correct', 0) + stats.get('incorrect', 0)
            accuracy = stats.get('correct', 0) / resolved if resolved > 0 else 0
            return {
                'model': model,
                'total': stats.get('total', 0),
                'resolved': resolved,
                'pending': stats.get('pending', 0),
                'accuracy': round(accuracy, 3)
            }
        
        # Overall stats
        all_stats = {}
        for m, stats in self.model_stats.items():
            resolved = stats.get('correct', 0) + stats.get('incorrect', 0)
            accuracy = stats.get('correct', 0) / resolved if resolved > 0 else 0
            all_stats[m] = {
                'total': stats.get('total', 0),
                'resolved': resolved,
                'accuracy': round(accuracy, 3)
            }
        
        return all_stats
    
    def get_pending_predictions(self, symbol: str = None) -> List[PredictionRecord]:
        """Get unresolved predictions."""
        pending = [p for p in self.predictions.values() if not p.resolved]
        if symbol:
            pending = [p for p in pending if p.symbol == symbol]
        return pending
    
    def _persist(self):
        """Persist predictions to disk."""
        try:
            data = {
                'predictions': {k: v.to_dict() for k, v in self.predictions.items()},
                'model_stats': dict(self.model_stats)
            }
            
            file_path = self.persist_path / 'predictions.json'
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist predictions: {e}")
    
    def _load(self):
        """Load predictions from disk."""
        file_path = self.persist_path / 'predictions.json'
        
        if not file_path.exists():
            return
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for k, v in data.get('predictions', {}).items():
                self.predictions[k] = PredictionRecord(**v)
            
            for m, stats in data.get('model_stats', {}).items():
                self.model_stats[m] = stats
        except Exception as e:
            logger.warning(f"Failed to load predictions: {e}")


class MemoryManager:
    """
    Central memory manager that coordinates all memory systems.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path.home() / '.nuble'
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.conversations = ConversationMemory(
            persist_path=str(self.base_path / 'conversations')
        )
        self.predictions = PredictionTracker(
            persist_path=str(self.base_path / 'predictions')
        )
        self.preferences = self._load_preferences()
    
    def get_user_preferences(self, user_id: str = "default") -> UserPreferences:
        """Get user preferences."""
        return self.preferences.get(user_id, UserPreferences(user_id=user_id))
    
    def update_preferences(self, user_id: str, **kwargs):
        """Update user preferences."""
        prefs = self.get_user_preferences(user_id)
        
        for key, value in kwargs.items():
            if hasattr(prefs, key):
                setattr(prefs, key, value)
        
        prefs.updated_at = datetime.now().isoformat()
        self.preferences[user_id] = prefs
        self._save_preferences()
    
    def add_to_watchlist(self, user_id: str, symbol: str):
        """Add symbol to watchlist."""
        prefs = self.get_user_preferences(user_id)
        if symbol.upper() not in prefs.watchlist:
            prefs.watchlist.append(symbol.upper())
            self.update_preferences(user_id, watchlist=prefs.watchlist)
    
    def _load_preferences(self) -> Dict[str, UserPreferences]:
        """Load all user preferences."""
        prefs = {}
        file_path = self.base_path / 'preferences.json'
        
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                for user_id, pref_data in data.items():
                    prefs[user_id] = UserPreferences.from_dict(pref_data)
            except Exception as e:
                logger.warning(f"Failed to load preferences: {e}")
        
        return prefs
    
    def _save_preferences(self):
        """Save all user preferences."""
        try:
            data = {k: v.to_dict() for k, v in self.preferences.items()}
            
            file_path = self.base_path / 'preferences.json'
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save preferences: {e}")
