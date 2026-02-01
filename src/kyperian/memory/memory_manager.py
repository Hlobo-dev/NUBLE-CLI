#!/usr/bin/env python3
"""
KYPERIAN Memory Manager

Persistent memory layer for user profiles, conversations, predictions, and learning.
Uses SQLite for lightweight, file-based storage.
"""

import os
import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """User profile with preferences and portfolio."""
    user_id: str
    name: Optional[str] = None
    risk_tolerance: str = 'moderate'  # conservative, moderate, aggressive
    portfolio: Dict[str, float] = field(default_factory=dict)
    watchlist: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'user_id': self.user_id,
            'name': self.name,
            'risk_tolerance': self.risk_tolerance,
            'portfolio': self.portfolio,
            'watchlist': self.watchlist,
            'preferences': self.preferences,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'updated_at': self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """Create from dictionary."""
        return cls(
            user_id=data['user_id'],
            name=data.get('name'),
            risk_tolerance=data.get('risk_tolerance', 'moderate'),
            portfolio=data.get('portfolio', {}),
            watchlist=data.get('watchlist', []),
            preferences=data.get('preferences', {}),
            created_at=datetime.fromisoformat(data['created_at']) if isinstance(data.get('created_at'), str) else data.get('created_at', datetime.now()),
            updated_at=datetime.fromisoformat(data['updated_at']) if isinstance(data.get('updated_at'), str) else data.get('updated_at', datetime.now())
        )


@dataclass
class Conversation:
    """Conversation history."""
    conversation_id: str
    user_id: str
    messages: List[Dict] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to the conversation."""
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'conversation_id': self.conversation_id,
            'user_id': self.user_id,
            'messages': self.messages,
            'context': self.context,
            'started_at': self.started_at.isoformat() if isinstance(self.started_at, datetime) else self.started_at,
            'updated_at': self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Conversation':
        """Create from dictionary."""
        return cls(
            conversation_id=data['conversation_id'],
            user_id=data['user_id'],
            messages=data.get('messages', []),
            context=data.get('context', {}),
            started_at=datetime.fromisoformat(data['started_at']) if isinstance(data.get('started_at'), str) else data.get('started_at', datetime.now()),
            updated_at=datetime.fromisoformat(data['updated_at']) if isinstance(data.get('updated_at'), str) else data.get('updated_at', datetime.now())
        )


@dataclass
class Prediction:
    """Tracked prediction for learning."""
    prediction_id: str
    user_id: str
    symbol: str
    prediction_type: str  # PRICE, DIRECTION, EVENT
    predicted_value: Any
    actual_value: Optional[Any] = None
    confidence: float = 0.5
    horizon_days: int = 5
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    was_correct: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'prediction_id': self.prediction_id,
            'user_id': self.user_id,
            'symbol': self.symbol,
            'prediction_type': self.prediction_type,
            'predicted_value': self.predicted_value,
            'actual_value': self.actual_value,
            'confidence': self.confidence,
            'horizon_days': self.horizon_days,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'resolved_at': self.resolved_at.isoformat() if isinstance(self.resolved_at, datetime) else self.resolved_at,
            'was_correct': self.was_correct,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Prediction':
        """Create from dictionary."""
        return cls(
            prediction_id=data['prediction_id'],
            user_id=data['user_id'],
            symbol=data['symbol'],
            prediction_type=data['prediction_type'],
            predicted_value=data['predicted_value'],
            actual_value=data.get('actual_value'),
            confidence=data.get('confidence', 0.5),
            horizon_days=data.get('horizon_days', 5),
            created_at=datetime.fromisoformat(data['created_at']) if isinstance(data.get('created_at'), str) else data.get('created_at', datetime.now()),
            resolved_at=datetime.fromisoformat(data['resolved_at']) if data.get('resolved_at') else None,
            was_correct=data.get('was_correct'),
            metadata=data.get('metadata', {})
        )


class MemoryManager:
    """
    KYPERIAN Memory Manager
    
    Handles persistent storage for:
    - User profiles and preferences
    - Conversation history
    - Prediction tracking and learning
    - Feedback and corrections
    
    Uses SQLite for lightweight, portable storage.
    
    Example:
        memory = MemoryManager("~/.kyperian/memory.db")
        
        # Create/update user profile
        profile = UserProfile(user_id="user_123", risk_tolerance="aggressive")
        memory.save_user_profile(profile)
        
        # Save conversation
        conv = Conversation(conversation_id="conv_456", user_id="user_123")
        conv.add_message("user", "Should I buy AAPL?")
        memory.save_conversation(conv)
        
        # Track prediction
        pred = Prediction(
            prediction_id="pred_789",
            user_id="user_123",
            symbol="AAPL",
            prediction_type="DIRECTION",
            predicted_value="UP",
            confidence=0.75
        )
        memory.save_prediction(pred)
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the memory manager.
        
        Args:
            db_path: Path to SQLite database file.
                     Defaults to ~/.kyperian/memory.db
        """
        if db_path is None:
            home = os.path.expanduser("~")
            kyperian_dir = os.path.join(home, ".kyperian")
            os.makedirs(kyperian_dir, exist_ok=True)
            db_path = os.path.join(kyperian_dir, "memory.db")
        
        self.db_path = db_path
        self._init_database()
        
        logger.info(f"MemoryManager initialized with database: {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # User profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    risk_tolerance TEXT DEFAULT 'moderate',
                    portfolio TEXT DEFAULT '{}',
                    watchlist TEXT DEFAULT '[]',
                    preferences TEXT DEFAULT '{}',
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    messages TEXT DEFAULT '[]',
                    context TEXT DEFAULT '{}',
                    started_at TEXT,
                    updated_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                )
            """)
            
            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    symbol TEXT,
                    prediction_type TEXT,
                    predicted_value TEXT,
                    actual_value TEXT,
                    confidence REAL DEFAULT 0.5,
                    horizon_days INTEGER DEFAULT 5,
                    created_at TEXT,
                    resolved_at TEXT,
                    was_correct INTEGER,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                )
            """)
            
            # Feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    conversation_id TEXT,
                    rating INTEGER,
                    comment TEXT,
                    created_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id),
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_resolved ON predictions(resolved_at)")
    
    # ========== User Profile Methods ==========
    
    def save_user_profile(self, profile: UserProfile):
        """Save or update a user profile."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            profile.updated_at = datetime.now()
            
            cursor.execute("""
                INSERT OR REPLACE INTO user_profiles
                (user_id, name, risk_tolerance, portfolio, watchlist, preferences, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.user_id,
                profile.name,
                profile.risk_tolerance,
                json.dumps(profile.portfolio),
                json.dumps(profile.watchlist),
                json.dumps(profile.preferences),
                profile.created_at.isoformat() if isinstance(profile.created_at, datetime) else profile.created_at,
                profile.updated_at.isoformat()
            ))
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get a user profile by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            
            if row:
                return UserProfile(
                    user_id=row['user_id'],
                    name=row['name'],
                    risk_tolerance=row['risk_tolerance'],
                    portfolio=json.loads(row['portfolio']) if row['portfolio'] else {},
                    watchlist=json.loads(row['watchlist']) if row['watchlist'] else [],
                    preferences=json.loads(row['preferences']) if row['preferences'] else {},
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
                    updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else datetime.now()
                )
            return None
    
    def update_portfolio(self, user_id: str, portfolio: Dict[str, float]):
        """Update user's portfolio."""
        profile = self.get_user_profile(user_id)
        if profile:
            profile.portfolio = portfolio
            self.save_user_profile(profile)
        else:
            profile = UserProfile(user_id=user_id, portfolio=portfolio)
            self.save_user_profile(profile)
    
    def update_watchlist(self, user_id: str, watchlist: List[str]):
        """Update user's watchlist."""
        profile = self.get_user_profile(user_id)
        if profile:
            profile.watchlist = watchlist
            self.save_user_profile(profile)
        else:
            profile = UserProfile(user_id=user_id, watchlist=watchlist)
            self.save_user_profile(profile)
    
    # ========== Conversation Methods ==========
    
    def save_conversation(self, conversation: Conversation):
        """Save or update a conversation."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            conversation.updated_at = datetime.now()
            
            cursor.execute("""
                INSERT OR REPLACE INTO conversations
                (conversation_id, user_id, messages, context, started_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                conversation.conversation_id,
                conversation.user_id,
                json.dumps(conversation.messages),
                json.dumps(conversation.context),
                conversation.started_at.isoformat() if isinstance(conversation.started_at, datetime) else conversation.started_at,
                conversation.updated_at.isoformat()
            ))
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,))
            row = cursor.fetchone()
            
            if row:
                return Conversation(
                    conversation_id=row['conversation_id'],
                    user_id=row['user_id'],
                    messages=json.loads(row['messages']) if row['messages'] else [],
                    context=json.loads(row['context']) if row['context'] else {},
                    started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else datetime.now(),
                    updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else datetime.now()
                )
            return None
    
    def get_user_conversations(self, user_id: str, limit: int = 50) -> List[Conversation]:
        """Get all conversations for a user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit)
            )
            rows = cursor.fetchall()
            
            return [
                Conversation(
                    conversation_id=row['conversation_id'],
                    user_id=row['user_id'],
                    messages=json.loads(row['messages']) if row['messages'] else [],
                    context=json.loads(row['context']) if row['context'] else {},
                    started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else datetime.now(),
                    updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else datetime.now()
                )
                for row in rows
            ]
    
    # ========== Prediction Methods ==========
    
    def save_prediction(self, prediction: Prediction):
        """Save or update a prediction."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO predictions
                (prediction_id, user_id, symbol, prediction_type, predicted_value, actual_value,
                 confidence, horizon_days, created_at, resolved_at, was_correct, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.prediction_id,
                prediction.user_id,
                prediction.symbol,
                prediction.prediction_type,
                json.dumps(prediction.predicted_value),
                json.dumps(prediction.actual_value) if prediction.actual_value is not None else None,
                prediction.confidence,
                prediction.horizon_days,
                prediction.created_at.isoformat() if isinstance(prediction.created_at, datetime) else prediction.created_at,
                prediction.resolved_at.isoformat() if prediction.resolved_at else None,
                1 if prediction.was_correct else (0 if prediction.was_correct is False else None),
                json.dumps(prediction.metadata)
            ))
    
    def get_prediction(self, prediction_id: str) -> Optional[Prediction]:
        """Get a prediction by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE prediction_id = ?", (prediction_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_prediction(row)
            return None
    
    def get_unresolved_predictions(self) -> List[Prediction]:
        """Get all unresolved predictions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE resolved_at IS NULL")
            rows = cursor.fetchall()
            
            return [self._row_to_prediction(row) for row in rows]
    
    def resolve_prediction(self, prediction_id: str, actual_value: Any, was_correct: bool):
        """Resolve a prediction with the actual outcome."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE predictions
                SET actual_value = ?, was_correct = ?, resolved_at = ?
                WHERE prediction_id = ?
            """, (
                json.dumps(actual_value),
                1 if was_correct else 0,
                datetime.now().isoformat(),
                prediction_id
            ))
    
    def get_prediction_accuracy(self, user_id: str = None, symbol: str = None) -> Dict:
        """Get prediction accuracy statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM predictions WHERE resolved_at IS NOT NULL"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                return {'total': 0, 'correct': 0, 'accuracy': 0.0}
            
            total = len(rows)
            correct = sum(1 for row in rows if row['was_correct'] == 1)
            
            return {
                'total': total,
                'correct': correct,
                'accuracy': correct / total if total > 0 else 0.0,
                'by_confidence': self._accuracy_by_confidence(rows)
            }
    
    def _row_to_prediction(self, row) -> Prediction:
        """Convert a database row to a Prediction object."""
        return Prediction(
            prediction_id=row['prediction_id'],
            user_id=row['user_id'],
            symbol=row['symbol'],
            prediction_type=row['prediction_type'],
            predicted_value=json.loads(row['predicted_value']) if row['predicted_value'] else None,
            actual_value=json.loads(row['actual_value']) if row['actual_value'] else None,
            confidence=row['confidence'],
            horizon_days=row['horizon_days'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
            resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
            was_correct=bool(row['was_correct']) if row['was_correct'] is not None else None,
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
    
    def _accuracy_by_confidence(self, rows) -> Dict:
        """Calculate accuracy by confidence bucket."""
        buckets = {
            'low (0-0.5)': {'total': 0, 'correct': 0},
            'medium (0.5-0.75)': {'total': 0, 'correct': 0},
            'high (0.75-1.0)': {'total': 0, 'correct': 0}
        }
        
        for row in rows:
            conf = row['confidence']
            correct = row['was_correct'] == 1
            
            if conf < 0.5:
                bucket = 'low (0-0.5)'
            elif conf < 0.75:
                bucket = 'medium (0.5-0.75)'
            else:
                bucket = 'high (0.75-1.0)'
            
            buckets[bucket]['total'] += 1
            if correct:
                buckets[bucket]['correct'] += 1
        
        for bucket in buckets:
            total = buckets[bucket]['total']
            correct = buckets[bucket]['correct']
            buckets[bucket]['accuracy'] = correct / total if total > 0 else 0.0
        
        return buckets
    
    # ========== Feedback Methods ==========
    
    def save_feedback(self, user_id: str, conversation_id: str, rating: int, comment: str = None):
        """Save user feedback."""
        import uuid
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO feedback (feedback_id, user_id, conversation_id, rating, comment, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                user_id,
                conversation_id,
                rating,
                comment,
                datetime.now().isoformat()
            ))
    
    def get_average_rating(self, user_id: str = None) -> float:
        """Get average feedback rating."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute("SELECT AVG(rating) as avg_rating FROM feedback WHERE user_id = ?", (user_id,))
            else:
                cursor.execute("SELECT AVG(rating) as avg_rating FROM feedback")
            
            row = cursor.fetchone()
            return row['avg_rating'] if row and row['avg_rating'] else 0.0
    
    # ========== Utility Methods ==========
    
    def get_stats(self) -> Dict:
        """Get overall memory statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) as count FROM user_profiles")
            users = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM conversations")
            conversations = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM predictions")
            predictions = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE resolved_at IS NOT NULL")
            resolved = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM feedback")
            feedback = cursor.fetchone()['count']
            
            return {
                'users': users,
                'conversations': conversations,
                'predictions': predictions,
                'resolved_predictions': resolved,
                'feedback_count': feedback,
                'database_path': self.db_path
            }
    
    def clear_all(self):
        """Clear all data (use with caution!)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM feedback")
            cursor.execute("DELETE FROM predictions")
            cursor.execute("DELETE FROM conversations")
            cursor.execute("DELETE FROM user_profiles")
        
        logger.warning("All memory data cleared!")


__all__ = ['MemoryManager', 'UserProfile', 'Conversation', 'Prediction']
