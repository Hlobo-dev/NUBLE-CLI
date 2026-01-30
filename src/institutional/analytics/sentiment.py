"""
Sentiment Analysis Module - NLP-based sentiment analysis for financial text.
Uses FinBERT-style analysis with fallback to rule-based methods.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import math


@dataclass
class SentimentResult:
    """Result from sentiment analysis"""
    text: str
    sentiment: str  # 'positive', 'negative', 'neutral'
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    aspects: Dict[str, float] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)


@dataclass
class AggregateSentiment:
    """Aggregated sentiment across multiple sources"""
    overall_score: float  # -1 to 1
    overall_sentiment: str
    confidence: float
    news_sentiment: Optional[float] = None
    social_sentiment: Optional[float] = None
    analyst_sentiment: Optional[float] = None
    article_count: int = 0
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0


class SentimentAnalyzer:
    """
    Financial sentiment analysis engine.
    
    Features:
    - FinBERT-style sentiment classification
    - Financial lexicon-based analysis
    - Aspect-based sentiment extraction
    - Entity recognition for tickers/companies
    - Aggregation across multiple sources
    
    For production ML:
    - Install transformers: pip install transformers torch
    - Use: from transformers import AutoModelForSequenceClassification
    - Model: "ProsusAI/finbert" or "yiyanghkust/finbert-tone"
    """
    
    # Financial sentiment lexicons (subset - extend for production)
    POSITIVE_WORDS = {
        # Strong positive
        'surge', 'soar', 'skyrocket', 'breakthrough', 'outperform', 'beat',
        'exceed', 'record', 'boom', 'rally', 'bullish', 'upgrade',
        # Moderate positive  
        'gain', 'rise', 'grow', 'improve', 'strong', 'positive', 'profit',
        'revenue', 'success', 'opportunity', 'optimistic', 'confident',
        'upside', 'expansion', 'recovery', 'momentum', 'dividend',
        # Financial positive
        'buy', 'accumulate', 'overweight', 'outperform', 'recommend',
        'upbeat', 'robust', 'solid', 'healthy', 'promising',
    }
    
    NEGATIVE_WORDS = {
        # Strong negative
        'crash', 'plunge', 'collapse', 'bankruptcy', 'fraud', 'scandal',
        'crisis', 'default', 'downturn', 'bearish', 'downgrade',
        # Moderate negative
        'fall', 'drop', 'decline', 'loss', 'weak', 'negative', 'miss',
        'concern', 'risk', 'warning', 'uncertainty', 'volatile',
        'downside', 'contraction', 'recession', 'inflation',
        # Financial negative
        'sell', 'reduce', 'underweight', 'underperform', 'avoid',
        'cautious', 'sluggish', 'disappointing', 'challenging',
    }
    
    INTENSIFIERS = {
        'very': 1.5, 'extremely': 2.0, 'significantly': 1.5,
        'substantially': 1.5, 'sharply': 1.5, 'dramatically': 2.0,
        'slightly': 0.5, 'somewhat': 0.7, 'marginally': 0.5,
    }
    
    NEGATIONS = {'not', 'no', 'never', 'neither', 'nobody', 'nothing',
                 'nowhere', 'hardly', 'barely', 'scarcely', "n't", "nt"}
    
    # Financial entities pattern
    TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b(?=\s+(?:stock|shares|price))')
    
    def __init__(self, use_ml: bool = False):
        """
        Initialize sentiment analyzer.
        
        Args:
            use_ml: If True, attempt to load FinBERT model
        """
        self.use_ml = use_ml
        self._model = None
        self._tokenizer = None
        
        if use_ml:
            self._load_ml_model()
    
    def _load_ml_model(self):
        """Load FinBERT or similar model"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            model_name = "ProsusAI/finbert"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.eval()
            print(f"Loaded ML model: {model_name}")
        except ImportError:
            print("transformers not installed. Using rule-based sentiment.")
            self.use_ml = False
        except Exception as e:
            print(f"Could not load ML model: {e}. Using rule-based sentiment.")
            self.use_ml = False
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove special characters but keep $ for tickers
        text = re.sub(r'[^\w\s\$\.\,\!\?\-]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.lower()
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract financial entities (tickers, companies)"""
        entities = []
        
        # Find ticker symbols
        for match in self.TICKER_PATTERN.finditer(text.upper()):
            ticker = match.group(1) or match.group(2)
            if ticker and ticker not in ['THE', 'AND', 'FOR', 'ARE', 'BUT']:
                entities.append(ticker)
        
        return list(set(entities))
    
    def _rule_based_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Calculate sentiment using lexicon-based approach.
        
        Returns:
            Tuple of (sentiment_score, confidence)
        """
        text = self._preprocess_text(text)
        words = text.split()
        
        if not words:
            return 0.0, 0.0
        
        positive_score = 0.0
        negative_score = 0.0
        
        # Track negation window
        negation_window = 0
        current_intensifier = 1.0
        
        for i, word in enumerate(words):
            # Check for intensifiers
            if word in self.INTENSIFIERS:
                current_intensifier = self.INTENSIFIERS[word]
                continue
            
            # Check for negations
            if word in self.NEGATIONS:
                negation_window = 3  # Affect next 3 words
                continue
            
            # Score sentiment words
            is_negated = negation_window > 0
            
            if word in self.POSITIVE_WORDS:
                score = current_intensifier
                if is_negated:
                    negative_score += score * 0.7  # Negated positive = mild negative
                else:
                    positive_score += score
            elif word in self.NEGATIVE_WORDS:
                score = current_intensifier
                if is_negated:
                    positive_score += score * 0.5  # Negated negative = mild positive
                else:
                    negative_score += score
            
            # Decay negation window
            if negation_window > 0:
                negation_window -= 1
            
            # Reset intensifier
            current_intensifier = 1.0
        
        # Calculate final score
        total = positive_score + negative_score
        if total == 0:
            return 0.0, 0.2  # No sentiment words = neutral, low confidence
        
        sentiment_score = (positive_score - negative_score) / total
        
        # Confidence based on number of sentiment words
        word_ratio = total / len(words)
        confidence = min(0.3 + word_ratio * 2, 0.9)
        
        return sentiment_score, confidence
    
    def _ml_sentiment(self, text: str) -> Tuple[float, float]:
        """Calculate sentiment using ML model"""
        if not self._model or not self._tokenizer:
            return self._rule_based_sentiment(text)
        
        try:
            import torch
            
            inputs = self._tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
            
            # FinBERT: [negative, neutral, positive]
            probs = probabilities[0].tolist()
            
            # Calculate score: weighted average
            sentiment_score = probs[2] - probs[0]  # positive - negative
            confidence = max(probs)
            
            return sentiment_score, confidence
            
        except Exception as e:
            print(f"ML inference error: {e}")
            return self._rule_based_sentiment(text)
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a text.
        
        Args:
            text: Input text (headline, article, tweet, etc.)
            
        Returns:
            SentimentResult with score and details
        """
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                sentiment="neutral",
                score=0.0,
                confidence=0.0
            )
        
        # Get sentiment score
        if self.use_ml:
            score, confidence = self._ml_sentiment(text)
        else:
            score, confidence = self._rule_based_sentiment(text)
        
        # Determine sentiment label
        if score > 0.1:
            sentiment = "positive"
        elif score < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Extract entities
        entities = self._extract_entities(text)
        
        return SentimentResult(
            text=text[:500],  # Truncate for storage
            sentiment=sentiment,
            score=score,
            confidence=confidence,
            entities=entities
        )
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts"""
        return [self.analyze(text) for text in texts]
    
    def aggregate_sentiments(
        self,
        results: List[SentimentResult],
        weights: Optional[List[float]] = None
    ) -> AggregateSentiment:
        """
        Aggregate multiple sentiment results.
        
        Args:
            results: List of SentimentResult objects
            weights: Optional weights for each result (e.g., recency)
        """
        if not results:
            return AggregateSentiment(
                overall_score=0.0,
                overall_sentiment="neutral",
                confidence=0.0
            )
        
        if weights is None:
            weights = [1.0] * len(results)
        
        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum == 0:
            weights = [1.0 / len(results)] * len(results)
        else:
            weights = [w / weight_sum for w in weights]
        
        # Calculate weighted average
        weighted_score = sum(r.score * w for r, w in zip(results, weights))
        weighted_confidence = sum(r.confidence * w for r, w in zip(results, weights))
        
        # Count sentiments
        bullish = sum(1 for r in results if r.sentiment == "positive")
        bearish = sum(1 for r in results if r.sentiment == "negative")
        neutral = sum(1 for r in results if r.sentiment == "neutral")
        
        # Determine overall sentiment
        if weighted_score > 0.1:
            overall = "positive"
        elif weighted_score < -0.1:
            overall = "negative"
        else:
            overall = "neutral"
        
        return AggregateSentiment(
            overall_score=weighted_score,
            overall_sentiment=overall,
            confidence=weighted_confidence,
            article_count=len(results),
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral
        )
    
    def analyze_news_feed(
        self,
        articles: List[Dict],
        title_weight: float = 0.6,
        summary_weight: float = 0.4,
        recency_decay: float = 0.1
    ) -> AggregateSentiment:
        """
        Analyze sentiment across news articles.
        
        Args:
            articles: List of dicts with 'title', 'summary', 'published_at'
            title_weight: Weight for headline sentiment
            summary_weight: Weight for summary sentiment
            recency_decay: Decay factor for older articles
        """
        results = []
        weights = []
        
        now = datetime.now()
        
        for i, article in enumerate(articles):
            title = article.get('title', '')
            summary = article.get('summary', '')
            
            # Analyze title and summary
            title_result = self.analyze(title)
            summary_result = self.analyze(summary)
            
            # Combine scores
            combined_score = (
                title_result.score * title_weight + 
                summary_result.score * summary_weight
            )
            combined_confidence = (
                title_result.confidence * title_weight +
                summary_result.confidence * summary_weight
            )
            
            results.append(SentimentResult(
                text=title,
                sentiment="positive" if combined_score > 0.1 else "negative" if combined_score < -0.1 else "neutral",
                score=combined_score,
                confidence=combined_confidence,
                entities=title_result.entities + summary_result.entities
            ))
            
            # Calculate recency weight
            published = article.get('published_at')
            if published:
                if isinstance(published, str):
                    try:
                        published = datetime.fromisoformat(published.replace('Z', '+00:00'))
                    except:
                        published = now
                hours_old = (now - published.replace(tzinfo=None)).total_seconds() / 3600
                recency_weight = math.exp(-recency_decay * hours_old)
            else:
                recency_weight = 1.0 / (i + 1)  # Assume ordered by recency
            
            weights.append(recency_weight)
        
        return self.aggregate_sentiments(results, weights)
    
    def get_sentiment_summary(
        self,
        symbol: str,
        news_sentiment: Optional[float] = None,
        social_sentiment: Optional[float] = None,
        analyst_sentiment: Optional[float] = None
    ) -> Dict:
        """
        Create comprehensive sentiment summary.
        
        Returns dict with overall assessment and breakdown.
        """
        sentiments = []
        weights = []
        
        # News sentiment (highest weight)
        if news_sentiment is not None:
            sentiments.append(news_sentiment)
            weights.append(0.5)
        
        # Social sentiment
        if social_sentiment is not None:
            sentiments.append(social_sentiment)
            weights.append(0.25)
        
        # Analyst sentiment
        if analyst_sentiment is not None:
            sentiments.append(analyst_sentiment)
            weights.append(0.25)
        
        if not sentiments:
            overall = 0.0
        else:
            total_weight = sum(weights)
            overall = sum(s * w for s, w in zip(sentiments, weights)) / total_weight
        
        # Determine signal
        if overall > 0.3:
            signal = "strongly_bullish"
        elif overall > 0.1:
            signal = "bullish"
        elif overall < -0.3:
            signal = "strongly_bearish"
        elif overall < -0.1:
            signal = "bearish"
        else:
            signal = "neutral"
        
        return {
            "symbol": symbol,
            "overall_sentiment": overall,
            "signal": signal,
            "breakdown": {
                "news": news_sentiment,
                "social": social_sentiment,
                "analyst": analyst_sentiment,
            },
            "timestamp": datetime.now().isoformat()
        }
