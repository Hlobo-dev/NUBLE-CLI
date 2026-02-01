"""
Sentiment Analysis Module

Uses FinBERT for financial-domain sentiment analysis.
Optimized for speed with batching and caching.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Optional, Union
import numpy as np
from functools import lru_cache
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SentimentLabel(Enum):
    """Sentiment classification labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    text: str
    label: SentimentLabel
    score: float  # Confidence 0-1
    normalized_score: float  # -1 to +1 scale
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text[:100] + '...' if len(self.text) > 100 else self.text,
            'label': self.label.value,
            'confidence': self.score,
            'sentiment_score': self.normalized_score
        }


class SentimentAnalyzer:
    """
    Financial sentiment analyzer using FinBERT.
    
    FinBERT is pre-trained on financial text and outperforms
    general-purpose sentiment models on financial content.
    
    Usage:
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("Apple stock surges on strong earnings")
        print(result.normalized_score)  # +0.87
        
        # Batch analysis
        results = analyzer.analyze_batch([
            "NVIDIA beats estimates",
            "Tesla misses delivery targets",
            "Market opens flat"
        ])
    """
    
    # FinBERT model - fine-tuned for financial sentiment
    MODEL_NAME = "ProsusAI/finbert"
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = 16,
        cache_size: int = 1000
    ):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'mps', 'cpu', or None (auto-detect)
            batch_size: Batch size for inference
            cache_size: LRU cache size for repeated texts
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Cache for repeated texts
        self._cache_size = cache_size
        self._cache: Dict[str, SentimentResult] = {}
        
        logger.info(f"SentimentAnalyzer initialized (device: {self.device})")
    
    def _load_model(self):
        """Lazy load model on first use."""
        if self._model is None:
            logger.info(f"Loading FinBERT model: {self.model_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            
            # Move to device
            self._model = self._model.to(self.device)
            self._model.eval()
            
            logger.info("FinBERT model loaded successfully")
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze (headline, article, etc.)
            
        Returns:
            SentimentResult with label, confidence, and normalized score
        """
        # Check cache
        if text in self._cache:
            return self._cache[text]
        
        # Load model if needed
        self._load_model()
        
        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
        
        # Extract result
        probs_np = probs.cpu().numpy()[0]
        pred_idx = np.argmax(probs_np)
        
        # FinBERT labels: ['positive', 'negative', 'neutral']
        labels = [SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL]
        label = labels[pred_idx]
        confidence = float(probs_np[pred_idx])
        
        # Normalized score: -1 (negative) to +1 (positive)
        # Positive probability - Negative probability
        normalized = float(probs_np[0] - probs_np[1])
        
        result = SentimentResult(
            text=text,
            label=label,
            score=confidence,
            normalized_score=normalized
        )
        
        # Cache result
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = result
        
        return result
    
    def analyze_batch(
        self,
        texts: List[str],
        return_average: bool = False
    ) -> Union[List[SentimentResult], Dict]:
        """
        Analyze sentiment of multiple texts efficiently.
        
        Args:
            texts: List of texts to analyze
            return_average: If True, also return average sentiment
            
        Returns:
            List of SentimentResults, or dict with results and average
        """
        if not texts:
            return [] if not return_average else {'results': [], 'average': 0.0}
        
        # Check cache for all texts
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text in self._cache:
                results.append((i, self._cache[text]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts in batches
        if uncached_texts:
            self._load_model()
            
            for batch_start in range(0, len(uncached_texts), self.batch_size):
                batch_texts = uncached_texts[batch_start:batch_start + self.batch_size]
                batch_indices = uncached_indices[batch_start:batch_start + self.batch_size]
                
                # Tokenize batch
                inputs = self._tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Inference
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                
                probs_np = probs.cpu().numpy()
                labels = [SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL]
                
                # Process each result
                for j, (text, idx) in enumerate(zip(batch_texts, batch_indices)):
                    prob = probs_np[j]
                    pred_idx = np.argmax(prob)
                    
                    result = SentimentResult(
                        text=text,
                        label=labels[pred_idx],
                        score=float(prob[pred_idx]),
                        normalized_score=float(prob[0] - prob[1])
                    )
                    
                    # Cache
                    if len(self._cache) < self._cache_size:
                        self._cache[text] = result
                    
                    results.append((idx, result))
        
        # Sort by original order
        results.sort(key=lambda x: x[0])
        final_results = [r for _, r in results]
        
        if return_average:
            avg_score = np.mean([r.normalized_score for r in final_results])
            return {
                'results': final_results,
                'average': float(avg_score),
                'count': len(final_results),
                'positive_count': sum(1 for r in final_results if r.label == SentimentLabel.POSITIVE),
                'negative_count': sum(1 for r in final_results if r.label == SentimentLabel.NEGATIVE),
                'neutral_count': sum(1 for r in final_results if r.label == SentimentLabel.NEUTRAL)
            }
        
        return final_results
    
    def analyze_news_articles(
        self,
        articles: List[Dict]
    ) -> List[Dict]:
        """
        Analyze sentiment of news articles from StockNews API.
        
        Args:
            articles: List of article dicts with 'title' and optional 'text'
            
        Returns:
            Articles with added 'ml_sentiment' field
        """
        if not articles:
            return []
        
        # Extract texts for analysis
        texts = []
        for article in articles:
            # Use title + summary if available
            text = article.get('title', '')
            if article.get('text'):
                text += ' ' + article['text'][:500]  # First 500 chars of body
            texts.append(text)
        
        # Batch analyze
        results = self.analyze_batch(texts)
        
        # Add results to articles
        enriched = []
        for article, result in zip(articles, results):
            article_copy = article.copy()
            article_copy['ml_sentiment'] = result.to_dict()
            enriched.append(article_copy)
        
        return enriched
    
    def get_aggregate_sentiment(
        self,
        articles: List[Dict],
        weight_by_recency: bool = True
    ) -> Dict:
        """
        Get aggregate sentiment from multiple articles.
        
        Args:
            articles: List of articles with 'ml_sentiment' field
            weight_by_recency: Weight recent articles more heavily
            
        Returns:
            Aggregate sentiment metrics
        """
        if not articles:
            return {
                'aggregate_score': 0.0,
                'confidence': 0.0,
                'article_count': 0,
                'signal': 'NEUTRAL'
            }
        
        scores = []
        confidences = []
        
        for i, article in enumerate(articles):
            sentiment = article.get('ml_sentiment', {})
            score = sentiment.get('sentiment_score', 0.0)
            conf = sentiment.get('confidence', 0.5)
            
            # Apply recency weight (most recent = highest weight)
            if weight_by_recency:
                weight = (i + 1) / len(articles)  # Later = more recent = higher weight
            else:
                weight = 1.0
            
            scores.append(score * weight)
            confidences.append(conf * weight)
        
        # Normalize weights
        total_weight = sum((i + 1) / len(articles) for i in range(len(articles))) if weight_by_recency else len(articles)
        
        aggregate_score = sum(scores) / total_weight
        aggregate_confidence = sum(confidences) / total_weight
        
        # Determine signal
        if aggregate_score > 0.2 and aggregate_confidence > 0.6:
            signal = 'BULLISH'
        elif aggregate_score < -0.2 and aggregate_confidence > 0.6:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'
        
        return {
            'aggregate_score': aggregate_score,
            'confidence': aggregate_confidence,
            'article_count': len(articles),
            'signal': signal,
            'positive_ratio': sum(1 for a in articles if a.get('ml_sentiment', {}).get('sentiment_score', 0) > 0.2) / len(articles),
            'negative_ratio': sum(1 for a in articles if a.get('ml_sentiment', {}).get('sentiment_score', 0) < -0.2) / len(articles)
        }


def quick_sentiment(text: str) -> float:
    """
    Quick sentiment check without loading full model.
    Uses simple keyword matching as fallback.
    
    Returns: -1 to +1
    """
    text_lower = text.lower()
    
    positive_words = [
        'surge', 'soar', 'rally', 'beat', 'exceed', 'record', 'strong',
        'bullish', 'upgrade', 'buy', 'outperform', 'gain', 'profit',
        'growth', 'positive', 'optimistic', 'breakthrough'
    ]
    
    negative_words = [
        'plunge', 'crash', 'fall', 'miss', 'weak', 'bearish', 'downgrade',
        'sell', 'underperform', 'loss', 'decline', 'negative', 'warning',
        'risk', 'concern', 'lawsuit', 'investigation', 'fraud'
    ]
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count + neg_count == 0:
        return 0.0
    
    return (pos_count - neg_count) / (pos_count + neg_count)


# Test function
def test_analyzer():
    """Test the sentiment analyzer."""
    print("Testing Sentiment Analyzer...")
    print("=" * 50)
    
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "Apple stock surges 10% on record iPhone sales",
        "Tesla misses delivery targets, stock plunges",
        "Federal Reserve holds interest rates steady",
        "NVIDIA announces breakthrough AI chip technology",
        "Bank of America downgrades retail sector to sell",
    ]
    
    print("\nAnalyzing test headlines:")
    print("-" * 50)
    
    for text in test_texts:
        result = analyzer.analyze(text)
        emoji = "🟢" if result.normalized_score > 0.2 else ("🔴" if result.normalized_score < -0.2 else "⚪")
        print(f"{emoji} [{result.label.value:>8}] ({result.normalized_score:+.2f}) {text[:50]}...")
    
    print("\n" + "=" * 50)
    
    # Batch test
    print("\nBatch analysis with aggregation:")
    batch_result = analyzer.analyze_batch(test_texts, return_average=True)
    print(f"  Average sentiment: {batch_result['average']:+.2f}")
    print(f"  Positive: {batch_result['positive_count']}, Negative: {batch_result['negative_count']}, Neutral: {batch_result['neutral_count']}")
    
    print("\n✅ Sentiment analyzer working!")


if __name__ == "__main__":
    test_analyzer()
