"""
Emotion Detector with 3-Tier Fallback System
Tier 1: Transformer-based emotion detection (j-hartmann/emotion-english-distilroberta-base)
Tier 2: Sentiment analysis (cardiffnlp/twitter-roberta-base-sentiment)
Tier 3: VADER lexicon-based (rule-based, fast)

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import torch

# Tier 1 & 2: Hugging Face Transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers torch")

# Tier 3: VADER
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("VADER not available. Install with: pip install vaderSentiment")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionLabel(Enum):
    """Supported emotion labels"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    LOVE = "love"
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass
class EmotionResult:
    """Represents emotion detection result"""
    primary_emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    tier: str  # Which tier provided this result


class EmotionDetector:
    """
    Multi-tier emotion detection with automatic fallback.
    Automatically detects GPU/CPU and adjusts accordingly.
    """
    
    def __init__(
        self,
        emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
        sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        use_gpu: Optional[bool] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize emotion detector with multi-tier fallback.
        
        Args:
            emotion_model: Hugging Face model for Tier 1
            sentiment_model: Hugging Face model for Tier 2
            use_gpu: Force GPU usage (None = auto-detect)
            confidence_threshold: Minimum confidence threshold
        """
        self.confidence_threshold = confidence_threshold
        
        # Auto-detect GPU availability
        if use_gpu is None:
            self.device = 0 if torch.cuda.is_available() else -1
            self.use_gpu = torch.cuda.is_available()
        else:
            self.device = 0 if use_gpu else -1
            self.use_gpu = use_gpu
        
        logger.info(f"Using device: {'GPU' if self.use_gpu else 'CPU'}")
        
        # Initialize tiers
        self.tier1_detector = None  # 7-way emotion
        self.tier2_detector = None  # 3-way sentiment
        self.tier3_detector = None  # VADER
        
        self._initialize_tier1(emotion_model)
        self._initialize_tier2(sentiment_model)
        self._initialize_tier3()
        
        # Emotion history for tracking user emotional state
        self.emotion_history = []
        self.max_history = 10
    
    def _initialize_tier1(self, model_name: str):
        """Initialize Tier 1: Transformer-based emotion detection"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Tier 1 (Emotion Transformer) unavailable: transformers not installed")
            return
        
        try:
            logger.info(f"Loading Tier 1 emotion model: {model_name}")
            self.tier1_detector = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                device=self.device,
                top_k=None  # Return all emotions with scores
            )
            logger.info("✓ Tier 1 (Emotion Transformer) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tier 1: {e}")
            self.tier1_detector = None
    
    def _initialize_tier2(self, model_name: str):
        """Initialize Tier 2: Sentiment analysis"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Tier 2 (Sentiment) unavailable: transformers not installed")
            return
        
        try:
            logger.info(f"Loading Tier 2 sentiment model: {model_name}")
            self.tier2_detector = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=self.device,
                top_k=None
            )
            logger.info("✓ Tier 2 (Sentiment) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tier 2: {e}")
            self.tier2_detector = None
    
    def _initialize_tier3(self):
        """Initialize Tier 3: VADER sentiment"""
        if not VADER_AVAILABLE:
            logger.warning("Tier 3 (VADER) unavailable: vaderSentiment not installed")
            return
        
        try:
            self.tier3_detector = SentimentIntensityAnalyzer()
            logger.info("✓ Tier 3 (VADER) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tier 3: {e}")
            self.tier3_detector = None
    
    def detect(self, text: str, track_history: bool = True) -> EmotionResult:
        """
        Detect emotion with automatic fallback.
        
        Args:
            text: Input text to analyze
            track_history: Whether to add to emotion history
            
        Returns:
            Emotion detection result
        """
        # Try Tier 1 (Emotion Transformer)
        if self.tier1_detector is not None:
            try:
                result = self._detect_tier1(text)
                if track_history:
                    self._add_to_history(result)
                logger.debug(f"Detected emotion using Tier 1: {result.primary_emotion}")
                return result
            except Exception as e:
                logger.warning(f"Tier 1 failed: {e}. Falling back to Tier 2...")
        
        # Try Tier 2 (Sentiment)
        if self.tier2_detector is not None:
            try:
                result = self._detect_tier2(text)
                if track_history:
                    self._add_to_history(result)
                logger.debug(f"Detected emotion using Tier 2: {result.primary_emotion}")
                return result
            except Exception as e:
                logger.warning(f"Tier 2 failed: {e}. Falling back to Tier 3...")
        
        # Try Tier 3 (VADER)
        if self.tier3_detector is not None:
            try:
                result = self._detect_tier3(text)
                if track_history:
                    self._add_to_history(result)
                logger.debug(f"Detected emotion using Tier 3: {result.primary_emotion}")
                return result
            except Exception as e:
                logger.error(f"All tiers failed. Last error: {e}")
        
        # All tiers failed - return neutral
        return EmotionResult(
            primary_emotion=EmotionLabel.NEUTRAL.value,
            confidence=0.5,
            all_emotions={EmotionLabel.NEUTRAL.value: 0.5},
            tier="Fallback"
        )
    
    def _detect_tier1(self, text: str) -> EmotionResult:
        """Detect emotions using Tier 1 (Transformer)"""
        results = self.tier1_detector(text)[0]  # Returns list of dicts
        
        # Convert to emotion dict
        emotions = {item['label'].lower(): item['score'] for item in results}
        
        # Find primary emotion
        primary = max(emotions.items(), key=lambda x: x[1])
        
        return EmotionResult(
            primary_emotion=primary[0],
            confidence=primary[1],
            all_emotions=emotions,
            tier="Tier1-Transformer"
        )
    
    def _detect_tier2(self, text: str) -> EmotionResult:
        """Detect emotions using Tier 2 (Sentiment)"""
        results = self.tier2_detector(text)[0]
        
        # Convert sentiment to emotion-like labels
        emotions = {}
        for item in results:
            label = item['label'].lower()
            score = item['score']
            
            # Map sentiment labels to emotions
            if 'positive' in label or label == 'pos':
                emotions['positive'] = score
                emotions['joy'] = score * 0.8  # Approximate mapping
            elif 'negative' in label or label == 'neg':
                emotions['negative'] = score
                emotions['sadness'] = score * 0.5
                emotions['anger'] = score * 0.3
            elif 'neutral' in label or label == 'neu':
                emotions['neutral'] = score
        
        primary = max(emotions.items(), key=lambda x: x[1])
        
        return EmotionResult(
            primary_emotion=primary[0],
            confidence=primary[1],
            all_emotions=emotions,
            tier="Tier2-Sentiment"
        )
    
    def _detect_tier3(self, text: str) -> EmotionResult:
        """Detect emotions using Tier 3 (VADER)"""
        scores = self.tier3_detector.polarity_scores(text)
        
        # VADER returns: neg, neu, pos, compound
        emotions = {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
        
        # Map to emotion-like categories
        if scores['compound'] >= 0.05:
            emotions['joy'] = scores['pos'] * 0.8
            primary = 'joy' if scores['pos'] > 0.5 else 'positive'
        elif scores['compound'] <= -0.05:
            emotions['sadness'] = scores['neg'] * 0.5
            emotions['anger'] = scores['neg'] * 0.3
            primary = 'sadness' if scores['neg'] > 0.5 else 'negative'
        else:
            primary = 'neutral'
        
        confidence = abs(scores['compound']) if primary != 'neutral' else 0.7
        
        return EmotionResult(
            primary_emotion=primary,
            confidence=min(confidence, 1.0),
            all_emotions=emotions,
            tier="Tier3-VADER"
        )
    
    def get_emotional_state(self, window: int = 5) -> Dict[str, float]:
        """
        Get aggregate emotional state from recent history.
        
        Args:
            window: Number of recent detections to consider
            
        Returns:
            Dictionary of emotion scores
        """
        if not self.emotion_history:
            return {EmotionLabel.NEUTRAL.value: 1.0}
        
        recent = self.emotion_history[-window:]
        
        # Aggregate emotions
        emotion_counts = {}
        for result in recent:
            for emotion, score in result.all_emotions.items():
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + score
        
        # Normalize
        total = sum(emotion_counts.values())
        return {k: v/total for k, v in emotion_counts.items()}
    
    def get_emotional_trend(self) -> str:
        """
        Analyze emotional trend (improving, declining, stable).
        
        Returns:
            Trend description
        """
        if len(self.emotion_history) < 3:
            return "insufficient_data"
        
        recent_3 = self.emotion_history[-3:]
        
        # Simple trend analysis based on positive/negative
        positive_scores = []
        for result in recent_3:
            positive = result.all_emotions.get('joy', 0) + result.all_emotions.get('positive', 0)
            negative = result.all_emotions.get('sadness', 0) + result.all_emotions.get('anger', 0)
            positive_scores.append(positive - negative)
        
        if positive_scores[-1] > positive_scores[0] + 0.2:
            return "improving"
        elif positive_scores[-1] < positive_scores[0] - 0.2:
            return "declining"
        else:
            return "stable"
    
    def _add_to_history(self, result: EmotionResult):
        """Add detection result to history"""
        self.emotion_history.append(result)
        if len(self.emotion_history) > self.max_history:
            self.emotion_history = self.emotion_history[-self.max_history:]
    
    def reset_history(self):
        """Reset emotion history"""
        self.emotion_history = []
    
    def get_status(self) -> Dict[str, bool]:
        """Get status of all tiers"""
        return {
            "tier1_transformer": self.tier1_detector is not None,
            "tier2_sentiment": self.tier2_detector is not None,
            "tier3_vader": self.tier3_detector is not None,
            "gpu_available": self.use_gpu,
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = EmotionDetector()
    
    # Test sentences
    test_sentences = [
        "I'm so happy you're here!",
        "This is really frustrating and annoying",
        "I'm worried about what might happen",
        "That was an amazing surprise!",
        "I feel okay, nothing special",
        "I absolutely love this idea!",
    ]
    
    print("=" * 80)
    print("EMOTION DETECTOR - TESTING")
    print("=" * 80)
    print(f"\nStatus: {detector.get_status()}\n")
    
    for sentence in test_sentences:
        print(f"\nInput: {sentence}")
        result = detector.detect(sentence)
        
        print(f"  Primary: {result.primary_emotion:12} | "
              f"Confidence: {result.confidence:.2f} | "
              f"Tier: {result.tier}")
        
        # Show top 3 emotions
        top_emotions = sorted(result.all_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top emotions: {', '.join([f'{e}: {s:.2f}' for e, s in top_emotions])}")
    
    # Show emotional state
    print(f"\n{'='*80}")
    print("EMOTIONAL STATE ANALYSIS")
    print(f"{'='*80}")
    state = detector.get_emotional_state()
    print(f"Aggregate state: {state}")
    print(f"Trend: {detector.get_emotional_trend()}")
    
    print("\n" + "=" * 80)

