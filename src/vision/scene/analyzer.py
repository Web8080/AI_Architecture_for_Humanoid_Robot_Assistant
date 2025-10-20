"""
Scene Understanding with Multi-Tier Fallback Architecture

Implements robust scene analysis with automatic fallback:
- Tier 1: CLIP (ViT-L) - Vision-language understanding
- Tier 2: CLIP-small - Fast variant
- Tier 3: Classical CV (color histograms, basic features)

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
import cv2

# Tier 1 & 2: CLIP
try:
    import torch
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not available. Install with: pip install ftfy regex tqdm git+https://github.com/openai/CLIP.git")

logger = logging.getLogger(__name__)


@dataclass
class SceneDescription:
    """Scene description result"""
    labels: List[Tuple[str, float]]  # (label, confidence) pairs
    dominant_colors: List[Tuple[int, int, int]]  # RGB colors
    brightness: float  # 0-1
    indoor_outdoor_score: float  # <0.5 = indoor, >0.5 = outdoor
    tier: str = "Unknown"


@dataclass
class SceneResult:
    """Container for scene analysis results"""
    description: SceneDescription
    inference_time_ms: float
    tier_used: str
    image_shape: Tuple[int, int, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'labels': self.description.labels,
            'dominant_colors': self.description.dominant_colors,
            'brightness': self.description.brightness,
            'indoor_outdoor': 'outdoor' if self.description.indoor_outdoor_score > 0.5 else 'indoor',
            'indoor_outdoor_score': self.description.indoor_outdoor_score,
            'inference_time_ms': self.inference_time_ms,
            'tier_used': self.tier_used,
            'image_shape': self.image_shape
        }


class SceneAnalyzer:
    """Multi-tier scene analyzer with automatic fallback"""
    
    SCENE_LABELS = [
        "indoor", "outdoor", "urban", "nature", "office", "home",
        "street", "building", "forest", "beach", "mountain", "room",
        "kitchen", "bedroom", "bathroom", "living room", "restaurant",
        "shop", "park", "garden", "sky", "water", "grass"
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = self._detect_device()
        
        self.tier1_model = None
        self.tier1_preprocess = None
        self.tier2_model = None
        self.tier2_preprocess = None
        
        self.tier1_enabled = self.config.get('tier1_enabled', False)
        self.tier2_enabled = self.config.get('tier2_enabled', False)
        self.tier3_enabled = self.config.get('tier3_enabled', True)
        
        self._init_tiers()
    
    def _detect_device(self) -> str:
        """Auto-detect best available device"""
        if not CLIP_AVAILABLE:
            return 'cpu'
        
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _init_tiers(self):
        """Initialize all available tiers"""
        # Tier 1: CLIP ViT-L
        if self.tier1_enabled and CLIP_AVAILABLE:
            try:
                model_name = self.config.get('tier1_model', 'ViT-L/14')
                logger.info(f"Loading Tier 1 CLIP model: {model_name}")
                self.tier1_model, self.tier1_preprocess = clip.load(model_name, device=self.device)
                logger.info(f" Tier 1 (CLIP ViT-L) initialized on {self.device}")
            except Exception as e:
                logger.warning(f"Tier 1 (CLIP) failed: {e}")
        
        # Tier 2: CLIP small
        if self.tier2_enabled and CLIP_AVAILABLE:
            try:
                model_name = self.config.get('tier2_model', 'ViT-B/32')
                logger.info(f"Loading Tier 2 CLIP model: {model_name}")
                self.tier2_model, self.tier2_preprocess = clip.load(model_name, device='cpu')
                logger.info(" Tier 2 (CLIP ViT-B) initialized on CPU")
            except Exception as e:
                logger.warning(f"Tier 2 (CLIP) failed: {e}")
        
        # Tier 3: Classical CV
        if self.tier3_enabled:
            logger.info(" Tier 3 (Classical CV) initialized")
    
    def analyze(self, image: np.ndarray) -> SceneResult:
        """Analyze scene in image"""
        # Try Tier 1
        if self.tier1_enabled and self.tier1_model is not None:
            try:
                return self._analyze_tier1(image)
            except Exception as e:
                logger.warning(f"Tier 1 failed: {e}. Falling back to Tier 2.")
        
        # Try Tier 2
        if self.tier2_enabled and self.tier2_model is not None:
            try:
                return self._analyze_tier2(image)
            except Exception as e:
                logger.warning(f"Tier 2 failed: {e}. Falling back to Tier 3.")
        
        # Fallback to Tier 3
        if self.tier3_enabled:
            try:
                return self._analyze_tier3(image)
            except Exception as e:
                logger.error(f"All tiers failed: {e}")
                return self._empty_result(image)
        
        return self._empty_result(image)
    
    def _analyze_tier1(self, image: np.ndarray) -> SceneResult:
        """Tier 1: CLIP ViT-L"""
        import torch
        
        start_time = time.time()
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess
        image_input = self.tier1_preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Encode text labels
        text_inputs = torch.cat([clip.tokenize(f"a photo of {label}") for label in self.SCENE_LABELS]).to(self.device)
        
        # Get similarities
        with torch.no_grad():
            image_features = self.tier1_model.encode_image(image_input)
            text_features = self.tier1_model.encode_text(text_inputs)
            
            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Get top predictions
        values, indices = similarity[0].topk(5)
        labels = [(self.SCENE_LABELS[idx], float(val)) for val, idx in zip(values, indices)]
        
        # Get basic features
        dominant_colors = self._get_dominant_colors(image)
        brightness = self._get_brightness(image)
        indoor_outdoor = self._estimate_indoor_outdoor(labels)
        
        inference_time = (time.time() - start_time) * 1000
        
        description = SceneDescription(
            labels=labels,
            dominant_colors=dominant_colors,
            brightness=brightness,
            indoor_outdoor_score=indoor_outdoor,
            tier='Tier1-CLIP-L'
        )
        
        return SceneResult(description, inference_time, 'Tier1-CLIP-L', image.shape)
    
    def _analyze_tier2(self, image: np.ndarray) -> SceneResult:
        """Tier 2: CLIP ViT-B (small)"""
        import torch
        
        start_time = time.time()
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_image = Image.fromarray(image_rgb)
        
        image_input = self.tier2_preprocess(pil_image).unsqueeze(0).to('cpu')
        text_inputs = torch.cat([clip.tokenize(f"a photo of {label}") for label in self.SCENE_LABELS]).to('cpu')
        
        with torch.no_grad():
            image_features = self.tier2_model.encode_image(image_input)
            text_features = self.tier2_model.encode_text(text_inputs)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        values, indices = similarity[0].topk(5)
        labels = [(self.SCENE_LABELS[idx], float(val)) for val, idx in zip(values, indices)]
        
        dominant_colors = self._get_dominant_colors(image)
        brightness = self._get_brightness(image)
        indoor_outdoor = self._estimate_indoor_outdoor(labels)
        
        inference_time = (time.time() - start_time) * 1000
        
        description = SceneDescription(labels, dominant_colors, brightness, indoor_outdoor, 'Tier2-CLIP-B')
        
        return SceneResult(description, inference_time, 'Tier2-CLIP-B', image.shape)
    
    def _analyze_tier3(self, image: np.ndarray) -> SceneResult:
        """Tier 3: Classical CV features"""
        start_time = time.time()
        
        # Basic color and brightness analysis
        dominant_colors = self._get_dominant_colors(image)
        brightness = self._get_brightness(image)
        
        # Simple heuristics for scene classification
        labels = []
        
        # Blue dominant -> likely sky/outdoor
        if dominant_colors[0][2] > 150:  # High blue channel
            labels.append(("outdoor", 0.65))
            labels.append(("sky", 0.55))
            indoor_outdoor = 0.65
        # Green dominant -> nature
        elif dominant_colors[0][1] > 150:  # High green channel
            labels.append(("nature", 0.60))
            labels.append(("outdoor", 0.60))
            indoor_outdoor = 0.60
        # Darker images -> indoor
        elif brightness < 0.4:
            labels.append(("indoor", 0.60))
            labels.append(("room", 0.50))
            indoor_outdoor = 0.35
        else:
            labels.append(("indoor", 0.55))
            labels.append(("building", 0.45))
            indoor_outdoor = 0.40
        
        inference_time = (time.time() - start_time) * 1000
        
        description = SceneDescription(labels, dominant_colors, brightness, indoor_outdoor, 'Tier3-Classical')
        
        return SceneResult(description, inference_time, 'Tier3-Classical', image.shape)
    
    def _get_dominant_colors(self, image: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
        """Get dominant colors using k-means"""
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Sample for speed
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        # Convert BGR to RGB and return as tuples
        colors = []
        for center in centers:
            b, g, r = map(int, center)
            colors.append((r, g, b))
        
        return colors
    
    def _get_brightness(self, image: np.ndarray) -> float:
        """Get average brightness (0-1)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(gray.mean() / 255.0)
    
    def _estimate_indoor_outdoor(self, labels: List[Tuple[str, float]]) -> float:
        """Estimate indoor/outdoor score from labels"""
        outdoor_keywords = ['outdoor', 'sky', 'nature', 'street', 'beach', 'mountain', 'park', 'garden']
        indoor_keywords = ['indoor', 'room', 'office', 'home', 'kitchen', 'bedroom']
        
        outdoor_score = sum(conf for label, conf in labels if any(kw in label for kw in outdoor_keywords))
        indoor_score = sum(conf for label, conf in labels if any(kw in label for kw in indoor_keywords))
        
        total = outdoor_score + indoor_score
        if total > 0:
            return outdoor_score / total
        return 0.5
    
    def _empty_result(self, image: np.ndarray) -> SceneResult:
        """Return empty result"""
        description = SceneDescription([], [], 0.0, 0.5, 'None')
        return SceneResult(description, 0.0, 'None', image.shape)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'tier1_clip_l': self.tier1_model is not None,
            'tier2_clip_b': self.tier2_model is not None,
            'tier3_classical': self.tier3_enabled,
            'device': self.device
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("="*80)
    print("SCENE ANALYZER - TESTING")
    print("="*80)
    
    analyzer = SceneAnalyzer()
    print(f"\nStatus: {analyzer.get_status()}\n")
    
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = analyzer.analyze(test_image)
    
    print(f"Tier Used: {result.tier_used}")
    print(f"Labels: {result.description.labels[:3]}")
    print(f"Brightness: {result.description.brightness:.2f}")
    print(f"Indoor/Outdoor: {'outdoor' if result.description.indoor_outdoor_score > 0.5 else 'indoor'}")
    print(f"Inference Time: {result.inference_time_ms:.1f}ms")
    print("="*80)

