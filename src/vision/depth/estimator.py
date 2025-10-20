"""
Depth Estimation with Multi-Tier Fallback Architecture

Implements robust monocular depth estimation with automatic fallback:
- Tier 1: MiDaS v3.1 (DPT-Large) - Best quality
- Tier 2: MiDaS small - Fast, CPU-friendly
- Tier 3: Stereo matching (if stereo camera available) / Simple heuristics

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
import cv2

# Tier 1 & 2: MiDaS
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available")

# OpenCV for Tier 3
OPENCV_AVAILABLE = True

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    """Container for depth estimation results"""
    depth_map: np.ndarray  # Depth map (normalized 0-1 or metric)
    inference_time_ms: float
    tier_used: str
    is_metric: bool  # True if depth is in meters, False if relative
    min_depth: float
    max_depth: float
    image_shape: Tuple[int, int, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'depth_map_shape': self.depth_map.shape,
            'inference_time_ms': self.inference_time_ms,
            'tier_used': self.tier_used,
            'is_metric': self.is_metric,
            'min_depth': float(self.min_depth),
            'max_depth': float(self.max_depth),
            'image_shape': self.image_shape
        }


class DepthEstimator:
    """
    Multi-tier depth estimator with automatic fallback
    
    Tier 1: MiDaS DPT-Large (GPU) - Best quality
    Tier 2: MiDaS small (CPU) - Fast
    Tier 3: Stereo matching or heuristics - Fallback
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize depth estimator with multi-tier fallback
        
        Args:
            config: Configuration dict with tier settings
        """
        self.config = config or {}
        self.device = self._detect_device()
        
        # Initialize tiers
        self.tier1_model = None
        self.tier1_transform = None
        self.tier2_model = None
        self.tier2_transform = None
        
        self.tier1_enabled = self.config.get('tier1_enabled', False)  # Heavy model
        self.tier2_enabled = self.config.get('tier2_enabled', True)
        self.tier3_enabled = self.config.get('tier3_enabled', True)
        
        self._init_tiers()
    
    def _detect_device(self) -> str:
        """Auto-detect best available device"""
        if not TORCH_AVAILABLE:
            return 'cpu'
        
        import torch
        use_gpu = self.config.get('use_gpu', None)
        
        if use_gpu is None:
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        elif use_gpu:
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            return 'cpu'
    
    def _init_tiers(self):
        """Initialize all available tiers"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Only Tier 3 will work.")
            return
        
        import torch
        
        # Tier 1: MiDaS DPT-Large
        if self.tier1_enabled:
            try:
                model_type = self.config.get('tier1_model', 'DPT_Large')
                logger.info(f"Loading Tier 1 MiDaS model: {model_type}")
                
                self.tier1_model = torch.hub.load("intel-isl/MiDaS", model_type)
                self.tier1_model.to(self.device)
                self.tier1_model.eval()
                
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                self.tier1_transform = midas_transforms.dpt_transform
                
                logger.info(f" Tier 1 (MiDaS DPT-Large) initialized on {self.device}")
            except Exception as e:
                logger.warning(f"Tier 1 (MiDaS DPT-Large) failed to load: {e}")
        
        # Tier 2: MiDaS Small
        if self.tier2_enabled:
            try:
                model_type = self.config.get('tier2_model', 'MiDaS_small')
                logger.info(f"Loading Tier 2 MiDaS model: {model_type}")
                
                self.tier2_model = torch.hub.load("intel-isl/MiDaS", model_type)
                self.tier2_model.to('cpu')  # Always CPU for Tier 2
                self.tier2_model.eval()
                
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                self.tier2_transform = midas_transforms.small_transform
                
                logger.info(" Tier 2 (MiDaS Small) initialized on CPU")
            except Exception as e:
                logger.warning(f"Tier 2 (MiDaS Small) failed to load: {e}")
        
        # Tier 3: Classical CV (always available)
        if self.tier3_enabled:
            logger.info(" Tier 3 (Classical CV) initialized")
    
    def estimate(
        self,
        image: np.ndarray,
        normalize: bool = True
    ) -> DepthResult:
        """
        Estimate depth from image using best available tier
        
        Args:
            image: Input image (BGR format)
            normalize: If True, normalize depth to 0-1 range
            
        Returns:
            DepthResult with depth map
        """
        # Try Tier 1 (MiDaS DPT-Large)
        if self.tier1_enabled and self.tier1_model is not None:
            try:
                return self._estimate_tier1(image, normalize)
            except Exception as e:
                logger.warning(f"Tier 1 depth estimation failed: {e}. Falling back to Tier 2.")
        
        # Try Tier 2 (MiDaS Small)
        if self.tier2_enabled and self.tier2_model is not None:
            try:
                return self._estimate_tier2(image, normalize)
            except Exception as e:
                logger.warning(f"Tier 2 depth estimation failed: {e}. Falling back to Tier 3.")
        
        # Fallback to Tier 3 (Classical CV)
        if self.tier3_enabled:
            try:
                return self._estimate_tier3(image, normalize)
            except Exception as e:
                logger.error(f"All depth estimation tiers failed: {e}")
                return DepthResult(
                    depth_map=np.zeros(image.shape[:2], dtype=np.float32),
                    inference_time_ms=0.0,
                    tier_used='None',
                    is_metric=False,
                    min_depth=0.0,
                    max_depth=0.0,
                    image_shape=image.shape
                )
        
        return DepthResult(
            depth_map=np.zeros(image.shape[:2], dtype=np.float32),
            inference_time_ms=0.0,
            tier_used='None',
            is_metric=False,
            min_depth=0.0,
            max_depth=0.0,
            image_shape=image.shape
        )
    
    def _estimate_tier1(
        self,
        image: np.ndarray,
        normalize: bool
    ) -> DepthResult:
        """Tier 1: MiDaS DPT-Large"""
        import torch
        
        start_time = time.time()
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform
        input_batch = self.tier1_transform(img_rgb).to(self.device)
        
        # Inference
        with torch.no_grad():
            prediction = self.tier1_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize if requested
        if normalize:
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
        
        inference_time = (time.time() - start_time) * 1000
        
        logger.debug(f"Tier 1: Estimated depth in {inference_time:.1f}ms")
        
        return DepthResult(
            depth_map=depth_map.astype(np.float32),
            inference_time_ms=inference_time,
            tier_used='Tier1-MiDaS-DPT',
            is_metric=False,  # MiDaS produces relative depth
            min_depth=float(depth_map.min()),
            max_depth=float(depth_map.max()),
            image_shape=image.shape
        )
    
    def _estimate_tier2(
        self,
        image: np.ndarray,
        normalize: bool
    ) -> DepthResult:
        """Tier 2: MiDaS Small"""
        import torch
        
        start_time = time.time()
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform
        input_batch = self.tier2_transform(img_rgb).to('cpu')
        
        # Inference
        with torch.no_grad():
            prediction = self.tier2_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize if requested
        if normalize:
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
        
        inference_time = (time.time() - start_time) * 1000
        
        logger.debug(f"Tier 2: Estimated depth in {inference_time:.1f}ms")
        
        return DepthResult(
            depth_map=depth_map.astype(np.float32),
            inference_time_ms=inference_time,
            tier_used='Tier2-MiDaS-Small',
            is_metric=False,
            min_depth=float(depth_map.min()),
            max_depth=float(depth_map.max()),
            image_shape=image.shape
        )
    
    def _estimate_tier3(
        self,
        image: np.ndarray,
        normalize: bool
    ) -> DepthResult:
        """Tier 3: Classical CV heuristics"""
        start_time = time.time()
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use image gradients as proxy for depth (simple heuristic)
        # Stronger edges typically indicate closer objects
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Invert (high gradient = close = low depth value)
        depth_map = gradient_magnitude.max() - gradient_magnitude
        
        # Smooth
        depth_map = cv2.GaussianBlur(depth_map.astype(np.float32), (15, 15), 0)
        
        # Normalize if requested
        if normalize:
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
        
        inference_time = (time.time() - start_time) * 1000
        
        logger.debug(f"Tier 3: Estimated depth in {inference_time:.1f}ms")
        
        return DepthResult(
            depth_map=depth_map.astype(np.float32),
            inference_time_ms=inference_time,
            tier_used='Tier3-Heuristic',
            is_metric=False,
            min_depth=float(depth_map.min()),
            max_depth=float(depth_map.max()),
            image_shape=image.shape
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get estimator status"""
        return {
            'tier1_midas_dpt': self.tier1_model is not None,
            'tier2_midas_small': self.tier2_model is not None,
            'tier3_classical': self.tier3_enabled,
            'device': self.device,
            'torch_available': TORCH_AVAILABLE
        }


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("DEPTH ESTIMATOR - TESTING")
    print("="*80)
    
    # Initialize
    estimator = DepthEstimator()
    
    # Show status
    status = estimator.get_status()
    print(f"\nStatus: {status}\n")
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test depth estimation
    print("Testing depth estimation...")
    result = estimator.estimate(test_image, normalize=True)
    
    print(f"\nDepth Estimation Results:")
    print(f"  Tier Used: {result.tier_used}")
    print(f"  Inference Time: {result.inference_time_ms:.1f}ms")
    print(f"  Depth Map Shape: {result.depth_map.shape}")
    print(f"  Is Metric: {result.is_metric}")
    print(f"  Min Depth: {result.min_depth:.3f}")
    print(f"  Max Depth: {result.max_depth:.3f}")
    
    print("\n" + "="*80)

