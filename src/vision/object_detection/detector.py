"""
Object Detector with Multi-Tier Fallback Architecture

Implements robust object detection with automatic fallback:
- Tier 1: YOLOv8-x (GPU, highest accuracy)
- Tier 2: YOLOv8-nano (CPU-optimized, fast)
- Tier 3: Haar Cascades (classical CV, always works)

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

# Tier 1: YOLOv8 (Ultralytics)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLOv8 not available. Install with: pip install ultralytics")

# Tier 3: OpenCV for Haar Cascades
OPENCV_AVAILABLE = True  # Always available

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single object detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    tier: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'tier': self.tier
        }


@dataclass
class DetectionResult:
    """Container for detection results"""
    detections: List[Detection]
    inference_time_ms: float
    tier_used: str
    image_shape: Tuple[int, int, int]  # H, W, C
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'detections': [d.to_dict() for d in self.detections],
            'count': len(self.detections),
            'inference_time_ms': self.inference_time_ms,
            'tier_used': self.tier_used,
            'image_shape': self.image_shape
        }


class ObjectDetector:
    """
    Multi-tier object detector with automatic fallback
    
    Tier 1: YOLOv8-x (GPU) - Best accuracy, requires GPU
    Tier 2: YOLOv8-nano (CPU) - Fast, CPU-optimized  
    Tier 3: Haar Cascades - Classical CV, always works
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize object detector with multi-tier fallback
        
        Args:
            config: Configuration dict with tier settings
        """
        self.config = config or {}
        self.device = self._detect_device()
        
        # Initialize tiers
        self.tier1_model = None
        self.tier2_model = None
        self.tier3_cascades = {}
        
        self.tier1_enabled = self.config.get('tier1_enabled', True)
        self.tier2_enabled = self.config.get('tier2_enabled', True)
        self.tier3_enabled = self.config.get('tier3_enabled', True)
        
        self._init_tiers()
        
    def _detect_device(self) -> str:
        """Auto-detect best available device"""
        import torch
        
        use_gpu = self.config.get('use_gpu', None)
        
        if use_gpu is None:
            # Auto-detect
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
        # Tier 1: YOLOv8-x (Best)
        if self.tier1_enabled and YOLO_AVAILABLE:
            try:
                model_path = self.config.get('tier1_model', 'yolov8x.pt')
                logger.info(f"Loading Tier 1 model: {model_path}")
                self.tier1_model = YOLO(model_path)
                self.tier1_model.to(self.device)
                logger.info(f" Tier 1 (YOLOv8-x) initialized on {self.device}")
            except Exception as e:
                logger.warning(f"Tier 1 (YOLOv8-x) failed to load: {e}")
        
        # Tier 2: YOLOv8-nano (Fast)
        if self.tier2_enabled and YOLO_AVAILABLE:
            try:
                model_path = self.config.get('tier2_model', 'yolov8n.pt')
                logger.info(f"Loading Tier 2 model: {model_path}")
                self.tier2_model = YOLO(model_path)
                self.tier2_model.to('cpu')  # Always CPU for Tier 2
                logger.info(" Tier 2 (YOLOv8-nano) initialized on CPU")
            except Exception as e:
                logger.warning(f"Tier 2 (YOLOv8-nano) failed to load: {e}")
        
        # Tier 3: Haar Cascades (Fallback)
        if self.tier3_enabled:
            try:
                self._load_haar_cascades()
                logger.info(" Tier 3 (Haar Cascades) initialized")
            except Exception as e:
                logger.error(f"Tier 3 (Haar Cascades) failed to load: {e}")
    
    def _load_haar_cascades(self):
        """Load pre-trained Haar Cascades"""
        cascade_dir = cv2.data.haarcascades
        
        cascades = {
            'face': 'haarcascade_frontalface_default.xml',
            'eye': 'haarcascade_eye.xml',
            'upper_body': 'haarcascade_upperbody.xml',
            'full_body': 'haarcascade_fullbody.xml',
        }
        
        for name, filename in cascades.items():
            cascade_path = Path(cascade_dir) / filename
            if cascade_path.exists():
                self.tier3_cascades[name] = cv2.CascadeClassifier(str(cascade_path))
                logger.debug(f"Loaded Haar cascade: {name}")
    
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.25,
        max_detections: int = 100
    ) -> DetectionResult:
        """
        Detect objects in image using best available tier
        
        Args:
            image: Input image (BGR format, numpy array)
            confidence_threshold: Minimum confidence for detections
            max_detections: Maximum number of detections to return
            
        Returns:
            DetectionResult with detected objects
        """
        # Try Tier 1 (YOLOv8-x on GPU)
        if self.tier1_enabled and self.tier1_model is not None:
            try:
                return self._detect_tier1(image, confidence_threshold, max_detections)
            except Exception as e:
                logger.warning(f"Tier 1 detection failed: {e}. Falling back to Tier 2.")
        
        # Try Tier 2 (YOLOv8-nano on CPU)
        if self.tier2_enabled and self.tier2_model is not None:
            try:
                return self._detect_tier2(image, confidence_threshold, max_detections)
            except Exception as e:
                logger.warning(f"Tier 2 detection failed: {e}. Falling back to Tier 3.")
        
        # Fallback to Tier 3 (Haar Cascades)
        if self.tier3_enabled and self.tier3_cascades:
            try:
                return self._detect_tier3(image, confidence_threshold, max_detections)
            except Exception as e:
                logger.error(f"All detection tiers failed: {e}")
                return DetectionResult(
                    detections=[],
                    inference_time_ms=0.0,
                    tier_used='None',
                    image_shape=image.shape
                )
        
        # No tiers available
        logger.error("No detection tiers available!")
        return DetectionResult(
            detections=[],
            inference_time_ms=0.0,
            tier_used='None',
            image_shape=image.shape
        )
    
    def _detect_tier1(
        self,
        image: np.ndarray,
        confidence_threshold: float,
        max_detections: int
    ) -> DetectionResult:
        """Tier 1: YOLOv8-x detection (GPU)"""
        start_time = time.time()
        
        # Run inference
        results = self.tier1_model(image, conf=confidence_threshold, verbose=False)[0]
        
        inference_time = (time.time() - start_time) * 1000
        
        # Parse results
        detections = []
        boxes = results.boxes
        
        for i in range(min(len(boxes), max_detections)):
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = results.names[cls_id]
            
            detections.append(Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                tier='Tier1-YOLOv8x'
            ))
        
        logger.debug(f"Tier 1: Detected {len(detections)} objects in {inference_time:.1f}ms")
        
        return DetectionResult(
            detections=detections,
            inference_time_ms=inference_time,
            tier_used='Tier1-YOLOv8x',
            image_shape=image.shape
        )
    
    def _detect_tier2(
        self,
        image: np.ndarray,
        confidence_threshold: float,
        max_detections: int
    ) -> DetectionResult:
        """Tier 2: YOLOv8-nano detection (CPU)"""
        start_time = time.time()
        
        # Run inference on CPU
        results = self.tier2_model(image, conf=confidence_threshold, verbose=False)[0]
        
        inference_time = (time.time() - start_time) * 1000
        
        # Parse results
        detections = []
        boxes = results.boxes
        
        for i in range(min(len(boxes), max_detections)):
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = results.names[cls_id]
            
            detections.append(Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                tier='Tier2-YOLOv8n'
            ))
        
        logger.debug(f"Tier 2: Detected {len(detections)} objects in {inference_time:.1f}ms")
        
        return DetectionResult(
            detections=detections,
            inference_time_ms=inference_time,
            tier_used='Tier2-YOLOv8n',
            image_shape=image.shape
        )
    
    def _detect_tier3(
        self,
        image: np.ndarray,
        confidence_threshold: float,
        max_detections: int
    ) -> DetectionResult:
        """Tier 3: Haar Cascade detection (Classical CV)"""
        start_time = time.time()
        
        # Convert to grayscale for Haar cascades
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        detections = []
        detection_count = 0
        
        # Run each cascade
        for cascade_name, cascade in self.tier3_cascades.items():
            if detection_count >= max_detections:
                break
            
            # Detect objects
            objects = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in objects:
                if detection_count >= max_detections:
                    break
                
                detections.append(Detection(
                    class_id=hash(cascade_name) % 1000,  # Pseudo class ID
                    class_name=cascade_name,
                    confidence=0.75,  # Haar cascades don't provide confidence
                    bbox=(x, y, x + w, y + h),
                    tier='Tier3-Haar'
                ))
                detection_count += 1
        
        inference_time = (time.time() - start_time) * 1000
        
        logger.debug(f"Tier 3: Detected {len(detections)} objects in {inference_time:.1f}ms")
        
        return DetectionResult(
            detections=detections,
            inference_time_ms=inference_time,
            tier_used='Tier3-Haar',
            image_shape=image.shape
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get detector status"""
        return {
            'tier1_yolov8x': self.tier1_model is not None,
            'tier2_yolov8n': self.tier2_model is not None,
            'tier3_haar': len(self.tier3_cascades) > 0,
            'device': self.device,
            'haar_cascades': list(self.tier3_cascades.keys())
        }


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("OBJECT DETECTOR - TESTING")
    print("="*80)
    
    # Initialize detector
    detector = ObjectDetector()
    
    # Show status
    status = detector.get_status()
    print(f"\nStatus: {status}\n")
    
    # Create test image (random)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test detection
    print("Testing detection on random image...")
    result = detector.detect(test_image, confidence_threshold=0.25)
    
    print(f"\nDetection Results:")
    print(f"  Tier Used: {result.tier_used}")
    print(f"  Inference Time: {result.inference_time_ms:.1f}ms")
    print(f"  Detections: {len(result.detections)}")
    
    for det in result.detections[:5]:  # Show first 5
        print(f"    - {det.class_name}: {det.confidence:.2f} @ {det.bbox}")
    
    print("\n" + "="*80)

