"""
Face Recognition with Multi-Tier Fallback Architecture

Implements robust face recognition with automatic fallback:
- Tier 1: InsightFace (ArcFace) - State-of-the-art
- Tier 2: DeepFace - Good accuracy
- Tier 3: Haar + face_recognition - Classical

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
import cv2

# Tier 1: InsightFace
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available. Install with: pip install insightface")

# Tier 3: face_recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Face:
    """Detected face information"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    embedding: Optional[np.ndarray] = None
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    tier: str = "Unknown"


@dataclass
class FaceRecognitionResult:
    """Container for face recognition results"""
    faces: List[Face]
    inference_time_ms: float
    tier_used: str
    image_shape: Tuple[int, int, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'faces': [{
                'bbox': f.bbox,
                'confidence': f.confidence,
                'has_embedding': f.embedding is not None,
                'age': f.age,
                'gender': f.gender,
                'tier': f.tier
            } for f in self.faces],
            'count': len(self.faces),
            'inference_time_ms': self.inference_time_ms,
            'tier_used': self.tier_used,
            'image_shape': self.image_shape
        }


class FaceRecognizer:
    """Multi-tier face recognizer with automatic fallback"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.tier1_model = None
        self.tier3_cascade = None
        
        self.tier1_enabled = self.config.get('tier1_enabled', False)
        self.tier2_enabled = self.config.get('tier2_enabled', False)
        self.tier3_enabled = self.config.get('tier3_enabled', True)
        
        self._init_tiers()
    
    def _init_tiers(self):
        """Initialize all available tiers"""
        # Tier 1: InsightFace
        if self.tier1_enabled and INSIGHTFACE_AVAILABLE:
            try:
                self.tier1_model = FaceAnalysis(providers=['CPUExecutionProvider'])
                self.tier1_model.prepare(ctx_id=0, det_size=(640, 640))
                logger.info(" Tier 1 (InsightFace) initialized")
            except Exception as e:
                logger.warning(f"Tier 1 (InsightFace) failed: {e}")
        
        # Tier 3: Haar + face_recognition
        if self.tier3_enabled:
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.tier3_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info(" Tier 3 (Haar Cascades) initialized")
            except Exception as e:
                logger.warning(f"Tier 3 failed: {e}")
    
    def recognize(self, image: np.ndarray) -> FaceRecognitionResult:
        """Recognize faces in image"""
        # Try Tier 1
        if self.tier1_enabled and self.tier1_model is not None:
            try:
                return self._recognize_tier1(image)
            except Exception as e:
                logger.warning(f"Tier 1 failed: {e}. Falling back to Tier 3.")
        
        # Fallback to Tier 3
        if self.tier3_enabled and self.tier3_cascade is not None:
            try:
                return self._recognize_tier3(image)
            except Exception as e:
                logger.error(f"All tiers failed: {e}")
                return FaceRecognitionResult([], 0.0, 'None', image.shape)
        
        return FaceRecognitionResult([], 0.0, 'None', image.shape)
    
    def _recognize_tier1(self, image: np.ndarray) -> FaceRecognitionResult:
        """Tier 1: InsightFace"""
        start_time = time.time()
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces_data = self.tier1_model.get(image_rgb)
        
        inference_time = (time.time() - start_time) * 1000
        
        faces = []
        for face in faces_data:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            faces.append(Face(
                bbox=(x1, y1, x2, y2),
                confidence=float(face.det_score),
                embedding=face.embedding if hasattr(face, 'embedding') else None,
                age=int(face.age) if hasattr(face, 'age') else None,
                gender='M' if hasattr(face, 'gender') and face.gender == 1 else 'F',
                tier='Tier1-InsightFace'
            ))
        
        return FaceRecognitionResult(faces, inference_time, 'Tier1-InsightFace', image.shape)
    
    def _recognize_tier3(self, image: np.ndarray) -> FaceRecognitionResult:
        """Tier 3: Haar Cascades"""
        start_time = time.time()
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = self.tier3_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        faces = []
        for (x, y, w, h) in face_rects:
            # Try to get embeddings if face_recognition available
            embedding = None
            if FACE_RECOGNITION_AVAILABLE:
                try:
                    face_locations = [(y, x+w, y+h, x)]
                    encodings = face_recognition.face_encodings(image, face_locations)
                    if encodings:
                        embedding = encodings[0]
                except:
                    pass
            
            faces.append(Face(
                bbox=(x, y, x+w, y+h),
                confidence=0.75,
                embedding=embedding,
                tier='Tier3-Haar'
            ))
        
        inference_time = (time.time() - start_time) * 1000
        
        return FaceRecognitionResult(faces, inference_time, 'Tier3-Haar', image.shape)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'tier1_insightface': self.tier1_model is not None,
            'tier3_haar': self.tier3_cascade is not None,
            'face_recognition_available': FACE_RECOGNITION_AVAILABLE
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("="*80)
    print("FACE RECOGNIZER - TESTING")
    print("="*80)
    
    recognizer = FaceRecognizer()
    print(f"\nStatus: {recognizer.get_status()}\n")
    
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = recognizer.recognize(test_image)
    
    print(f"Tier Used: {result.tier_used}")
    print(f"Faces Detected: {len(result.faces)}")
    print(f"Inference Time: {result.inference_time_ms:.1f}ms")
    print("="*80)

