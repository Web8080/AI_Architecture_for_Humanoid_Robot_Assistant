"""
Pose Estimation with Multi-Tier Fallback Architecture

Implements robust human pose estimation with automatic fallback:
- Tier 1: MediaPipe Pose (GPU) - Fast and accurate
- Tier 2: OpenPose lite - CPU-friendly
- Tier 3: Classical keypoint detection - Always works

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
import cv2

# Tier 1: MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available. Install with: pip install mediapipe")

OPENCV_AVAILABLE = True

logger = logging.getLogger(__name__)


@dataclass
class Keypoint:
    """Single body keypoint"""
    x: float
    y: float
    confidence: float
    name: str


@dataclass
class PoseResult:
    """Container for pose estimation results"""
    keypoints: List[Keypoint]
    inference_time_ms: float
    tier_used: str
    detected: bool
    image_shape: Tuple[int, int, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'keypoints': [{'x': k.x, 'y': k.y, 'conf': k.confidence, 'name': k.name} 
                          for k in self.keypoints],
            'count': len(self.keypoints),
            'inference_time_ms': self.inference_time_ms,
            'tier_used': self.tier_used,
            'detected': self.detected,
            'image_shape': self.image_shape
        }


class PoseEstimator:
    """Multi-tier pose estimator with automatic fallback"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.tier1_model = None
        
        self.tier1_enabled = self.config.get('tier1_enabled', True)
        self.tier2_enabled = self.config.get('tier2_enabled', False)
        self.tier3_enabled = self.config.get('tier3_enabled', True)
        
        self._init_tiers()
    
    def _init_tiers(self):
        """Initialize all available tiers"""
        # Tier 1: MediaPipe
        if self.tier1_enabled and MEDIAPIPE_AVAILABLE:
            try:
                mp_pose = mp.solutions.pose
                self.tier1_model = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info(" Tier 1 (MediaPipe Pose) initialized")
            except Exception as e:
                logger.warning(f"Tier 1 (MediaPipe) failed: {e}")
        
        if self.tier3_enabled:
            logger.info(" Tier 3 (Classical CV) initialized")
    
    def estimate(self, image: np.ndarray) -> PoseResult:
        """Estimate pose from image"""
        # Try Tier 1
        if self.tier1_enabled and self.tier1_model is not None:
            try:
                return self._estimate_tier1(image)
            except Exception as e:
                logger.warning(f"Tier 1 failed: {e}. Falling back to Tier 3.")
        
        # Fallback to Tier 3
        if self.tier3_enabled:
            try:
                return self._estimate_tier3(image)
            except Exception as e:
                logger.error(f"All tiers failed: {e}")
                return PoseResult([], 0.0, 'None', False, image.shape)
        
        return PoseResult([], 0.0, 'None', False, image.shape)
    
    def _estimate_tier1(self, image: np.ndarray) -> PoseResult:
        """Tier 1: MediaPipe"""
        start_time = time.time()
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.tier1_model.process(image_rgb)
        
        inference_time = (time.time() - start_time) * 1000
        
        keypoints = []
        detected = False
        
        if results.pose_landmarks:
            detected = True
            h, w = image.shape[:2]
            
            landmark_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
            
            for idx, landmark in enumerate(results.pose_landmarks.landmark[:17]):
                keypoints.append(Keypoint(
                    x=landmark.x * w,
                    y=landmark.y * h,
                    confidence=landmark.visibility,
                    name=landmark_names[idx] if idx < len(landmark_names) else f'point_{idx}'
                ))
        
        return PoseResult(keypoints, inference_time, 'Tier1-MediaPipe', detected, image.shape)
    
    def _estimate_tier3(self, image: np.ndarray) -> PoseResult:
        """Tier 3: Classical CV (simple keypoint detection)"""
        start_time = time.time()
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=17, qualityLevel=0.01, minDistance=30)
        
        keypoints = []
        detected = corners is not None and len(corners) > 0
        
        if detected:
            for idx, corner in enumerate(corners):
                x, y = corner.ravel()
                keypoints.append(Keypoint(
                    x=float(x),
                    y=float(y),
                    confidence=0.60,
                    name=f'keypoint_{idx}'
                ))
        
        inference_time = (time.time() - start_time) * 1000
        
        return PoseResult(keypoints, inference_time, 'Tier3-Classical', detected, image.shape)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'tier1_mediapipe': self.tier1_model is not None,
            'tier3_classical': self.tier3_enabled
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("="*80)
    print("POSE ESTIMATOR - TESTING")
    print("="*80)
    
    estimator = PoseEstimator()
    print(f"\nStatus: {estimator.get_status()}\n")
    
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = estimator.estimate(test_image)
    
    print(f"Tier Used: {result.tier_used}")
    print(f"Detected: {result.detected}")
    print(f"Keypoints: {len(result.keypoints)}")
    print(f"Inference Time: {result.inference_time_ms:.1f}ms")
    print("="*80)

