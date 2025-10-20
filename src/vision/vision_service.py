"""
Unified Vision Service - Orchestrates all Computer Vision components

Integrates all CV modules with multi-tier fallback:
- Object Detection
- Segmentation
- Depth Estimation
- Pose Estimation  
- Face Recognition
- Scene Understanding

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np

from .object_detection.detector import ObjectDetector, DetectionResult
from .segmentation.segmenter import Segmenter, SegmentationResult
from .depth.estimator import DepthEstimator, DepthResult
from .pose.estimator import PoseEstimator, PoseResult
from .face.recognizer import FaceRecognizer, FaceRecognitionResult
from .scene.analyzer import SceneAnalyzer, SceneResult

logger = logging.getLogger(__name__)


@dataclass
class VisionRequest:
    """Request for vision processing"""
    image: np.ndarray
    detect_objects: bool = True
    segment: bool = False
    estimate_depth: bool = False
    detect_pose: bool = False
    recognize_faces: bool = False
    analyze_scene: bool = False
    confidence_threshold: float = 0.25


@dataclass
class VisionResponse:
    """Comprehensive vision processing response"""
    # Results from each component
    detections: Optional[DetectionResult] = None
    segmentation: Optional[SegmentationResult] = None
    depth: Optional[DepthResult] = None
    pose: Optional[PoseResult] = None
    faces: Optional[FaceRecognitionResult] = None
    scene: Optional[SceneResult] = None
    
    # Metadata
    total_latency_ms: float = 0.0
    tiers_used: Dict[str, str] = None
    components_status: Dict[str, bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {}
        
        if self.detections:
            result['detections'] = self.detections.to_dict()
        if self.segmentation:
            result['segmentation'] = self.segmentation.to_dict()
        if self.depth:
            result['depth'] = self.depth.to_dict()
        if self.pose:
            result['pose'] = self.pose.to_dict()
        if self.faces:
            result['faces'] = self.faces.to_dict()
        if self.scene:
            result['scene'] = self.scene.to_dict()
        
        result['total_latency_ms'] = self.total_latency_ms
        result['tiers_used'] = self.tiers_used or {}
        result['components_status'] = self.components_status or {}
        
        return result


class VisionService:
    """
    Unified Vision Service with multi-tier fallback
    
    Orchestrates all CV components for comprehensive visual understanding
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Vision Service
        
        Args:
            config: Configuration dict for all CV components
        """
        self.config = config or {}
        
        # Initialize all components
        logger.info("Initializing Vision Service components...")
        
        object_detector_config = self.config.get('object_detector', {})
        self.object_detector = ObjectDetector(object_detector_config)
        
        segmenter_config = self.config.get('segmenter', {})
        self.segmenter = Segmenter(segmenter_config)
        
        depth_estimator_config = self.config.get('depth_estimator', {})
        self.depth_estimator = DepthEstimator(depth_estimator_config)
        
        pose_estimator_config = self.config.get('pose_estimator', {})
        self.pose_estimator = PoseEstimator(pose_estimator_config)
        
        face_recognizer_config = self.config.get('face_recognizer', {})
        self.face_recognizer = FaceRecognizer(face_recognizer_config)
        
        scene_analyzer_config = self.config.get('scene_analyzer', {})
        self.scene_analyzer = SceneAnalyzer(scene_analyzer_config)
        
        logger.info(" Vision Service initialized with 6 components")
    
    def process(self, request: VisionRequest) -> VisionResponse:
        """
        Process image through requested CV pipeline
        
        Args:
            request: VisionRequest specifying what to process
            
        Returns:
            VisionResponse with results from all requested components
        """
        start_time = time.time()
        
        response = VisionResponse()
        response.tiers_used = {}
        response.components_status = {}
        
        # Object Detection
        if request.detect_objects:
            try:
                response.detections = self.object_detector.detect(
                    request.image,
                    confidence_threshold=request.confidence_threshold
                )
                response.tiers_used['object_detection'] = response.detections.tier_used
                response.components_status['object_detection'] = True
                logger.debug(f"Object detection: {len(response.detections.detections)} objects found")
            except Exception as e:
                logger.error(f"Object detection failed: {e}")
                response.components_status['object_detection'] = False
        
        # Segmentation
        if request.segment:
            try:
                response.segmentation = self.segmenter.segment(
                    request.image,
                    confidence_threshold=request.confidence_threshold
                )
                response.tiers_used['segmentation'] = response.segmentation.tier_used
                response.components_status['segmentation'] = True
                logger.debug(f"Segmentation: {len(response.segmentation.masks)} masks generated")
            except Exception as e:
                logger.error(f"Segmentation failed: {e}")
                response.components_status['segmentation'] = False
        
        # Depth Estimation
        if request.estimate_depth:
            try:
                response.depth = self.depth_estimator.estimate(request.image)
                response.tiers_used['depth_estimation'] = response.depth.tier_used
                response.components_status['depth_estimation'] = True
                logger.debug(f"Depth estimation: range {response.depth.min_depth:.2f} to {response.depth.max_depth:.2f}")
            except Exception as e:
                logger.error(f"Depth estimation failed: {e}")
                response.components_status['depth_estimation'] = False
        
        # Pose Estimation
        if request.detect_pose:
            try:
                response.pose = self.pose_estimator.estimate(request.image)
                response.tiers_used['pose_estimation'] = response.pose.tier_used
                response.components_status['pose_estimation'] = True
                logger.debug(f"Pose estimation: {len(response.pose.keypoints)} keypoints detected")
            except Exception as e:
                logger.error(f"Pose estimation failed: {e}")
                response.components_status['pose_estimation'] = False
        
        # Face Recognition
        if request.recognize_faces:
            try:
                response.faces = self.face_recognizer.recognize(request.image)
                response.tiers_used['face_recognition'] = response.faces.tier_used
                response.components_status['face_recognition'] = True
                logger.debug(f"Face recognition: {len(response.faces.faces)} faces found")
            except Exception as e:
                logger.error(f"Face recognition failed: {e}")
                response.components_status['face_recognition'] = False
        
        # Scene Analysis
        if request.analyze_scene:
            try:
                response.scene = self.scene_analyzer.analyze(request.image)
                response.tiers_used['scene_analysis'] = response.scene.tier_used
                response.components_status['scene_analysis'] = True
                logger.debug(f"Scene analysis: {response.scene.description.labels[:2]}")
            except Exception as e:
                logger.error(f"Scene analysis failed: {e}")
                response.components_status['scene_analysis'] = False
        
        # Calculate total latency
        response.total_latency_ms = (time.time() - start_time) * 1000
        
        return response
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get status of all components"""
        return {
            'object_detector': self.object_detector.get_status()['tier2_yolov8n'] or 
                              self.object_detector.get_status()['tier3_haar'],
            'segmenter': self.segmenter.get_status()['tier2_yolov8seg'] or 
                        self.segmenter.get_status()['tier3_classical'],
            'depth_estimator': self.depth_estimator.get_status()['tier2_midas_small'] or 
                              self.depth_estimator.get_status()['tier3_classical'],
            'pose_estimator': self.pose_estimator.get_status()['tier1_mediapipe'] or 
                             self.pose_estimator.get_status()['tier3_classical'],
            'face_recognizer': self.face_recognizer.get_status()['tier3_haar'],
            'scene_analyzer': self.scene_analyzer.get_status()['tier3_classical']
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status of all components"""
        return {
            'object_detector': self.object_detector.get_status(),
            'segmenter': self.segmenter.get_status(),
            'depth_estimator': self.depth_estimator.get_status(),
            'pose_estimator': self.pose_estimator.get_status(),
            'face_recognizer': self.face_recognizer.get_status(),
            'scene_analyzer': self.scene_analyzer.get_status()
        }


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("VISION SERVICE - TESTING")
    print("="*80)
    
    # Initialize service
    service = VisionService()
    
    # Show status
    status = service.get_component_status()
    print(f"\nComponent Status:")
    for component, ready in status.items():
        icon = "" if ready else ""
        print(f"  {icon} {component}")
    
    print("\n" + "-"*80)
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test full pipeline
    print("\nTesting full vision pipeline...")
    
    request = VisionRequest(
        image=test_image,
        detect_objects=True,
        segment=False,  # Skip for speed
        estimate_depth=True,
        detect_pose=True,
        recognize_faces=True,
        analyze_scene=True
    )
    
    response = service.process(request)
    
    print(f"\nResults:")
    print(f"  Total Latency: {response.total_latency_ms:.1f}ms")
    print(f"\n  Tiers Used:")
    for component, tier in response.tiers_used.items():
        print(f"    - {component}: {tier}")
    
    print(f"\n  Component Results:")
    if response.detections:
        print(f"    - Objects: {len(response.detections.detections)} detected")
    if response.depth:
        print(f"    - Depth: {response.depth.depth_map.shape}")
    if response.pose:
        print(f"    - Pose: {len(response.pose.keypoints)} keypoints")
    if response.faces:
        print(f"    - Faces: {len(response.faces.faces)} detected")
    if response.scene:
        print(f"    - Scene: {response.scene.description.labels[:2]}")
    
    print("\n" + "="*80)

