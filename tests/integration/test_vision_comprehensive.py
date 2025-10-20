"""
Comprehensive Test Suite for Computer Vision Module

Tests all CV components with multi-tier fallback validation

Author: Victor Ibhafidon
Date: October 2025
"""

import sys
from pathlib import Path
# Add src to path (tests/integration/ -> project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import cv2
import logging
from typing import Dict, Any

from src.vision.object_detection.detector import ObjectDetector
from src.vision.segmentation.segmenter import Segmenter
from src.vision.depth.estimator import DepthEstimator
from src.vision.pose.estimator import PoseEstimator
from src.vision.face.recognizer import FaceRecognizer
from src.vision.scene.analyzer import SceneAnalyzer
from src.vision.vision_service import VisionService, VisionRequest

logging.basicConfig(level=logging.WARNING)


def create_test_image(height=480, width=640) -> np.ndarray:
    """Create a test image with some patterns"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some colored rectangles
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue
    cv2.rectangle(image, (300, 200), (400, 300), (0, 255, 0), -1)  # Green
    cv2.circle(image, (320, 240), 50, (0, 0, 255), -1)  # Red circle
    
    return image


def test_object_detector() -> bool:
    """Test Object Detector"""
    print("\n" + "="*80)
    print("TESTING: Object Detector")
    print("="*80)
    
    try:
        detector = ObjectDetector()
        status = detector.get_status()
        
        print(f"\nTier Status:")
        print(f"  Tier 1 (YOLOv8-x): {'READY' if status['tier1_yolov8x'] else 'NOT AVAILABLE'}")
        print(f"  Tier 2 (YOLOv8-n): {'READY' if status['tier2_yolov8n'] else 'NOT AVAILABLE'}")
        print(f"  Tier 3 (Haar):     {'READY' if status['tier3_haar'] else 'NOT AVAILABLE'}")
        print(f"  Device: {status['device']}")
        
        # Test detection
        test_image = create_test_image()
        result = detector.detect(test_image, confidence_threshold=0.25)
        
        print(f"\nDetection Results:")
        print(f"  Tier Used: {result.tier_used}")
        print(f"  Inference Time: {result.inference_time_ms:.1f}ms")
        print(f"  Detections: {len(result.detections)}")
        
        print(f"\nOBJECT DETECTOR: PASS")
        return True
        
    except Exception as e:
        print(f"\nOBJECT DETECTOR: FAIL - {e}")
        return False


def test_segmenter() -> bool:
    """Test Segmenter"""
    print("\n" + "="*80)
    print("TESTING: Segmenter")
    print("="*80)
    
    try:
        segmenter = Segmenter()
        status = segmenter.get_status()
        
        print(f"\nTier Status:")
        print(f"  Tier 1 (SAM):        {'READY' if status['tier1_sam'] else 'NOT AVAILABLE'}")
        print(f"  Tier 2 (YOLOv8-seg): {'READY' if status['tier2_yolov8seg'] else 'NOT AVAILABLE'}")
        print(f"  Tier 3 (Classical):  {'READY' if status['tier3_classical'] else 'NOT AVAILABLE'}")
        
        # Test segmentation
        test_image = create_test_image()
        result = segmenter.segment(test_image, confidence_threshold=0.25)
        
        print(f"\nSegmentation Results:")
        print(f"  Tier Used: {result.tier_used}")
        print(f"  Inference Time: {result.inference_time_ms:.1f}ms")
        print(f"  Masks: {len(result.masks)}")
        
        print(f"\nSEGMENTER: PASS")
        return True
        
    except Exception as e:
        print(f"\nSEGMENTER: FAIL - {e}")
        return False


def test_depth_estimator() -> bool:
    """Test Depth Estimator"""
    print("\n" + "="*80)
    print("TESTING: Depth Estimator")
    print("="*80)
    
    try:
        estimator = DepthEstimator()
        status = estimator.get_status()
        
        print(f"\nTier Status:")
        print(f"  Tier 1 (MiDaS-DPT):  {'READY' if status['tier1_midas_dpt'] else 'NOT AVAILABLE'}")
        print(f"  Tier 2 (MiDaS-Small): {'READY' if status['tier2_midas_small'] else 'NOT AVAILABLE'}")
        print(f"  Tier 3 (Classical):   {'READY' if status['tier3_classical'] else 'NOT AVAILABLE'}")
        
        # Test depth estimation
        test_image = create_test_image()
        result = estimator.estimate(test_image, normalize=True)
        
        print(f"\nDepth Estimation Results:")
        print(f"  Tier Used: {result.tier_used}")
        print(f"  Inference Time: {result.inference_time_ms:.1f}ms")
        print(f"  Depth Map Shape: {result.depth_map.shape}")
        print(f"  Min Depth: {result.min_depth:.3f}")
        print(f"  Max Depth: {result.max_depth:.3f}")
        
        print(f"\nDEPTH ESTIMATOR: PASS")
        return True
        
    except Exception as e:
        print(f"\nDEPTH ESTIMATOR: FAIL - {e}")
        return False


def test_pose_estimator() -> bool:
    """Test Pose Estimator"""
    print("\n" + "="*80)
    print("TESTING: Pose Estimator")
    print("="*80)
    
    try:
        estimator = PoseEstimator()
        status = estimator.get_status()
        
        print(f"\nTier Status:")
        print(f"  Tier 1 (MediaPipe): {'READY' if status['tier1_mediapipe'] else 'NOT AVAILABLE'}")
        print(f"  Tier 3 (Classical): {'READY' if status['tier3_classical'] else 'NOT AVAILABLE'}")
        
        # Test pose estimation
        test_image = create_test_image()
        result = estimator.estimate(test_image)
        
        print(f"\nPose Estimation Results:")
        print(f"  Tier Used: {result.tier_used}")
        print(f"  Inference Time: {result.inference_time_ms:.1f}ms")
        print(f"  Detected: {result.detected}")
        print(f"  Keypoints: {len(result.keypoints)}")
        
        print(f"\nPOSE ESTIMATOR: PASS")
        return True
        
    except Exception as e:
        print(f"\nPOSE ESTIMATOR: FAIL - {e}")
        return False


def test_face_recognizer() -> bool:
    """Test Face Recognizer"""
    print("\n" + "="*80)
    print("TESTING: Face Recognizer")
    print("="*80)
    
    try:
        recognizer = FaceRecognizer()
        status = recognizer.get_status()
        
        print(f"\nTier Status:")
        print(f"  Tier 1 (InsightFace): {'READY' if status['tier1_insightface'] else 'NOT AVAILABLE'}")
        print(f"  Tier 3 (Haar):        {'READY' if status['tier3_haar'] else 'NOT AVAILABLE'}")
        
        # Test face recognition
        test_image = create_test_image()
        result = recognizer.recognize(test_image)
        
        print(f"\nFace Recognition Results:")
        print(f"  Tier Used: {result.tier_used}")
        print(f"  Inference Time: {result.inference_time_ms:.1f}ms")
        print(f"  Faces Detected: {len(result.faces)}")
        
        print(f"\nFACE RECOGNIZER: PASS")
        return True
        
    except Exception as e:
        print(f"\nFACE RECOGNIZER: FAIL - {e}")
        return False


def test_scene_analyzer() -> bool:
    """Test Scene Analyzer"""
    print("\n" + "="*80)
    print("TESTING: Scene Analyzer")
    print("="*80)
    
    try:
        analyzer = SceneAnalyzer()
        status = analyzer.get_status()
        
        print(f"\nTier Status:")
        print(f"  Tier 1 (CLIP-L):    {'READY' if status['tier1_clip_l'] else 'NOT AVAILABLE'}")
        print(f"  Tier 2 (CLIP-B):    {'READY' if status['tier2_clip_b'] else 'NOT AVAILABLE'}")
        print(f"  Tier 3 (Classical): {'READY' if status['tier3_classical'] else 'NOT AVAILABLE'}")
        
        # Test scene analysis
        test_image = create_test_image()
        result = analyzer.analyze(test_image)
        
        print(f"\nScene Analysis Results:")
        print(f"  Tier Used: {result.tier_used}")
        print(f"  Inference Time: {result.inference_time_ms:.1f}ms")
        print(f"  Labels: {result.description.labels[:3]}")
        print(f"  Brightness: {result.description.brightness:.2f}")
        
        print(f"\nSCENE ANALYZER: PASS")
        return True
        
    except Exception as e:
        print(f"\nSCENE ANALYZER: FAIL - {e}")
        return False


def test_vision_service() -> bool:
    """Test Unified Vision Service"""
    print("\n" + "="*80)
    print("TESTING: Full Vision Service Integration")
    print("="*80)
    
    try:
        service = VisionService()
        
        # Show component status
        status = service.get_component_status()
        print(f"\nComponents Ready:")
        for component, ready in status.items():
            icon = "READY" if ready else "NOT AVAILABLE"
            print(f"  {component}: {icon}")
        
        # Test full pipeline
        test_image = create_test_image()
        
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
        
        print(f"\nFull Pipeline Results:")
        print(f"  Total Latency: {response.total_latency_ms:.1f}ms")
        print(f"\n  Tiers Used:")
        for component, tier in response.tiers_used.items():
            print(f"    {component}: {tier}")
        
        print(f"\nVISION SERVICE: PASS")
        return True
        
    except Exception as e:
        print(f"\nVISION SERVICE: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("COMPUTER VISION MODULE - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    tests = [
        ("Object Detector", test_object_detector),
        ("Segmenter", test_segmenter),
        ("Depth Estimator", test_depth_estimator),
        ("Pose Estimator", test_pose_estimator),
        ("Face Recognizer", test_face_recognizer),
        ("Scene Analyzer", test_scene_analyzer),
        ("Full Vision Service", test_vision_service),
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        icon = "  " if passed else "  "
        print(f"  {icon} {name}: {status}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    pass_rate = (passed_count / total_count) * 100
    
    print(f"\n{passed_count}/{total_count} tests passed ({pass_rate:.0f}%)")
    
    if passed_count == total_count:
        print("\nAll tests passed! Computer Vision module is ready.")
    else:
        print(f"\n{total_count - passed_count} test(s) failed. Check logs above.")
    
    print("\n" + "="*80)
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

