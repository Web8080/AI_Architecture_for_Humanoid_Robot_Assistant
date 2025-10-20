# Computer Vision Module

Multi-tier fallback architecture for robust visual perception in humanoid robots.

## Overview

The Computer Vision module provides 6 core capabilities with automatic fallback mechanisms:

1. **Object Detection** - Detect and classify objects in images
2. **Segmentation** - Generate pixel-perfect masks for objects
3. **Depth Estimation** - Estimate 3D depth from 2D images
4. **Pose Estimation** - Detect human body keypoints
5. **Face Recognition** - Detect and recognize faces
6. **Scene Understanding** - Analyze and describe scenes

## Architecture

Each component implements a 3-tier fallback system:

- **Tier 1**: Best quality models (GPU-accelerated, heavy)
- **Tier 2**: Fast models (CPU-optimized, lightweight)
- **Tier 3**: Classical CV (always works, rule-based)

### Multi-Tier Fallback Example

```python
from src.vision.object_detection.detector import ObjectDetector

detector = ObjectDetector()
# Automatically tries:
# 1. YOLOv8-x on GPU
# 2. YOLOv8-nano on CPU
# 3. Haar Cascades (classical CV)
# System NEVER fails!

result = detector.detect(image)
print(f"Used: {result.tier_used}")
```

## Components

### 1. Object Detection

**Models:**
- Tier 1: YOLOv8-x (Best accuracy, requires GPU)
- Tier 2: YOLOv8-nano (Fast, CPU-friendly)
- Tier 3: Haar Cascades (Classical CV, always works)

**Usage:**
```python
from src.vision.object_detection.detector import ObjectDetector

detector = ObjectDetector()
result = detector.detect(image, confidence_threshold=0.25)

for detection in result.detections:
    print(f"{detection.class_name}: {detection.confidence:.2f}")
```

### 2. Segmentation

**Models:**
- Tier 1: SAM (Segment Anything Model)
- Tier 2: YOLOv8-seg (Instance segmentation)
- Tier 3: GrabCut/Watershed (Classical CV)

**Usage:**
```python
from src.vision.segmentation.segmenter import Segmenter

segmenter = Segmenter()
result = segmenter.segment(image)

for mask in result.masks:
    print(f"Mask area: {mask.area}, confidence: {mask.confidence}")
```

### 3. Depth Estimation

**Models:**
- Tier 1: MiDaS DPT-Large (Transformer-based)
- Tier 2: MiDaS Small (Lightweight)
- Tier 3: Gradient-based heuristics

**Usage:**
```python
from src.vision.depth.estimator import DepthEstimator

estimator = DepthEstimator()
result = estimator.estimate(image, normalize=True)

depth_map = result.depth_map  # (H, W) float32 array
```

### 4. Pose Estimation

**Models:**
- Tier 1: MediaPipe Pose (Fast & accurate)
- Tier 3: Classical keypoint detection

**Usage:**
```python
from src.vision.pose.estimator import PoseEstimator

estimator = PoseEstimator()
result = estimator.estimate(image)

for keypoint in result.keypoints:
    print(f"{keypoint.name}: ({keypoint.x}, {keypoint.y})")
```

### 5. Face Recognition

**Models:**
- Tier 1: InsightFace (ArcFace embeddings)
- Tier 3: Haar Cascades + face_recognition

**Usage:**
```python
from src.vision.face.recognizer import FaceRecognizer

recognizer = FaceRecognizer()
result = recognizer.recognize(image)

for face in result.faces:
    print(f"Face at {face.bbox}, confidence: {face.confidence}")
```

### 6. Scene Understanding

**Models:**
- Tier 1: CLIP ViT-L (Vision-language understanding)
- Tier 2: CLIP ViT-B (Lightweight)
- Tier 3: Color histograms + heuristics

**Usage:**
```python
from src.vision.scene.analyzer import SceneAnalyzer

analyzer = SceneAnalyzer()
result = analyzer.analyze(image)

for label, conf in result.description.labels:
    print(f"{label}: {conf:.2f}")
```

## Unified Vision Service

Process multiple vision tasks in one pipeline:

```python
from src.vision.vision_service import VisionService, VisionRequest

service = VisionService()

request = VisionRequest(
    image=image,
    detect_objects=True,
    estimate_depth=True,
    detect_pose=True,
    analyze_scene=True
)

response = service.process(request)

print(f"Objects: {len(response.detections.detections)}")
print(f"Depth range: {response.depth.min_depth} to {response.depth.max_depth}")
print(f"Pose detected: {response.pose.detected}")
print(f"Scene: {response.scene.description.labels[0]}")
print(f"Total latency: {response.total_latency_ms}ms")
```

## Configuration

Configure tiers in `configs/base/system_config.yaml`:

```yaml
vision:
  object_detector:
    tier1_enabled: false  # YOLOv8-x (heavy)
    tier2_enabled: true   # YOLOv8-nano (fast)
    tier3_enabled: true   # Haar Cascades (always)
    use_gpu: null  # Auto-detect GPU
    confidence_threshold: 0.25
```

## Testing

Run comprehensive tests:

```bash
python scripts/test_vision_comprehensive.py
```

## Dependencies

### Core (Required)
- opencv-python >= 4.8.1
- numpy >= 1.24.0
- Pillow >= 10.1.0

### Tier 1/2 Models (Optional)
- ultralytics >= 8.0.200 (YOLO)
- segment-anything >= 1.0 (SAM)
- torch >= 2.1.0 (MiDaS, CLIP)
- mediapipe >= 0.10.0 (Pose)
- insightface >= 0.7.0 (Face)
- ftfy, regex, tqdm (CLIP)

### Fallbacks work without these!

## Performance

Typical latencies (CPU-only, Tier 2/3):

| Component | Latency | Notes |
|-----------|---------|-------|
| Object Detection | 50-200ms | YOLOv8-nano or Haar |
| Segmentation | 100-500ms | YOLOv8-seg or classical |
| Depth Estimation | 200-1000ms | MiDaS-small or heuristics |
| Pose Estimation | 20-100ms | MediaPipe or classical |
| Face Recognition | 30-150ms | Haar cascades |
| Scene Analysis | 50-200ms | Classical features |

With GPU (Tier 1): 10-50ms per component

## Error Handling

The system NEVER crashes:

```python
# Even if all tiers fail, returns empty result
result = detector.detect(image)

if result.tier_used == 'None':
    print("All tiers failed, but system still functional")
else:
    print(f"Successfully used: {result.tier_used}")
```

## Contributing

When adding new components:

1. Implement 3-tier fallback
2. Add auto-detection for GPU/CPU
3. Return structured result dataclasses
4. Include get_status() method
5. Add tests to test_vision_comprehensive.py

## License

MIT License

## Author

Victor Ibhafidon
October 2025

