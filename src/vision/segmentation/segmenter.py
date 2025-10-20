"""
Segmentation with Multi-Tier Fallback Architecture

Implements robust image segmentation with automatic fallback:
- Tier 1: SAM (Segment Anything Model) - State-of-the-art
- Tier 2: YOLOv8-seg - Fast instance segmentation
- Tier 3: GrabCut / Watershed - Classical CV

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

# Tier 1: SAM (Segment Anything)
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logging.warning("SAM not available. Install with: pip install segment-anything")

# Tier 2: YOLOv8-seg
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Tier 3: OpenCV for classical segmentation
OPENCV_AVAILABLE = True

logger = logging.getLogger(__name__)


@dataclass
class SegmentMask:
    """Single segmentation mask"""
    mask: np.ndarray  # Binary mask
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    area: int
    label: Optional[str] = None
    tier: str = "Unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'area': self.area,
            'label': self.label,
            'tier': self.tier,
            'mask_shape': self.mask.shape
        }


@dataclass
class SegmentationResult:
    """Container for segmentation results"""
    masks: List[SegmentMask]
    inference_time_ms: float
    tier_used: str
    image_shape: Tuple[int, int, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'masks': [m.to_dict() for m in self.masks],
            'count': len(self.masks),
            'inference_time_ms': self.inference_time_ms,
            'tier_used': self.tier_used,
            'image_shape': self.image_shape
        }


class Segmenter:
    """
    Multi-tier segmentation with automatic fallback
    
    Tier 1: SAM (Segment Anything) - Best quality
    Tier 2: YOLOv8-seg - Fast instance segmentation
    Tier 3: GrabCut/Watershed - Classical CV
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize segmenter with multi-tier fallback
        
        Args:
            config: Configuration dict with tier settings
        """
        self.config = config or {}
        self.device = self._detect_device()
        
        # Initialize tiers
        self.tier1_model = None
        self.tier2_model = None
        
        self.tier1_enabled = self.config.get('tier1_enabled', False)  # SAM is heavy
        self.tier2_enabled = self.config.get('tier2_enabled', True)
        self.tier3_enabled = self.config.get('tier3_enabled', True)
        
        self._init_tiers()
    
    def _detect_device(self) -> str:
        """Auto-detect best available device"""
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
        # Tier 1: SAM
        if self.tier1_enabled and SAM_AVAILABLE:
            try:
                model_type = self.config.get('tier1_model_type', 'vit_h')
                checkpoint = self.config.get('tier1_checkpoint', 'sam_vit_h.pth')
                
                logger.info(f"Loading Tier 1 SAM model: {model_type}")
                sam = sam_model_registry[model_type](checkpoint=checkpoint)
                sam.to(device=self.device)
                self.tier1_model = SamPredictor(sam)
                logger.info(f" Tier 1 (SAM) initialized on {self.device}")
            except Exception as e:
                logger.warning(f"Tier 1 (SAM) failed to load: {e}")
        
        # Tier 2: YOLOv8-seg
        if self.tier2_enabled and YOLO_AVAILABLE:
            try:
                model_path = self.config.get('tier2_model', 'yolov8n-seg.pt')
                logger.info(f"Loading Tier 2 model: {model_path}")
                self.tier2_model = YOLO(model_path)
                self.tier2_model.to('cpu')
                logger.info(" Tier 2 (YOLOv8-seg) initialized")
            except Exception as e:
                logger.warning(f"Tier 2 (YOLOv8-seg) failed to load: {e}")
        
        # Tier 3: Classical CV (always available)
        if self.tier3_enabled:
            logger.info(" Tier 3 (GrabCut/Watershed) initialized")
    
    def segment(
        self,
        image: np.ndarray,
        prompts: Optional[List[Tuple[int, int]]] = None,
        confidence_threshold: float = 0.25
    ) -> SegmentationResult:
        """
        Segment image using best available tier
        
        Args:
            image: Input image (BGR format)
            prompts: Optional list of (x, y) points for prompted segmentation
            confidence_threshold: Minimum confidence for segments
            
        Returns:
            SegmentationResult with masks
        """
        # Try Tier 1 (SAM)
        if self.tier1_enabled and self.tier1_model is not None:
            try:
                return self._segment_tier1(image, prompts, confidence_threshold)
            except Exception as e:
                logger.warning(f"Tier 1 segmentation failed: {e}. Falling back to Tier 2.")
        
        # Try Tier 2 (YOLOv8-seg)
        if self.tier2_enabled and self.tier2_model is not None:
            try:
                return self._segment_tier2(image, confidence_threshold)
            except Exception as e:
                logger.warning(f"Tier 2 segmentation failed: {e}. Falling back to Tier 3.")
        
        # Fallback to Tier 3 (Classical CV)
        if self.tier3_enabled:
            try:
                return self._segment_tier3(image, prompts)
            except Exception as e:
                logger.error(f"All segmentation tiers failed: {e}")
                return SegmentationResult(
                    masks=[],
                    inference_time_ms=0.0,
                    tier_used='None',
                    image_shape=image.shape
                )
        
        return SegmentationResult(
            masks=[],
            inference_time_ms=0.0,
            tier_used='None',
            image_shape=image.shape
        )
    
    def _segment_tier1(
        self,
        image: np.ndarray,
        prompts: Optional[List[Tuple[int, int]]],
        confidence_threshold: float
    ) -> SegmentationResult:
        """Tier 1: SAM segmentation"""
        start_time = time.time()
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image
        self.tier1_model.set_image(image_rgb)
        
        masks_list = []
        
        if prompts:
            # Prompted segmentation
            for point in prompts:
                point_coords = np.array([point])
                point_labels = np.array([1])  # 1 = foreground point
                
                masks, scores, _ = self.tier1_model.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False
                )
                
                if scores[0] >= confidence_threshold:
                    mask = masks[0]
                    bbox = self._mask_to_bbox(mask)
                    
                    masks_list.append(SegmentMask(
                        mask=mask,
                        bbox=bbox,
                        confidence=float(scores[0]),
                        area=int(mask.sum()),
                        tier='Tier1-SAM'
                    ))
        else:
            # Auto segmentation (grid prompts)
            h, w = image.shape[:2]
            grid_size = 4
            for i in range(grid_size):
                for j in range(grid_size):
                    x = int((j + 0.5) * w / grid_size)
                    y = int((i + 0.5) * h / grid_size)
                    
                    point_coords = np.array([[x, y]])
                    point_labels = np.array([1])
                    
                    masks, scores, _ = self.tier1_model.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=False
                    )
                    
                    if scores[0] >= confidence_threshold:
                        mask = masks[0]
                        bbox = self._mask_to_bbox(mask)
                        
                        masks_list.append(SegmentMask(
                            mask=mask,
                            bbox=bbox,
                            confidence=float(scores[0]),
                            area=int(mask.sum()),
                            tier='Tier1-SAM'
                        ))
        
        inference_time = (time.time() - start_time) * 1000
        
        logger.debug(f"Tier 1: Generated {len(masks_list)} masks in {inference_time:.1f}ms")
        
        return SegmentationResult(
            masks=masks_list,
            inference_time_ms=inference_time,
            tier_used='Tier1-SAM',
            image_shape=image.shape
        )
    
    def _segment_tier2(
        self,
        image: np.ndarray,
        confidence_threshold: float
    ) -> SegmentationResult:
        """Tier 2: YOLOv8-seg"""
        start_time = time.time()
        
        # Run inference
        results = self.tier2_model(image, conf=confidence_threshold, verbose=False)[0]
        
        inference_time = (time.time() - start_time) * 1000
        
        masks_list = []
        
        if results.masks is not None:
            for i, (mask, box) in enumerate(zip(results.masks.data, results.boxes)):
                mask_np = mask.cpu().numpy().astype(np.uint8)
                
                # Resize mask to original image size
                mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = results.names[cls_id]
                
                masks_list.append(SegmentMask(
                    mask=mask_resized,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    area=int(mask_resized.sum()),
                    label=label,
                    tier='Tier2-YOLOv8seg'
                ))
        
        logger.debug(f"Tier 2: Generated {len(masks_list)} masks in {inference_time:.1f}ms")
        
        return SegmentationResult(
            masks=masks_list,
            inference_time_ms=inference_time,
            tier_used='Tier2-YOLOv8seg',
            image_shape=image.shape
        )
    
    def _segment_tier3(
        self,
        image: np.ndarray,
        prompts: Optional[List[Tuple[int, int]]]
    ) -> SegmentationResult:
        """Tier 3: GrabCut/Watershed segmentation"""
        start_time = time.time()
        
        masks_list = []
        
        if prompts and len(prompts) > 0:
            # Use GrabCut around prompts
            for point in prompts:
                x, y = point
                
                # Create bounding box around point
                margin = 50
                h, w = image.shape[:2]
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w, x + margin)
                y2 = min(h, y + margin)
                
                rect = (x1, y1, x2 - x1, y2 - y1)
                
                # Initialize masks
                mask = np.zeros(image.shape[:2], np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                # Run GrabCut
                try:
                    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                    
                    # Extract foreground
                    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                    
                    if mask2.sum() > 100:  # Minimum area threshold
                        bbox = self._mask_to_bbox(mask2)
                        
                        masks_list.append(SegmentMask(
                            mask=mask2,
                            bbox=bbox,
                            confidence=0.70,  # Fixed confidence for classical method
                            area=int(mask2.sum()),
                            tier='Tier3-GrabCut'
                        ))
                except Exception as e:
                    logger.debug(f"GrabCut failed for point {point}: {e}")
        else:
            # Use watershed for automatic segmentation
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Noise removal
                kernel = np.ones((3, 3), np.uint8)
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
                
                # Sure background area
                sure_bg = cv2.dilate(opening, kernel, iterations=3)
                
                # Finding sure foreground area
                dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
                _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
                
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg, sure_fg)
                
                # Marker labelling
                _, markers = cv2.connectedComponents(sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0
                
                # Apply watershed
                markers = cv2.watershed(image, markers)
                
                # Extract each region as a mask
                for label in np.unique(markers):
                    if label <= 1:  # Skip background and boundaries
                        continue
                    
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    mask[markers == label] = 1
                    
                    if mask.sum() > 500:  # Minimum area
                        bbox = self._mask_to_bbox(mask)
                        
                        masks_list.append(SegmentMask(
                            mask=mask,
                            bbox=bbox,
                            confidence=0.65,
                            area=int(mask.sum()),
                            tier='Tier3-Watershed'
                        ))
            except Exception as e:
                logger.debug(f"Watershed segmentation failed: {e}")
        
        inference_time = (time.time() - start_time) * 1000
        
        logger.debug(f"Tier 3: Generated {len(masks_list)} masks in {inference_time:.1f}ms")
        
        return SegmentationResult(
            masks=masks_list,
            inference_time_ms=inference_time,
            tier_used='Tier3-Classical',
            image_shape=image.shape
        )
    
    def _mask_to_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert binary mask to bounding box"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return (0, 0, 0, 0)
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return (int(cmin), int(rmin), int(cmax), int(rmax))
    
    def get_status(self) -> Dict[str, Any]:
        """Get segmenter status"""
        return {
            'tier1_sam': self.tier1_model is not None,
            'tier2_yolov8seg': self.tier2_model is not None,
            'tier3_classical': self.tier3_enabled,
            'device': self.device
        }


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("SEGMENTER - TESTING")
    print("="*80)
    
    # Initialize
    segmenter = Segmenter()
    
    # Show status
    status = segmenter.get_status()
    print(f"\nStatus: {status}\n")
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test segmentation
    print("Testing segmentation...")
    result = segmenter.segment(test_image, confidence_threshold=0.25)
    
    print(f"\nSegmentation Results:")
    print(f"  Tier Used: {result.tier_used}")
    print(f"  Inference Time: {result.inference_time_ms:.1f}ms")
    print(f"  Masks: {len(result.masks)}")
    
    for i, mask in enumerate(result.masks[:3]):  # Show first 3
        print(f"    Mask {i+1}: area={mask.area}, conf={mask.confidence:.2f}, bbox={mask.bbox}")
    
    print("\n" + "="*80)

