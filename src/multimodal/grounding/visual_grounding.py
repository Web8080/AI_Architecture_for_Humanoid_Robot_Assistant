"""
Visual Grounding System

PURPOSE:
    Implements visual grounding for referring expression comprehension.
    Enables the robot to understand natural language references to objects
    in the visual scene (e.g., "bring me that red cup on the left").

PIPELINE CONTEXT:
    
    Visual Grounding Flow:
    Text + Image → Object Detection → Referring Expression → Grounded Object
         ↓              ↓                    ↓                    ↓
    ┌─────────┐ ┌─────────────┐ ┌─────────────────┐ ┌─────────────┐
    │ "red    │ │ YOLOv11     │ │ Spatial         │ │ Bounding    │
    │ cup on  │ │ Object      │ │ Relationships   │ │ Box +       │
    │ left"   │ │ Detection   │ │ Color/Shape     │ │ Confidence  │
    └─────────┘ └─────────────┘ └─────────────────┘ └─────────────┘

WHY VISUAL GROUNDING MATTERS:
    Current System: Separate object detection and language understanding
    With Grounding: Unified understanding of "what" and "where"
    
    Benefits:
    - Natural language object references
    - Spatial relationship understanding
    - Color and attribute matching
    - Multi-object disambiguation
    - Human-like object selection

HOW IT WORKS:
    1. Object Detection: Find all objects in the scene
    2. Feature Extraction: Extract visual features for each object
    3. Language Processing: Parse referring expression
    4. Matching: Match language to visual features
    5. Ranking: Rank objects by grounding confidence

INTEGRATION WITH EXISTING SYSTEM:
    - Uses existing object detection (YOLOv11)
    - Uses existing NLP (entity extraction, parsing)
    - Adds grounding-specific matching algorithms
    - Enables natural object manipulation commands

RELATED FILES:
    - src/vision/object_detection/detector.py: Object detection
    - src/nlp/entities/extractor.py: Entity extraction
    - src/agents/multimodal_fusion.py: Multimodal fusion
    - configs/base/system_config.yaml: Grounding configuration

USAGE:
    # Initialize visual grounding
    grounding = VisualGrounding(config)
    
    # Ground referring expression
    result = await grounding.ground_expression(
        image=camera_image,
        expression="the red cup on the left"
    )
    
    # Get grounded object
    if result.grounded:
        bbox = result.bounding_box
        confidence = result.confidence
        print(f"Found {result.object_class} at {bbox} with {confidence:.2f} confidence")

Author: Victor Ibhafidon
Date: October 2025
Version: 1.0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import cv2
import re
import json
import time
from enum import Enum

# Import existing modules
from src.vision.object_detection.detector import ObjectDetector
from src.nlp.entities.extractor import EntityExtractor

logger = logging.getLogger(__name__)


class SpatialRelation(Enum):
    """Spatial relationship types"""
    LEFT = "left"
    RIGHT = "right"
    ABOVE = "above"
    BELOW = "below"
    NEAR = "near"
    FAR = "far"
    IN_FRONT = "in_front"
    BEHIND = "behind"
    ON = "on"
    UNDER = "under"


@dataclass
class GroundingResult:
    """Result from visual grounding"""
    grounded: bool
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    object_class: Optional[str] = None
    confidence: float = 0.0
    reasoning: List[str] = None
    alternatives: List[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class ReferringExpression:
    """Parsed referring expression"""
    object_type: str
    attributes: Dict[str, Any]  # color, size, shape, etc.
    spatial_relations: List[Tuple[SpatialRelation, str]]  # (relation, reference_object)
    quantifiers: List[str]  # "the", "a", "that", "this"
    confidence: float


class VisualGrounding:
    """
    Visual grounding system for referring expression comprehension
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize visual grounding system
        
        Args:
            config: Configuration for grounding models and parameters
        """
        self.config = config
        
        # Initialize components
        self.object_detector = ObjectDetector(config.get('object_detection', {}))
        self.entity_extractor = EntityExtractor(config.get('entity_extraction', {}))
        
        # Grounding parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.max_alternatives = config.get('max_alternatives', 5)
        
        # Color mapping for object attributes
        self.color_mapping = {
            'red': [(0, 0, 100), (0, 0, 255)],
            'blue': [(100, 0, 0), (255, 0, 0)],
            'green': [(0, 100, 0), (0, 255, 0)],
            'yellow': [(0, 255, 255), (0, 255, 255)],
            'orange': [(0, 165, 255), (0, 165, 255)],
            'purple': [(128, 0, 128), (128, 0, 128)],
            'pink': [(203, 192, 255), (203, 192, 255)],
            'brown': [(42, 42, 165), (42, 42, 165)],
            'black': [(0, 0, 0), (50, 50, 50)],
            'white': [(200, 200, 200), (255, 255, 255)],
            'gray': [(100, 100, 100), (150, 150, 150)],
            'grey': [(100, 100, 100), (150, 150, 150)]
        }
        
        # Size mapping
        self.size_mapping = {
            'small': (0.0, 0.3),
            'medium': (0.3, 0.7),
            'large': (0.7, 1.0),
            'tiny': (0.0, 0.2),
            'huge': (0.8, 1.0)
        }
        
        logger.info("Visual grounding system initialized")
    
    async def ground_expression(self, 
                              image: np.ndarray,
                              expression: str) -> GroundingResult:
        """
        Ground a referring expression in an image
        
        Args:
            image: Input image
            expression: Natural language referring expression
        
        Returns:
            Grounding result with bounding box and confidence
        """
        try:
            logger.info(f"Grounding expression: '{expression}'")
            
            # 1. Parse referring expression
            parsed_expression = self._parse_expression(expression)
            logger.info(f"Parsed expression: {parsed_expression}")
            
            # 2. Detect objects in image
            detection_result = await self.object_detector.detect(image)
            if not detection_result.objects:
                return GroundingResult(
                    grounded=False,
                    error_message="No objects detected in image"
                )
            
            logger.info(f"Detected {len(detection_result.objects)} objects")
            
            # 3. Extract visual features for each object
            object_features = self._extract_object_features(image, detection_result.objects)
            
            # 4. Match expression to objects
            matches = self._match_expression_to_objects(parsed_expression, object_features)
            
            # 5. Rank matches by confidence
            ranked_matches = self._rank_matches(matches)
            
            if not ranked_matches:
                return GroundingResult(
                    grounded=False,
                    error_message="No objects match the referring expression"
                )
            
            # 6. Return best match
            best_match = ranked_matches[0]
            
            return GroundingResult(
                grounded=True,
                bounding_box=best_match['bounding_box'],
                object_class=best_match['object_class'],
                confidence=best_match['confidence'],
                reasoning=best_match['reasoning'],
                alternatives=ranked_matches[1:self.max_alternatives]
            )
            
        except Exception as e:
            logger.error(f"Visual grounding failed: {e}")
            return GroundingResult(
                grounded=False,
                error_message=str(e)
            )
    
    def _parse_expression(self, expression: str) -> ReferringExpression:
        """
        Parse natural language referring expression
        
        Args:
            expression: Natural language expression
        
        Returns:
            Parsed referring expression
        """
        expression = expression.lower().strip()
        
        # Extract object type
        object_type = self._extract_object_type(expression)
        
        # Extract attributes (color, size, shape)
        attributes = self._extract_attributes(expression)
        
        # Extract spatial relations
        spatial_relations = self._extract_spatial_relations(expression)
        
        # Extract quantifiers
        quantifiers = self._extract_quantifiers(expression)
        
        # Calculate parsing confidence
        confidence = self._calculate_parsing_confidence(object_type, attributes, spatial_relations)
        
        return ReferringExpression(
            object_type=object_type,
            attributes=attributes,
            spatial_relations=spatial_relations,
            quantifiers=quantifiers,
            confidence=confidence
        )
    
    def _extract_object_type(self, expression: str) -> str:
        """Extract object type from expression"""
        # Common object types
        object_types = [
            'cup', 'bottle', 'phone', 'book', 'remote', 'keys', 'pen', 'pencil',
            'laptop', 'mouse', 'keyboard', 'chair', 'table', 'person', 'hand',
            'face', 'car', 'truck', 'bike', 'ball', 'toy', 'food', 'plate',
            'bowl', 'spoon', 'fork', 'knife', 'glass', 'mug', 'bag', 'box'
        ]
        
        for obj_type in object_types:
            if obj_type in expression:
                return obj_type
        
        # Default to generic object
        return 'object'
    
    def _extract_attributes(self, expression: str) -> Dict[str, Any]:
        """Extract object attributes from expression"""
        attributes = {}
        
        # Extract color
        for color in self.color_mapping.keys():
            if color in expression:
                attributes['color'] = color
                break
        
        # Extract size
        for size in self.size_mapping.keys():
            if size in expression:
                attributes['size'] = size
                break
        
        # Extract shape (basic)
        shapes = ['round', 'square', 'rectangular', 'circular', 'oval', 'triangular']
        for shape in shapes:
            if shape in expression:
                attributes['shape'] = shape
                break
        
        return attributes
    
    def _extract_spatial_relations(self, expression: str) -> List[Tuple[SpatialRelation, str]]:
        """Extract spatial relations from expression"""
        relations = []
        
        # Spatial relation patterns
        spatial_patterns = {
            r'\b(left|leftmost)\b': (SpatialRelation.LEFT, 'image'),
            r'\b(right|rightmost)\b': (SpatialRelation.RIGHT, 'image'),
            r'\b(above|over|on top of)\b': (SpatialRelation.ABOVE, 'reference'),
            r'\b(below|under|beneath)\b': (SpatialRelation.BELOW, 'reference'),
            r'\b(near|close to|next to)\b': (SpatialRelation.NEAR, 'reference'),
            r'\b(far|away from)\b': (SpatialRelation.FAR, 'reference'),
            r'\b(in front of)\b': (SpatialRelation.IN_FRONT, 'reference'),
            r'\b(behind)\b': (SpatialRelation.BEHIND, 'reference'),
            r'\b(on)\b': (SpatialRelation.ON, 'reference'),
            r'\b(under)\b': (SpatialRelation.UNDER, 'reference')
        }
        
        for pattern, (relation, ref_type) in spatial_patterns.items():
            matches = re.finditer(pattern, expression)
            for match in matches:
                relations.append((relation, ref_type))
        
        return relations
    
    def _extract_quantifiers(self, expression: str) -> List[str]:
        """Extract quantifiers from expression"""
        quantifiers = []
        
        quantifier_words = ['the', 'a', 'an', 'this', 'that', 'these', 'those']
        for word in quantifier_words:
            if word in expression:
                quantifiers.append(word)
        
        return quantifiers
    
    def _calculate_parsing_confidence(self, object_type, attributes, spatial_relations) -> float:
        """Calculate confidence in expression parsing"""
        confidence = 0.5  # Base confidence
        
        if object_type != 'object':
            confidence += 0.2
        
        if attributes:
            confidence += 0.1 * len(attributes)
        
        if spatial_relations:
            confidence += 0.1 * len(spatial_relations)
        
        return min(confidence, 1.0)
    
    def _extract_object_features(self, image: np.ndarray, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract visual features for each detected object
        
        Args:
            image: Input image
            objects: List of detected objects
        
        Returns:
            List of object features
        """
        features = []
        
        for obj in objects:
            bbox = obj['bbox']  # x1, y1, x2, y2
            x1, y1, x2, y2 = bbox
            
            # Extract object region
            object_region = image[y1:y2, x1:x2]
            
            # Extract color features
            color_features = self._extract_color_features(object_region)
            
            # Extract size features
            size_features = self._extract_size_features(bbox, image.shape)
            
            # Extract shape features
            shape_features = self._extract_shape_features(object_region)
            
            # Extract position features
            position_features = self._extract_position_features(bbox, image.shape)
            
            features.append({
                'object': obj,
                'bbox': bbox,
                'color': color_features,
                'size': size_features,
                'shape': shape_features,
                'position': position_features,
                'object_class': obj.get('class', 'unknown'),
                'confidence': obj.get('confidence', 0.0)
            })
        
        return features
    
    def _extract_color_features(self, object_region: np.ndarray) -> Dict[str, Any]:
        """Extract color features from object region"""
        if object_region.size == 0:
            return {'dominant_color': 'unknown', 'color_histogram': []}
        
        # Convert to HSV for better color analysis
        hsv_region = cv2.cvtColor(object_region, cv2.COLOR_BGR2HSV)
        
        # Calculate dominant color
        pixels = hsv_region.reshape(-1, 3)
        dominant_hue = np.median(pixels[:, 0])
        
        # Map hue to color name
        color_name = self._hue_to_color_name(dominant_hue)
        
        # Calculate color histogram
        hist = cv2.calcHist([hsv_region], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        
        return {
            'dominant_color': color_name,
            'dominant_hue': float(dominant_hue),
            'color_histogram': hist.flatten().tolist()
        }
    
    def _hue_to_color_name(self, hue: float) -> str:
        """Map HSV hue value to color name"""
        if 0 <= hue < 10 or 170 <= hue <= 180:
            return 'red'
        elif 10 <= hue < 25:
            return 'orange'
        elif 25 <= hue < 35:
            return 'yellow'
        elif 35 <= hue < 85:
            return 'green'
        elif 85 <= hue < 130:
            return 'blue'
        elif 130 <= hue < 170:
            return 'purple'
        else:
            return 'unknown'
    
    def _extract_size_features(self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """Extract size features from bounding box"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Normalize by image size
        image_area = image_shape[0] * image_shape[1]
        relative_area = area / image_area
        
        # Determine size category
        if relative_area < 0.01:
            size_category = 'tiny'
        elif relative_area < 0.05:
            size_category = 'small'
        elif relative_area < 0.2:
            size_category = 'medium'
        elif relative_area < 0.5:
            size_category = 'large'
        else:
            size_category = 'huge'
        
        return {
            'width': width,
            'height': height,
            'area': area,
            'relative_area': relative_area,
            'size_category': size_category
        }
    
    def _extract_shape_features(self, object_region: np.ndarray) -> Dict[str, Any]:
        """Extract shape features from object region"""
        if object_region.size == 0:
            return {'shape_category': 'unknown', 'aspect_ratio': 1.0}
        
        # Convert to grayscale
        gray = cv2.cvtColor(object_region, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'shape_category': 'unknown', 'aspect_ratio': 1.0}
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Determine shape category
        if 0.8 <= aspect_ratio <= 1.2:
            shape_category = 'square'
        elif aspect_ratio > 1.5:
            shape_category = 'rectangular'
        elif aspect_ratio < 0.67:
            shape_category = 'tall'
        else:
            shape_category = 'round'
        
        return {
            'shape_category': shape_category,
            'aspect_ratio': aspect_ratio,
            'contour_area': cv2.contourArea(largest_contour)
        }
    
    def _extract_position_features(self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """Extract position features from bounding box"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Normalize position
        norm_x = center_x / image_shape[1]
        norm_y = center_y / image_shape[0]
        
        # Determine horizontal position
        if norm_x < 0.33:
            horizontal_pos = 'left'
        elif norm_x > 0.67:
            horizontal_pos = 'right'
        else:
            horizontal_pos = 'center'
        
        # Determine vertical position
        if norm_y < 0.33:
            vertical_pos = 'top'
        elif norm_y > 0.67:
            vertical_pos = 'bottom'
        else:
            vertical_pos = 'middle'
        
        return {
            'center_x': center_x,
            'center_y': center_y,
            'norm_x': norm_x,
            'norm_y': norm_y,
            'horizontal_position': horizontal_pos,
            'vertical_position': vertical_pos
        }
    
    def _match_expression_to_objects(self, 
                                   expression: ReferringExpression,
                                   object_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Match referring expression to object features
        
        Args:
            expression: Parsed referring expression
            object_features: List of object features
        
        Returns:
            List of matches with confidence scores
        """
        matches = []
        
        for features in object_features:
            match_score = 0.0
            reasoning = []
            
            # Match object type
            if expression.object_type == features['object_class'] or expression.object_type == 'object':
                match_score += 0.3
                reasoning.append(f"Object type matches: {features['object_class']}")
            else:
                reasoning.append(f"Object type mismatch: expected {expression.object_type}, got {features['object_class']}")
            
            # Match attributes
            for attr_name, attr_value in expression.attributes.items():
                if attr_name == 'color':
                    if self._match_color(attr_value, features['color']):
                        match_score += 0.2
                        reasoning.append(f"Color matches: {attr_value}")
                    else:
                        reasoning.append(f"Color mismatch: expected {attr_value}, got {features['color']['dominant_color']}")
                
                elif attr_name == 'size':
                    if self._match_size(attr_value, features['size']):
                        match_score += 0.2
                        reasoning.append(f"Size matches: {attr_value}")
                    else:
                        reasoning.append(f"Size mismatch: expected {attr_value}, got {features['size']['size_category']}")
                
                elif attr_name == 'shape':
                    if self._match_shape(attr_value, features['shape']):
                        match_score += 0.1
                        reasoning.append(f"Shape matches: {attr_value}")
                    else:
                        reasoning.append(f"Shape mismatch: expected {attr_value}, got {features['shape']['shape_category']}")
            
            # Match spatial relations
            for relation, ref_type in expression.spatial_relations:
                if self._match_spatial_relation(relation, features, object_features):
                    match_score += 0.2
                    reasoning.append(f"Spatial relation matches: {relation.value}")
                else:
                    reasoning.append(f"Spatial relation mismatch: {relation.value}")
            
            # Add base confidence from object detection
            match_score += features['confidence'] * 0.1
            
            matches.append({
                'object': features['object'],
                'bounding_box': features['bbox'],
                'object_class': features['object_class'],
                'confidence': match_score,
                'reasoning': reasoning,
                'features': features
            })
        
        return matches
    
    def _match_color(self, expected_color: str, color_features: Dict[str, Any]) -> bool:
        """Match expected color to detected color"""
        detected_color = color_features.get('dominant_color', 'unknown')
        return expected_color.lower() == detected_color.lower()
    
    def _match_size(self, expected_size: str, size_features: Dict[str, Any]) -> bool:
        """Match expected size to detected size"""
        detected_size = size_features.get('size_category', 'unknown')
        return expected_size.lower() == detected_size.lower()
    
    def _match_shape(self, expected_shape: str, shape_features: Dict[str, Any]) -> bool:
        """Match expected shape to detected shape"""
        detected_shape = shape_features.get('shape_category', 'unknown')
        return expected_shape.lower() == detected_shape.lower()
    
    def _match_spatial_relation(self, 
                              relation: SpatialRelation,
                              target_features: Dict[str, Any],
                              all_features: List[Dict[str, Any]]) -> bool:
        """Match spatial relation between objects"""
        # Simplified spatial relation matching
        # In a full implementation, this would be more sophisticated
        
        target_pos = target_features['position']
        
        if relation == SpatialRelation.LEFT:
            return target_pos['horizontal_position'] == 'left'
        elif relation == SpatialRelation.RIGHT:
            return target_pos['horizontal_position'] == 'right'
        elif relation == SpatialRelation.ABOVE:
            return target_pos['vertical_position'] == 'top'
        elif relation == SpatialRelation.BELOW:
            return target_pos['vertical_position'] == 'bottom'
        
        # For other relations, would need more complex logic
        return True
    
    def _rank_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank matches by confidence score"""
        # Filter matches above threshold
        valid_matches = [m for m in matches if m['confidence'] >= self.confidence_threshold]
        
        # Sort by confidence (descending)
        ranked_matches = sorted(valid_matches, key=lambda x: x['confidence'], reverse=True)
        
        return ranked_matches
    
    async def batch_ground_expressions(self, 
                                     image: np.ndarray,
                                     expressions: List[str]) -> List[GroundingResult]:
        """
        Ground multiple referring expressions in the same image
        
        Args:
            image: Input image
            expressions: List of referring expressions
        
        Returns:
            List of grounding results
        """
        results = []
        
        for expression in expressions:
            result = await self.ground_expression(image, expression)
            results.append(result)
        
        return results
    
    def get_grounding_statistics(self) -> Dict[str, Any]:
        """Get grounding system statistics"""
        return {
            'confidence_threshold': self.confidence_threshold,
            'max_alternatives': self.max_alternatives,
            'supported_colors': list(self.color_mapping.keys()),
            'supported_sizes': list(self.size_mapping.keys()),
            'supported_spatial_relations': [rel.value for rel in SpatialRelation],
            'timestamp': time.time()
        }


# Example usage and testing
async def main():
    """Example usage of visual grounding"""
    config = {
        'object_detection': {
            'tier1_enabled': True,
            'tier2_enabled': True,
            'tier3_enabled': True
        },
        'entity_extraction': {
            'tier1_enabled': True,
            'tier2_enabled': True,
            'tier3_enabled': True
        },
        'confidence_threshold': 0.3,
        'max_alternatives': 3
    }
    
    # Initialize visual grounding
    grounding = VisualGrounding(config)
    
    # Test with synthetic image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test expressions
    expressions = [
        "the red cup",
        "the object on the left",
        "that small bottle",
        "the round object",
        "the thing near the center"
    ]
    
    for expression in expressions:
        result = await grounding.ground_expression(test_image, expression)
        print(f"Expression: '{expression}'")
        print(f"Grounded: {result.grounded}")
        if result.grounded:
            print(f"Bounding box: {result.bounding_box}")
            print(f"Object class: {result.object_class}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Reasoning: {result.reasoning}")
        print("-" * 50)
    
    # Test batch grounding
    batch_results = await grounding.batch_ground_expressions(test_image, expressions)
    print(f"Batch grounding completed: {len(batch_results)} results")
    
    # Get statistics
    stats = grounding.get_grounding_statistics()
    print(f"Grounding statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
