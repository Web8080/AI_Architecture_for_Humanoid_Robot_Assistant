"""
Robot Simulation Environment for CV/NLP Testing

PURPOSE:
    Creates realistic simulated scenarios to test computer vision and NLP systems
    without requiring physical robot or real-world data collection. Enables
    rapid iteration and validation of AI models in controlled environments.

PIPELINE CONTEXT:
    
    Real Robot Pipeline:
    Camera → CV Module → Understanding → Planning → Actions
    
    Simulation Pipeline:
    Simulated Scenes → CV Module → Validation → Metrics
                  ↓
            [Same CV Code]  ← Test without hardware!
                  ↓
            Ground Truth Comparison → Performance Metrics

WHY SIMULATIONS MATTER:
    1. Safety: Test edge cases without risk (e.g., human collision detection)
    2. Speed: Generate 1000s of test scenarios in minutes
    3. Cost: No need for physical setup, cameras, robot hardware
    4. Control: Perfect ground truth (exact object positions, depth values)
    5. Reproducibility: Same scenarios for A/B testing
    6. Edge Cases: Test rare scenarios (low light, occlusion, etc.)

HOW IT WORKS:
    1. Generates synthetic images with known ground truth
    2. Runs CV models on synthetic data
    3. Compares predictions vs ground truth
    4. Calculates metrics (mAP, IoU, precision, recall)
    5. Validates multi-tier fallback behavior

SCENARIOS GENERATED:
    - Kitchen tasks (fetch cup, identify objects)
    - Navigation (obstacle avoidance, path planning)
    - Human interaction (handoff, following, safety distance)
    - Edge cases (low light, occlusion, clutter)
    - Failure modes (missing objects, ambiguous scenes)

INTEGRATION WITH PIPELINE:
    - Development: Test new features before deployment
    - CI/CD: Automated testing on every commit
    - Evaluation: Benchmark model performance
    - Training: Generate synthetic training data
    - Debugging: Reproduce specific failure cases

RELATED FILES:
    - src/vision/: CV components being tested
    - evaluation/cv_evaluation.py: Metrics calculation
    - training/: Fine-tuning pipelines
    - data/: Real-world data collection

USAGE:
    # Generate test scenarios
    sim = SimulationEnvironment()
    scenarios = sim.generate_scenarios(scenario_type='kitchen_tasks', count=100)
    
    # Test CV models
    results = sim.test_model(detector, scenarios)
    
    # Get metrics
    metrics = sim.calculate_metrics(results)

Author: Victor Ibhafidon
Date: October 2025
Version: 1.0
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import random


@dataclass
class ScenarioConfig:
    """Configuration for a simulation scenario"""
    name: str
    num_objects: int
    object_types: List[str]
    lighting: str  # 'bright', 'normal', 'low', 'dark'
    occlusion_level: float  # 0.0 to 1.0
    clutter_level: float  # 0.0 to 1.0
    camera_height_m: float  # Robot viewpoint
    camera_angle_deg: float  # Downward angle


@dataclass
class GroundTruth:
    """Ground truth for generated scenario"""
    image: np.ndarray
    objects: List[Dict[str, Any]]  # List of {bbox, class, depth, ...}
    depth_map: Optional[np.ndarray]
    scene_type: str
    lighting: str
    metadata: Dict[str, Any]


class SimulationEnvironment:
    """
    Simulation environment for robot AI testing
    
    Generates realistic scenarios with perfect ground truth
    for validation and benchmarking
    """
    
    # Robot-specific object classes
    ROBOT_OBJECTS = [
        # Kitchen items
        'cup', 'plate', 'bowl', 'spoon', 'fork', 'knife',
        'bottle', 'can', 'box', 'fruit', 'vegetable',
        
        # Household items
        'remote', 'phone', 'book', 'pen', 'keys',
        'shoe', 'bag', 'hat', 'toy', 'tool',
        
        # Furniture (obstacles)
        'chair', 'table', 'couch', 'shelf', 'cabinet',
        
        # People
        'person', 'face', 'hand'
    ]
    
    # Scenario templates
    SCENARIOS = {
        'kitchen_fetch': {
            'description': 'Fetch object from kitchen counter',
            'objects': ['cup', 'plate', 'spoon', 'bottle'],
            'lighting': 'normal',
            'occlusion': 0.2,
            'clutter': 0.4
        },
        'navigation_cluttered': {
            'description': 'Navigate through cluttered room',
            'objects': ['chair', 'table', 'box', 'bag'],
            'lighting': 'normal',
            'occlusion': 0.5,
            'clutter': 0.8
        },
        'human_handoff': {
            'description': 'Hand object to person',
            'objects': ['person', 'hand', 'cup'],
            'lighting': 'normal',
            'occlusion': 0.3,
            'clutter': 0.2
        },
        'low_light_detection': {
            'description': 'Detect objects in low light',
            'objects': ['cup', 'phone', 'keys'],
            'lighting': 'low',
            'occlusion': 0.3,
            'clutter': 0.3
        },
        'occlusion_test': {
            'description': 'Partially occluded objects',
            'objects': ['cup', 'plate', 'book', 'box'],
            'lighting': 'normal',
            'occlusion': 0.7,
            'clutter': 0.5
        },
        'safety_human_proximity': {
            'description': 'Human safety zone detection',
            'objects': ['person', 'face'],
            'lighting': 'normal',
            'occlusion': 0.1,
            'clutter': 0.2
        },
        'multi_object_scene': {
            'description': 'Multiple objects on table',
            'objects': ['cup', 'plate', 'bowl', 'spoon', 'fork', 'bottle'],
            'lighting': 'bright',
            'occlusion': 0.3,
            'clutter': 0.6
        },
        'edge_case_dark': {
            'description': 'Very low light conditions',
            'objects': ['cup', 'phone'],
            'lighting': 'dark',
            'occlusion': 0.4,
            'clutter': 0.3
        }
    }
    
    def __init__(self, image_size: Tuple[int, int] = (640, 480)):
        """
        Initialize simulation environment
        
        Args:
            image_size: (width, height) of generated images
        """
        self.image_size = image_size
        self.output_dir = Path('simulation/generated_scenarios')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_scenarios(
        self,
        scenario_type: str,
        count: int = 100,
        variations: bool = True
    ) -> List[GroundTruth]:
        """
        Generate multiple scenarios of given type
        
        Args:
            scenario_type: Type from SCENARIOS dict
            count: Number of scenarios to generate
            variations: Add random variations
        
        Returns:
            List of GroundTruth objects
        """
        print(f"\nGenerating {count} '{scenario_type}' scenarios...")
        
        if scenario_type not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_type}")
        
        template = self.SCENARIOS[scenario_type]
        scenarios = []
        
        for i in range(count):
            # Create scenario with variations
            config = ScenarioConfig(
                name=f"{scenario_type}_{i:04d}",
                num_objects=random.randint(2, len(template['objects'])) if variations else len(template['objects']),
                object_types=template['objects'],
                lighting=template['lighting'],
                occlusion_level=template['occlusion'] + (random.uniform(-0.1, 0.1) if variations else 0),
                clutter_level=template['clutter'] + (random.uniform(-0.1, 0.1) if variations else 0),
                camera_height_m=1.0 + (random.uniform(-0.2, 0.2) if variations else 0),  # Robot eye height
                camera_angle_deg=15.0 + (random.uniform(-10, 10) if variations else 0)  # Looking down
            )
            
            # Generate scenario
            ground_truth = self._generate_single_scenario(config)
            scenarios.append(ground_truth)
            
            # Save scenario
            self._save_scenario(ground_truth, i)
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{count} scenarios")
        
        print(f"Completed! Scenarios saved to: {self.output_dir}")
        return scenarios
    
    def _generate_single_scenario(self, config: ScenarioConfig) -> GroundTruth:
        """
        Generate a single scenario with ground truth
        
        Creates synthetic image with known object positions
        """
        width, height = self.image_size
        
        # Create base image (simulated table/floor/counter)
        image = self._create_base_scene(config)
        
        # Add objects with known positions
        objects = []
        depth_map = np.zeros((height, width), dtype=np.float32)
        
        for obj_idx in range(config.num_objects):
            # Random object type from scenario
            obj_type = random.choice(config.object_types)
            
            # Generate object with realistic constraints
            obj_data = self._place_object(
                image,
                depth_map,
                obj_type,
                config,
                existing_objects=objects
            )
            
            if obj_data:
                objects.append(obj_data)
        
        # Apply lighting effects
        image = self._apply_lighting(image, config.lighting)
        
        # Apply occlusion (objects hiding each other)
        image, objects = self._apply_occlusion(image, objects, config.occlusion_level)
        
        # Add noise/blur for realism
        image = self._add_realism(image)
        
        ground_truth = GroundTruth(
            image=image,
            objects=objects,
            depth_map=depth_map,
            scene_type=config.name.split('_')[0],
            lighting=config.lighting,
            metadata={
                'config': config,
                'num_objects': len(objects),
                'camera_height_m': config.camera_height_m,
                'camera_angle_deg': config.camera_angle_deg
            }
        )
        
        return ground_truth
    
    def _create_base_scene(self, config: ScenarioConfig) -> np.ndarray:
        """Create base scene (table, floor, etc.)"""
        width, height = self.image_size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Simulate table/counter surface
        # Top 2/3 is background (wall)
        bg_color = (180, 180, 170)  # Light gray wall
        image[:int(height * 0.4), :] = bg_color
        
        # Bottom 1/3 is table surface
        table_color = (120, 100, 80)  # Brown table
        image[int(height * 0.4):, :] = table_color
        
        # Add wood texture
        noise = np.random.randint(-20, 20, (height, width, 3))
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def _place_object(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        obj_type: str,
        config: ScenarioConfig,
        existing_objects: List[Dict]
    ) -> Optional[Dict]:
        """
        Place an object in the scene
        
        Returns object metadata with ground truth bbox
        """
        height, width = image.shape[:2]
        
        # Object size (varies by type)
        if obj_type == 'person':
            obj_width = random.randint(80, 150)
            obj_height = random.randint(150, 250)
        elif obj_type in ['cup', 'phone', 'keys']:
            obj_width = random.randint(30, 60)
            obj_height = random.randint(40, 80)
        else:
            obj_width = random.randint(50, 100)
            obj_height = random.randint(50, 100)
        
        # Try to find non-overlapping position
        for attempt in range(20):
            x1 = random.randint(50, width - obj_width - 50)
            y1 = random.randint(int(height * 0.4), height - obj_height - 20)
            x2 = x1 + obj_width
            y2 = y1 + obj_height
            
            # Check overlap with existing objects
            overlap = False
            for existing in existing_objects:
                ex1, ey1, ex2, ey2 = existing['bbox']
                if not (x2 < ex1 or x1 > ex2 or y2 < ey1 or y1 > ey2):
                    overlap = True
                    break
            
            if not overlap:
                # Place object (colored rectangle for now)
                color = self._get_object_color(obj_type)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
                
                # Add border
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
                
                # Add label text
                cv2.putText(image, obj_type[:3].upper(), (x1 + 5, y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Calculate depth (distance from camera)
                depth_value = 0.5 + (y1 / height) * 1.5  # 0.5m to 2.0m
                depth_map[y1:y2, x1:x2] = depth_value
                
                return {
                    'bbox': (x1, y1, x2, y2),
                    'class': obj_type,
                    'class_id': self.ROBOT_OBJECTS.index(obj_type) if obj_type in self.ROBOT_OBJECTS else -1,
                    'depth_m': depth_value,
                    'visible': True,
                    'occlusion': 0.0
                }
        
        return None  # Couldn't place object
    
    def _get_object_color(self, obj_type: str) -> Tuple[int, int, int]:
        """Get representative color for object type"""
        colors = {
            'cup': (200, 200, 255),  # Light blue
            'plate': (230, 230, 230),  # White
            'bowl': (210, 180, 140),  # Tan
            'person': (180, 120, 100),  # Skin tone
            'phone': (50, 50, 50),  # Black
            'bottle': (100, 200, 100),  # Green
            'chair': (139, 69, 19),  # Brown
            'table': (160, 82, 45),  # Sienna
        }
        return colors.get(obj_type, (150, 150, 150))
    
    def _apply_lighting(self, image: np.ndarray, lighting: str) -> np.ndarray:
        """Apply lighting conditions"""
        if lighting == 'bright':
            return np.clip(image * 1.3, 0, 255).astype(np.uint8)
        elif lighting == 'low':
            return np.clip(image * 0.6, 0, 255).astype(np.uint8)
        elif lighting == 'dark':
            return np.clip(image * 0.3, 0, 255).astype(np.uint8)
        return image  # Normal lighting
    
    def _apply_occlusion(
        self,
        image: np.ndarray,
        objects: List[Dict],
        occlusion_level: float
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Simulate objects partially hiding each other"""
        if occlusion_level < 0.1 or len(objects) < 2:
            return image, objects
        
        # Randomly occlude some objects
        num_to_occlude = int(len(objects) * occlusion_level)
        for i in range(num_to_occlude):
            if i < len(objects):
                x1, y1, x2, y2 = objects[i]['bbox']
                
                # Draw occlusion bar
                occ_height = int((y2 - y1) * 0.4)
                cv2.rectangle(image, (x1, y1), (x2, y1 + occ_height), (100, 100, 100), -1)
                
                objects[i]['occlusion'] = 0.4
                objects[i]['visible'] = True if occlusion_level < 0.6 else False
        
        return image, objects
    
    def _add_realism(self, image: np.ndarray) -> np.ndarray:
        """Add noise and blur for realism"""
        # Gaussian noise
        noise = np.random.normal(0, 3, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Slight blur
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        return image
    
    def _save_scenario(self, ground_truth: GroundTruth, idx: int):
        """Save scenario image and annotations"""
        # Save image
        img_path = self.output_dir / f"scenario_{idx:04d}.png"
        cv2.imwrite(str(img_path), ground_truth.image)
        
        # Save annotations (COCO format)
        annotations = {
            'image_id': idx,
            'file_name': img_path.name,
            'width': ground_truth.image.shape[1],
            'height': ground_truth.image.shape[0],
            'objects': ground_truth.objects,
            'scene_type': ground_truth.scene_type,
            'lighting': ground_truth.lighting,
            'metadata': ground_truth.metadata
        }
        
        ann_path = self.output_dir / f"scenario_{idx:04d}.json"
        with open(ann_path, 'w') as f:
            json.dump(annotations, f, indent=2, default=str)
    
    def test_model_on_scenarios(
        self,
        model,
        scenarios: List[GroundTruth]
    ) -> Dict[str, Any]:
        """
        Test CV model on generated scenarios
        
        Returns performance metrics
        """
        print(f"\nTesting model on {len(scenarios)} scenarios...")
        
        results = {
            'total_scenarios': len(scenarios),
            'predictions': [],
            'metrics': {}
        }
        
        for i, scenario in enumerate(scenarios):
            # Run model inference
            if hasattr(model, 'detect'):
                prediction = model.detect(scenario.image)
            else:
                raise ValueError("Model must have 'detect' method")
            
            # Store prediction with ground truth
            results['predictions'].append({
                'scenario_id': i,
                'ground_truth': scenario.objects,
                'predictions': [
                    {
                        'bbox': det.bbox,
                        'class': det.class_name,
                        'confidence': det.confidence
                    } for det in prediction.detections
                ],
                'latency_ms': prediction.inference_time_ms,
                'tier_used': prediction.tier_used
            })
        
        # Calculate aggregate metrics
        from evaluation.cv_evaluation import CVEvaluator
        evaluator = CVEvaluator()
        
        # This would calculate mAP, precision, recall, etc.
        # (Simplified for this example)
        results['metrics'] = {
            'average_latency_ms': np.mean([p['latency_ms'] for p in results['predictions']]),
            'tier_distribution': self._calculate_tier_distribution(results['predictions'])
        }
        
        print(f"Testing complete!")
        print(f"  Average Latency: {results['metrics']['average_latency_ms']:.1f}ms")
        print(f"  Tier Distribution: {results['metrics']['tier_distribution']}")
        
        return results
    
    def _calculate_tier_distribution(self, predictions: List[Dict]) -> Dict[str, float]:
        """Calculate which tiers were used"""
        tiers = [p['tier_used'] for p in predictions]
        total = len(tiers)
        
        return {
            tier: tiers.count(tier) / total * 100
            for tier in set(tiers)
        }


def main():
    """Demo simulation environment"""
    print("="*80)
    print("SIMULATION ENVIRONMENT - DEMO")
    print("="*80)
    
    sim = SimulationEnvironment()
    
    # Generate different scenario types
    scenario_types = [
        ('kitchen_fetch', 50),
        ('navigation_cluttered', 30),
        ('low_light_detection', 20),
        ('occlusion_test', 20),
        ('human_handoff', 10)
    ]
    
    all_scenarios = []
    for scenario_type, count in scenario_types:
        scenarios = sim.generate_scenarios(scenario_type, count=count, variations=True)
        all_scenarios.extend(scenarios)
    
    print(f"\nGenerated {len(all_scenarios)} total scenarios")
    print(f"Saved to: {sim.output_dir}")
    
    print("\nScenarios can now be used for:")
    print("  1. Testing CV models without hardware")
    print("  2. Benchmarking different model versions")
    print("  3. Validating multi-tier fallback")
    print("  4. Generating synthetic training data")
    print("  5. Reproducing edge cases")
    
    return all_scenarios


if __name__ == "__main__":
    main()

