"""
Robust Data Collection Strategy for Robot AI Training

PURPOSE:
    Defines comprehensive data collection scenarios for fine-tuning CV and NLP models
    on robot-specific tasks. Ensures collected data covers all operational scenarios,
    edge cases, and failure modes the robot will encounter.

PIPELINE CONTEXT:
    
    Data Collection Pipeline:
    Scenarios → Capture → Annotation → Quality Control → Training
                  ↓
        ┌─────────┴────────────┐
        │  Image Collection    │  → Camera frames
        │  Video Recording     │  → Temporal data
        │  Sensor Data         │  → Depth, IMU
        │  Annotations         │  → Ground truth
        └──────────────────────┘
                  ↓
        Training Data → Model Fine-Tuning → Improved Performance

WHY ROBUST DATA COLLECTION MATTERS:
    1. Model Quality: Garbage in, garbage out
    2. Domain Adaptation: Pre-trained models need robot-specific data
    3. Edge Cases: Cover failure modes and rare scenarios
    4. Balanced Dataset: Avoid bias toward common scenarios
    5. Ground Truth: Enable supervised learning and validation

COLLECTION STRATEGY:
    - Systematic scenario coverage (not random)
    - Multiple viewpoints and lighting conditions
    - Realistic robot operating conditions
    - Edge cases and failure modes
    - Quality control and validation

DATA TYPES COLLECTED:
    1. Images: RGB frames from robot cameras
    2. Depth: Depth maps from stereo/RGBD sensors
    3. Video: Temporal sequences for tracking
    4. Annotations: Bounding boxes, masks, keypoints
    5. Metadata: Scene description, lighting, conditions

SCENARIOS (Comprehensive):
    Priority 1 (Core Operations):
        - Kitchen tasks (fetch, place, identify)
        - Navigation (path planning, obstacle avoidance)
        - Human interaction (handoff, following, proximity)
    
    Priority 2 (Extended Operations):
        - Multi-object scenes (clutter, sorting)
        - Different lighting (bright, normal, low, dark)
        - Various viewpoints (table, floor, shelf)
    
    Priority 3 (Edge Cases):
        - Occlusion (partial visibility)
        - Similar objects (disambiguation)
        - Transparent/reflective objects
        - Moving objects
        - Failure modes

INTEGRATION WITH PIPELINE:
    - Training: Collected data → Fine-tuning → Improved models
    - Evaluation: Held-out test set → Performance metrics
    - Continuous Learning: New scenarios → Incremental training
    - Failure Analysis: Failure cases → Targeted data collection

QUALITY CONTROL:
    - Double annotation (2 annotators per image)
    - Review process (senior annotator validates)
    - Inter-annotator agreement (Cohen's Kappa > 0.8)
    - Automated checks (format, completeness)

RELATED FILES:
    - simulation/sim_environment.py: Synthetic data generation
    - training/finetune_yolov11.py: Model fine-tuning
    - evaluation/cv_evaluation.py: Performance metrics
    - configs/: Annotation guidelines and schemas

USAGE:
    # Define collection plan
    strategy = DataCollectionStrategy()
    
    # Get scenarios for Priority 1
    scenarios = strategy.get_priority_1_scenarios()
    
    # Execute collection
    strategy.execute_collection(scenarios)
    
    # Quality control
    strategy.validate_collected_data()

Author: Victor Ibhafidon
Date: October 2025
Version: 1.0
"""

import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import yaml


@dataclass
class CollectionScenario:
    """Definition of a data collection scenario"""
    id: str
    name: str
    description: str
    priority: int  # 1=critical, 2=important, 3=nice-to-have
    category: str  # kitchen, navigation, interaction, edge_case
    
    # Target collection metrics
    target_images: int
    target_videos: int
    target_duration_minutes: int
    
    # Scene requirements
    objects_required: List[str]
    environment: str  # kitchen, living_room, office, etc.
    lighting_conditions: List[str]  # bright, normal, low, dark
    viewpoints: List[str]  # table_height, floor, shelf, etc.
    
    # Annotation requirements
    annotation_types: List[str]  # bbox, segmentation, depth, keypoints
    ground_truth_required: bool
    
    # Quality requirements
    min_object_visibility: float  # 0.0 to 1.0
    max_occlusion: float  # 0.0 to 1.0
    min_image_quality: str  # low, medium, high
    
    # Instructions for data collector
    setup_instructions: str
    capture_instructions: str
    success_criteria: str
    
    # Metadata
    estimated_time_hours: float
    difficulty: str  # easy, medium, hard
    equipment_needed: List[str]
    

class DataCollectionStrategy:
    """
    Comprehensive data collection strategy for robot AI
    
    Defines all scenarios, priorities, and quality requirements
    """
    
    def __init__(self, output_dir: str = "data/collected"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scenarios = self._define_all_scenarios()
        self.collection_log = []
    
    def _define_all_scenarios(self) -> List[CollectionScenario]:
        """
        Define all data collection scenarios
        
        Comprehensive list covering all robot operations
        """
        scenarios = []
        
        # ====================================================================
        # PRIORITY 1: CORE OPERATIONS (Critical for basic functionality)
        # ====================================================================
        
        # Scenario 1: Kitchen - Fetch Cup from Counter
        scenarios.append(CollectionScenario(
            id="P1_001",
            name="Fetch Cup from Counter",
            description="Robot identifies and fetches a cup from kitchen counter",
            priority=1,
            category="kitchen",
            target_images=500,
            target_videos=50,
            target_duration_minutes=120,
            objects_required=['cup', 'counter', 'background_objects'],
            environment="kitchen",
            lighting_conditions=['bright', 'normal', 'low'],
            viewpoints=['robot_eye_level', 'slightly_elevated'],
            annotation_types=['bbox', 'segmentation', 'depth'],
            ground_truth_required=True,
            min_object_visibility=0.7,
            max_occlusion=0.3,
            min_image_quality='high',
            setup_instructions="""
                1. Place 1-3 cups on kitchen counter at various positions
                2. Add 2-4 other objects (plates, bottles, etc.) for realism
                3. Vary cup types: glass, ceramic, plastic, with/without handle
                4. Capture from robot's eye level (0.8-1.2m height)
                5. Include close-up and medium distance shots
            """,
            capture_instructions="""
                1. Capture 10 images per cup position
                2. Vary lighting (open/close blinds, lights on/off)
                3. Small camera movements for viewpoint diversity
                4. Include some frames with robot arm partially visible
                5. Record 5-10 second videos of approaching cup
            """,
            success_criteria="""
                - Cup clearly visible in all frames
                - Bounding box annotation accurate within 5 pixels
                - Multiple lighting conditions represented
                - At least 3 different cup types captured
            """,
            estimated_time_hours=3.0,
            difficulty="easy",
            equipment_needed=['RGB camera', 'depth sensor', 'tripod', 'measuring tape']
        ))
        
        # Scenario 2: Kitchen - Multiple Objects on Table
        scenarios.append(CollectionScenario(
            id="P1_002",
            name="Multi-Object Table Scene",
            description="Multiple objects on table - identification and localization",
            priority=1,
            category="kitchen",
            target_images=800,
            target_videos=80,
            target_duration_minutes=180,
            objects_required=['cup', 'plate', 'bowl', 'spoon', 'fork', 'bottle', 'fruit'],
            environment="kitchen",
            lighting_conditions=['bright', 'normal'],
            viewpoints=['table_height', 'robot_eye_level'],
            annotation_types=['bbox', 'segmentation', 'keypoints'],
            ground_truth_required=True,
            min_object_visibility=0.6,
            max_occlusion=0.4,
            min_image_quality='high',
            setup_instructions="""
                1. Arrange 3-7 objects on table in natural configurations
                2. Vary object spacing (tight clusters to spread out)
                3. Include different object combinations
                4. Use real household items, not toys
                5. Measure and record object positions for ground truth
            """,
            capture_instructions="""
                1. Capture from multiple angles (0°, 30°, 60° viewing angle)
                2. Vary distance: close (0.5m), medium (1.0m), far (2.0m)
                3. Capture with different background contexts
                4. Include some frames with human hands interacting
                5. Record videos of objects being moved/rearranged
            """,
            success_criteria="""
                - All objects annotated with tight bounding boxes
                - Occlusion properly labeled
                - At least 5 different object configurations
                - Clear depth information for all objects
            """,
            estimated_time_hours=4.5,
            difficulty="medium",
            equipment_needed=['RGB-D camera', 'table', 'household objects', 'annotation software']
        ))
        
        # Scenario 3: Navigation - Obstacle Detection
        scenarios.append(CollectionScenario(
            id="P1_003",
            name="Navigation Obstacle Detection",
            description="Detect and avoid obstacles during navigation",
            priority=1,
            category="navigation",
            target_images=600,
            target_videos=100,
            target_duration_minutes=150,
            objects_required=['chair', 'table', 'person', 'box', 'bag', 'toy'],
            environment="living_room",
            lighting_conditions=['normal', 'low'],
            viewpoints=['robot_base_level'],
            annotation_types=['bbox', 'segmentation', 'depth'],
            ground_truth_required=True,
            min_object_visibility=0.5,
            max_occlusion=0.5,
            min_image_quality='medium',
            setup_instructions="""
                1. Create realistic obstacle scenarios in open space
                2. Place obstacles at varying distances (0.5m to 5m)
                3. Include moving obstacles (people walking)
                4. Vary obstacle sizes (small toys to large furniture)
                5. Create both simple (1-2 obstacles) and complex (5+ obstacles) scenes
            """,
            capture_instructions="""
                1. Capture while robot is stationary and moving
                2. Include obstacles at different angles
                3. Capture partial obstacles (e.g., chair leg visible)
                4. Record videos of navigation through obstacle course
                5. Include failure cases (collision scenarios)
            """,
            success_criteria="""
                - All obstacles annotated with safety margins
                - Moving vs static obstacles labeled
                - Distance to obstacles recorded
                - Navigation paths marked
            """,
            estimated_time_hours=3.5,
            difficulty="medium",
            equipment_needed=['Mobile robot platform', 'wide-angle camera', 'depth sensor', 'safety spotter']
        ))
        
        # Scenario 4: Human Interaction - Object Handoff
        scenarios.append(CollectionScenario(
            id="P1_004",
            name="Human-Robot Object Handoff",
            description="Robot hands object to human or receives from human",
            priority=1,
            category="interaction",
            target_images=400,
            target_videos=80,
            target_duration_minutes=120,
            objects_required=['person', 'hand', 'cup', 'bottle', 'phone', 'book'],
            environment="living_room",
            lighting_conditions=['bright', 'normal'],
            viewpoints=['robot_eye_level'],
            annotation_types=['bbox', 'keypoints', 'depth'],
            ground_truth_required=True,
            min_object_visibility=0.7,
            max_occlusion=0.3,
            min_image_quality='high',
            setup_instructions="""
                1. Work with 3-5 different people (diverse ages, genders)
                2. Position person at various distances (0.5m to 2.0m)
                3. Capture handoffs from different angles
                4. Include both robot-to-human and human-to-robot transfers
                5. Use various objects (heavy, light, fragile, large, small)
            """,
            capture_instructions="""
                1. Capture full handoff sequence (approach, transfer, release)
                2. Focus on hand positions and orientations
                3. Include face detection for social interaction
                4. Record videos of complete handoff interactions
                5. Capture successful and failed handoffs
            """,
            success_criteria="""
                - Hand keypoints annotated (wrist, palm, fingers)
                - Object transfer point clearly visible
                - Person's intent clearly readable (reaching, waiting)
                - Safe interaction distance maintained
            """,
            estimated_time_hours=3.0,
            difficulty="medium",
            equipment_needed=['RGB camera', 'depth sensor', 'multiple participants', 'various objects']
        ))
        
        # ====================================================================
        # PRIORITY 2: EXTENDED OPERATIONS (Important for robustness)
        # ====================================================================
        
        # Scenario 5: Low Light Conditions
        scenarios.append(CollectionScenario(
            id="P2_001",
            name="Low Light Object Detection",
            description="Detect objects in low light/evening conditions",
            priority=2,
            category="edge_case",
            target_images=300,
            target_videos=30,
            target_duration_minutes=90,
            objects_required=['cup', 'phone', 'keys', 'remote'],
            environment="living_room",
            lighting_conditions=['low', 'dark'],
            viewpoints=['robot_eye_level'],
            annotation_types=['bbox'],
            ground_truth_required=True,
            min_object_visibility=0.4,
            max_occlusion=0.4,
            min_image_quality='medium',
            setup_instructions="""
                1. Close blinds and dim lights progressively
                2. Test with desk lamps, smartphone lights
                3. Capture at dusk/evening with natural light
                4. Include scenarios with single light source
                5. Vary light direction (front, side, back lighting)
            """,
            capture_instructions="""
                1. Measure light levels with lux meter
                2. Capture same scene at different light levels
                3. Test ISO/exposure settings
                4. Include IR/night vision if available
                5. Document light source positions
            """,
            success_criteria="""
                - Objects still detectable in low light
                - Light level documented for each image
                - Annotations adjusted for reduced visibility
                - At least 3 different lighting levels tested
            """,
            estimated_time_hours=2.5,
            difficulty="medium",
            equipment_needed=['Light meter', 'adjustable lights', 'camera with good low-light performance']
        ))
        
        # Scenario 6: Occlusion and Partial Visibility
        scenarios.append(CollectionScenario(
            id="P2_002",
            name="Occluded Objects",
            description="Objects partially hidden by other objects",
            priority=2,
            category="edge_case",
            target_images=400,
            target_videos=40,
            target_duration_minutes=120,
            objects_required=['cup', 'plate', 'bowl', 'box', 'book'],
            environment="kitchen",
            lighting_conditions=['normal'],
            viewpoints=['table_height', 'robot_eye_level'],
            annotation_types=['bbox', 'segmentation', 'occlusion_mask'],
            ground_truth_required=True,
            min_object_visibility=0.3,  # Allow heavy occlusion
            max_occlusion=0.7,
            min_image_quality='high',
            setup_instructions="""
                1. Deliberately create occlusion scenarios
                2. Place objects behind, in front of each other
                3. Vary occlusion level (20%, 50%, 70%)
                4. Include transparent/translucent occluders (glass)
                5. Create realistic cluttered scenes
            """,
            capture_instructions="""
                1. Annotate visible and occluded portions separately
                2. Mark occlusion boundaries precisely
                3. Capture from angles where occlusion changes
                4. Include videos of removing occluding objects
                5. Document which objects occlude which
            """,
            success_criteria="""
                - Occlusion percentage labeled
                - Visible parts accurately annotated
                - Full object extent estimated (even if not visible)
                - At least 10 different occlusion patterns
            """,
            estimated_time_hours=3.0,
            difficulty="hard",
            equipment_needed=['Multiple objects', 'annotation tool with occlusion support', 'patient annotators']
        ))
        
        # Scenario 7: Transparent and Reflective Objects
        scenarios.append(CollectionScenario(
            id="P2_003",
            name="Transparent/Reflective Objects",
            description="Glass cups, mirrors, metallic objects",
            priority=2,
            category="edge_case",
            target_images=200,
            target_videos=20,
            target_duration_minutes=60,
            objects_required=['glass_cup', 'glass_plate', 'mirror', 'metallic_object'],
            environment="kitchen",
            lighting_conditions=['bright', 'normal'],
            viewpoints=['multiple_angles'],
            annotation_types=['bbox', 'segmentation'],
            ground_truth_required=True,
            min_object_visibility=0.5,
            max_occlusion=0.3,
            min_image_quality='high',
            setup_instructions="""
                1. Use glass and metallic objects
                2. Vary background (plain, patterned, cluttered)
                3. Test different lighting angles
                4. Include reflections and refractions
                5. Use colored and clear glass
            """,
            capture_instructions="""
                1. Capture reflections separately
                2. Use polarizing filter if available
                3. Mark reflection areas in annotations
                4. Include edge cases (nearly invisible glass)
                5. Document material properties
            """,
            success_criteria="""
                - Object boundaries marked despite transparency
                - Reflections noted in metadata
                - Multiple angles to disambiguate
                - Material type labeled
            """,
            estimated_time_hours=2.0,
            difficulty="hard",
            equipment_needed=['Glass/metallic objects', 'good lighting setup', 'polarizing filter (optional)']
        ))
        
        # ====================================================================
        # PRIORITY 3: EDGE CASES & FAILURE MODES (For robustness)
        # ====================================================================
        
        # Scenario 8: Similar Objects Disambiguation
        scenarios.append(CollectionScenario(
            id="P3_001",
            name="Similar Objects Disambiguation",
            description="Multiple similar objects - which one to fetch?",
            priority=3,
            category="edge_case",
            target_images=200,
            target_videos=20,
            target_duration_minutes=60,
            objects_required=['multiple_cups', 'multiple_plates', 'similar_objects'],
            environment="kitchen",
            lighting_conditions=['normal'],
            viewpoints=['robot_eye_level'],
            annotation_types=['bbox', 'instance_id'],
            ground_truth_required=True,
            min_object_visibility=0.7,
            max_occlusion=0.2,
            min_image_quality='high',
            setup_instructions="""
                1. Place 3-5 very similar objects (same type, color, size)
                2. Add subtle differences (labels, minor damage, orientation)
                3. Test user disambiguation (point, describe, spatial reference)
                4. Create ambiguous scenarios
                5. Include identical objects
            """,
            capture_instructions="""
                1. Annotate each instance with unique ID
                2. Mark distinguishing features
                3. Capture spatial relationships
                4. Include user pointing/gesturing in frame
                5. Document which object user intended
            """,
            success_criteria="""
                - Each instance uniquely identified
                - Distinguishing features annotated
                - Spatial references marked
                - User intent clearly documented
            """,
            estimated_time_hours=2.0,
            difficulty="medium",
            equipment_needed=['Multiple identical/similar objects', 'participants for disambiguation']
        ))
        
        # Scenario 9: Failure Cases
        scenarios.append(CollectionScenario(
            id="P3_002",
            name="Failure Mode Collection",
            description="Deliberately collect scenarios where system fails",
            priority=3,
            category="failure_analysis",
            target_images=150,
            target_videos=30,
            target_duration_minutes=90,
            objects_required=['challenging_objects'],
            environment="various",
            lighting_conditions=['extreme', 'challenging'],
            viewpoints=['unusual_angles'],
            annotation_types=['failure_type', 'bbox'],
            ground_truth_required=True,
            min_object_visibility=0.0,  # Can be zero (object not detectable)
            max_occlusion=1.0,
            min_image_quality='low',
            setup_instructions="""
                1. Create deliberately challenging scenarios:
                   - Extreme lighting (very dark, very bright, backlit)
                   - Heavy occlusion (>70%)
                   - Motion blur
                   - Out of focus
                   - Unusual viewpoints (top-down, ground-level)
                2. Use edge-case objects (tiny, huge, unusual shapes)
                3. Test system limits
            """,
            capture_instructions="""
                1. Annotate why this case is challenging
                2. Label failure type (detection, classification, localization)
                3. Document what would make this easier
                4. Include recovery attempts
                5. Note any safety implications
            """,
            success_criteria="""
                - Failure mode clearly documented
                - Root cause analyzed
                - Potential solutions noted
                - At least 10 different failure types collected
            """,
            estimated_time_hours=2.5,
            difficulty="hard",
            equipment_needed=['Current deployed system', 'creative problem-finding mindset']
        ))
        
        # Scenario 10: Dynamic Scenes (Moving Objects)
        scenarios.append(CollectionScenario(
            id="P3_003",
            name="Dynamic Scene - Moving Objects",
            description="Objects and people in motion",
            priority=3,
            category="edge_case",
            target_images=0,  # Video-only scenario
            target_videos=100,
            target_duration_minutes=180,
            objects_required=['person', 'moving_objects'],
            environment="living_room",
            lighting_conditions=['normal'],
            viewpoints=['robot_eye_level'],
            annotation_types=['bbox_sequence', 'tracking_id'],
            ground_truth_required=True,
            min_object_visibility=0.5,
            max_occlusion=0.4,
            min_image_quality='medium',
            setup_instructions="""
                1. Record people walking through scene
                2. Objects being carried/moved
                3. Doors opening/closing
                4. Pets moving around
                5. Robot arm in motion
            """,
            capture_instructions="""
                1. Maintain consistent tracking IDs across frames
                2. Handle objects entering/leaving frame
                3. Track through brief occlusions
                4. Annotate motion blur
                5. Mark sudden appearance/disappearance
            """,
            success_criteria="""
                - Smooth tracking across frames
                - Consistent IDs maintained
                - Motion properly captured
                - At least 20 different movement patterns
            """,
            estimated_time_hours=4.0,
            difficulty="hard",
            equipment_needed=['Video annotation tool', 'participants', 'tracking software']
        ))
        
        return scenarios
    
    def get_scenarios_by_priority(self, priority: int) -> List[CollectionScenario]:
        """Get all scenarios of a given priority"""
        return [s for s in self.scenarios if s.priority == priority]
    
    def get_scenarios_by_category(self, category: str) -> List[CollectionScenario]:
        """Get all scenarios in a category"""
        return [s for s in self.scenarios if s.category == category]
    
    def generate_collection_plan(self, weeks: int = 8) -> Dict[str, Any]:
        """
        Generate a collection plan over N weeks
        
        Distributes scenarios across time with priorities
        """
        plan = {
            'duration_weeks': weeks,
            'total_scenarios': len(self.scenarios),
            'weekly_schedule': []
        }
        
        # Priority 1: Weeks 1-4
        # Priority 2: Weeks 5-6
        # Priority 3: Weeks 7-8
        
        p1_scenarios = self.get_scenarios_by_priority(1)
        p2_scenarios = self.get_scenarios_by_priority(2)
        p3_scenarios = self.get_scenarios_by_priority(3)
        
        # Distribute across weeks
        weeks_p1 = 4
        weeks_p2 = 2
        weeks_p3 = 2
        
        for week in range(weeks):
            week_plan = {
                'week': week + 1,
                'scenarios': []
            }
            
            if week < weeks_p1:
                # Priority 1 scenarios
                scenarios_this_week = p1_scenarios[week::weeks_p1]
                week_plan['scenarios'] = scenarios_this_week
                week_plan['focus'] = 'Core Operations'
            elif week < weeks_p1 + weeks_p2:
                # Priority 2 scenarios
                week_in_p2 = week - weeks_p1
                scenarios_this_week = p2_scenarios[week_in_p2::weeks_p2]
                week_plan['scenarios'] = scenarios_this_week
                week_plan['focus'] = 'Extended Operations'
            else:
                # Priority 3 scenarios
                week_in_p3 = week - weeks_p1 - weeks_p2
                scenarios_this_week = p3_scenarios[week_in_p3::weeks_p3]
                week_plan['scenarios'] = scenarios_this_week
                week_plan['focus'] = 'Edge Cases'
            
            plan['weekly_schedule'].append(week_plan)
        
        return plan
    
    def export_collection_plan(self, filename: str = "collection_plan.yaml"):
        """Export collection plan to YAML"""
        plan = self.generate_collection_plan(weeks=8)
        
        # Convert to serializable format
        export_data = {
            'duration_weeks': plan['duration_weeks'],
            'total_scenarios': plan['total_scenarios'],
            'weekly_schedule': []
        }
        
        for week_plan in plan['weekly_schedule']:
            week_data = {
                'week': week_plan['week'],
                'focus': week_plan['focus'],
                'scenarios': [
                    {
                        'id': s.id,
                        'name': s.name,
                        'description': s.description,
                        'target_images': s.target_images,
                        'estimated_hours': s.estimated_time_hours
                    } for s in week_plan['scenarios']
                ]
            }
            export_data['weekly_schedule'].append(week_data)
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"Collection plan exported to: {output_path}")
        return output_path


def main():
    """Generate and display data collection strategy"""
    print("="*80)
    print("ROBUST DATA COLLECTION STRATEGY")
    print("="*80)
    
    strategy = DataCollectionStrategy()
    
    # Show summary
    print(f"\nTotal Scenarios Defined: {len(strategy.scenarios)}")
    print(f"  Priority 1 (Critical):  {len(strategy.get_scenarios_by_priority(1))}")
    print(f"  Priority 2 (Important): {len(strategy.get_scenarios_by_priority(2))}")
    print(f"  Priority 3 (Edge Cases): {len(strategy.get_scenarios_by_priority(3))}")
    
    # Show categories
    categories = set(s.category for s in strategy.scenarios)
    print(f"\nCategories:")
    for cat in categories:
        count = len(strategy.get_scenarios_by_priority(cat))
        print(f"  - {cat}: {count} scenarios")
    
    # Calculate totals
    total_images = sum(s.target_images for s in strategy.scenarios)
    total_videos = sum(s.target_videos for s in strategy.scenarios)
    total_hours = sum(s.estimated_time_hours for s in strategy.scenarios)
    
    print(f"\nTotal Collection Targets:")
    print(f"  Images: {total_images:,}")
    print(f"  Videos: {total_videos:,}")
    print(f"  Estimated Time: {total_hours:.1f} hours ({total_hours/8:.1f} days)")
    
    # Generate 8-week plan
    plan = strategy.generate_collection_plan(weeks=8)
    print(f"\n8-Week Collection Plan:")
    for week_plan in plan['weekly_schedule']:
        print(f"\n  Week {week_plan['week']}: {week_plan['focus']}")
        for scenario in week_plan['scenarios']:
            print(f"    - {scenario.id}: {scenario.name} ({scenario.estimated_time_hours:.1f}h)")
    
    # Export plan
    strategy.export_collection_plan()
    
    print("\n" + "="*80)
    print("Data collection strategy ready!")
    print("Next steps:")
    print("  1. Review scenario definitions")
    print("  2. Assign data collectors")
    print("  3. Set up annotation infrastructure (Label Studio)")
    print("  4. Begin Priority 1 scenarios")
    print("="*80)


if __name__ == "__main__":
    main()

