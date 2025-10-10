# System Architecture: Humanoid Robot Assistant Brain

## Table of Contents
1. [Overview](#overview)
2. [Architectural Principles](#architectural-principles)
3. [System Components](#system-components)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Deployment Architecture](#deployment-architecture)
7. [Integration Points](#integration-points)

---

## Overview

The Humanoid Robot Assistant Brain is a distributed, cloud-edge hybrid AI system that provides:
- **Multimodal Perception**: Vision, language, audio, proprioception
- **Intelligent Reasoning**: Task planning, decision-making, memory
- **Safe Execution**: Multi-layer safety checks, fail-safes, explainability
- **Continuous Learning**: MLOps pipeline for model improvement

### Design Goals
- ✅ **Low Latency**: <100ms perception-to-action for safety-critical operations
- ✅ **High Reliability**: 99.9% uptime with graceful degradation
- ✅ **Scalability**: From single robot to fleet management
- ✅ **Safety-First**: Multiple redundant safety layers
- ✅ **Explainability**: Every decision can be traced and explained

---

## Architectural Principles

### 1. Microservices Architecture
Each major AI component runs as an independent service:
- Independently deployable and scalable
- Fault isolation (one service failure doesn't crash system)
- Technology flexibility (can use best tool for each job)
- Clear API contracts (gRPC for performance, REST for flexibility)

### 2. Cloud-Edge Hybrid
```
┌─────────────────────────────────────────────────────────────┐
│                         CLOUD                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • Heavy model training (A100/H100)                 │   │
│  │  • Large LLM inference (GPT-4, Llama-70B)          │   │
│  │  • Data aggregation & analytics                     │   │
│  │  • Model registry & serving                         │   │
│  │  • Fleet management & monitoring                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↕                                   │
│                 Secure WebSocket/gRPC                        │
│                          ↕                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    EDGE (Robot)                      │   │
│  │  ┌──────────────────────────────────────────────┐   │   │
│  │  │  • Real-time perception (YOLOv8, SAM)       │   │   │
│  │  │  • Small LLM (Llama-7B quantized)           │   │   │
│  │  │  • Low-level control & safety                │   │   │
│  │  │  • SLAM & localization                       │   │   │
│  │  │  • Offline capability (fallback policies)   │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Edge runs:**
- Real-time perception (vision, depth, pose)
- Safety-critical decision making
- Low-latency motor control
- Offline fallback behaviors

**Cloud runs:**
- Complex reasoning tasks
- Model training and updates
- Data aggregation from fleet
- Heavy NLP (large LLMs)

### 3. Layered Safety
```
Layer 4: Human Override (E-stop, remote disable)
Layer 3: High-Level Safety (task-level constraints)
Layer 2: Mid-Level Safety (collision avoidance, workspace limits)
Layer 1: Low-Level Safety (joint limits, force limits, watchdogs)
Layer 0: Hardware Safety (mechanical stops, compliant actuators)
```

### 4. Modular & Composable
- Plug-and-play components
- Standard interfaces (ROS2 messages, gRPC services)
- Easy to add new capabilities
- Support multiple robot platforms

---

## System Components

### Component Diagram

```
┌───────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR SERVICE                        │
│              (Coordinates all services, manages state)             │
└────────────┬───────────────┬──────────────┬─────────────┬─────────┘
             │               │              │             │
    ┌────────▼─────┐  ┌──────▼──────┐  ┌───▼─────┐  ┌───▼──────┐
    │  NLP Service │  │Vision Service│  │Multimodal│ │Planning  │
    │              │  │              │  │ Service  │  │ Service  │
    │ • Dialogue   │  │ • Detection  │  │          │  │          │
    │ • Intent     │  │ • Segment    │  │ • VLM    │  │ • Task   │
    │ • Emotion    │  │ • Pose       │  │ • Fusion │  │   Plan   │
    │ • RAG        │  │ • Depth      │  │ • Ground │  │ • Motion │
    │ • ASR/TTS    │  │ • Track      │  │ • VQA    │  │   Plan   │
    └──────┬───────┘  └──────┬───────┘  └────┬─────┘  └────┬─────┘
           │                 │               │             │
           └─────────────────┴───────────────┴─────────────┘
                             │
                   ┌─────────▼──────────┐
                   │   Memory Service   │
                   │                    │
                   │ • Episodic Memory  │
                   │ • Semantic Memory  │
                   │ • Working Memory   │
                   │ • Vector Store     │
                   └────────────────────┘
```

### 1. NLP Service

**Purpose**: Understand and generate natural language

**Components**:
- **Dialogue Manager**: Multi-turn conversation, context tracking
- **Intent Classifier**: Recognize user intentions (inherited from chapo-bot)
- **Entity Extractor**: NER for tasks, objects, locations, times
- **Emotion Detector**: Affective computing for empathetic responses
- **RAG System**: Knowledge retrieval for factual responses
- **LLM Integration**: On-device (Llama-7B) + cloud (GPT-4/Llama-70B)
- **ASR**: Whisper or similar for speech-to-text
- **TTS**: VITS or Bark for natural speech synthesis

**Key Features**:
- Evolved from chapo-bot intent routing
- Added transformer-based intent classification (BERT/RoBERTa fine-tuned)
- RAG for grounded responses (not hallucinating)
- Fallback to rule-based for safety-critical commands

**API**:
```python
POST /nlp/understand
{
  "text": "Can you bring me the red mug from the kitchen?",
  "session_id": "user_123",
  "context": {"location": "living_room"}
}

Response:
{
  "intent": "fetch_object",
  "entities": {
    "object": "mug",
    "color": "red",
    "location": "kitchen"
  },
  "confidence": 0.94,
  "emotion": "neutral",
  "response": "Sure, I'll bring you the red mug from the kitchen."
}
```

### 2. Vision Service

**Purpose**: Real-time visual perception

**Components**:
- **Object Detection**: YOLOv8/YOLO-NAS for real-time detection
- **Segmentation**: SAM for zero-shot segmentation
- **Pose Estimation**: MediaPipe/HRNet for human and object poses
- **Depth Estimation**: Stereo matching or monocular depth (MiDaS/Depth Anything)
- **Tracking**: ByteTrack or DeepSORT for multi-object tracking
- **Scene Understanding**: Panoptic segmentation, room layout

**Optimization**:
- TensorRT conversion for all models
- INT8 quantization where possible
- Dynamic batching for multiple cameras
- Efficient memory management (CUDA streams)

**API**:
```python
POST /vision/detect
{
  "image": "base64_encoded_image",
  "tasks": ["detection", "segmentation", "depth"],
  "confidence_threshold": 0.5
}

Response:
{
  "detections": [
    {"class": "person", "bbox": [x1, y1, x2, y2], "confidence": 0.92},
    {"class": "mug", "bbox": [x1, y1, x2, y2], "confidence": 0.88}
  ],
  "segmentation_mask": "...",
  "depth_map": "...",
  "latency_ms": 45
}
```

### 3. Multimodal Service

**Purpose**: Fuse information from multiple modalities

**Components**:
- **Vision-Language Models**: CLIP, BLIP-2, LLaVA for VLM tasks
- **Cross-Modal Attention**: Align vision, language, audio, touch
- **Visual Grounding**: "bring me that" → identify object
- **VQA**: "What color is the mug?" → "red"
- **Affordance Prediction**: What actions are possible with objects

**Key Capabilities**:
- Resolve ambiguous references ("that", "the one over there")
- Answer visual questions
- Understand spatial relationships
- Predict interaction outcomes

**API**:
```python
POST /multimodal/ground
{
  "text": "Pick up the red one",
  "image": "base64_image",
  "detected_objects": [...]
}

Response:
{
  "grounded_object": {
    "id": "obj_42",
    "class": "mug",
    "bbox": [...],
    "confidence": 0.91
  }
}
```

### 4. Planning Service

**Purpose**: High-level task and low-level motion planning

**Components**:
- **Task Planner**: Hierarchical task decomposition (PDDL or learned)
- **Motion Planner**: MoveIt2 + trajectory optimization
- **Collision Avoidance**: Real-time obstacle avoidance
- **Grasp Planner**: 6-DoF grasp pose generation
- **Navigation**: Path planning with dynamic obstacles

**Integration**:
- Takes goals from NLP/Multimodal
- Queries Vision for scene understanding
- Executes through ROS2 action servers
- Monitors execution with Perception feedback

**API**:
```python
POST /planning/plan_task
{
  "task": "fetch_object",
  "target_object": "mug_red_42",
  "goal_location": "living_room_table",
  "constraints": ["avoid_humans", "gentle_grasp"]
}

Response:
{
  "plan": [
    {"action": "navigate_to", "target": "kitchen", "duration": 5.2},
    {"action": "grasp_object", "target": "mug_red_42", "duration": 2.1},
    {"action": "navigate_to", "target": "living_room", "duration": 6.8},
    {"action": "place_object", "location": "table", "duration": 1.9}
  ],
  "estimated_duration": 16.0,
  "success_probability": 0.89
}
```

### 5. Perception Service (SLAM + Sensor Fusion)

**Purpose**: Localization, mapping, and sensor integration

**Components**:
- **Visual SLAM**: ORB-SLAM3 or similar
- **IMU Fusion**: Visual-inertial odometry
- **Occupancy Mapping**: 2D/3D maps for navigation
- **Place Recognition**: Where am I?
- **Dynamic Object Tracking**: Track moving obstacles

**API**:
```python
GET /perception/pose

Response:
{
  "position": {"x": 2.3, "y": 1.5, "z": 0.0},
  "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 1.57},
  "confidence": 0.96,
  "map_frame": "world"
}
```

### 6. Memory Service

**Purpose**: Store and retrieve experiences and knowledge

**Components**:
- **Episodic Memory**: Past experiences (evolved from chapo-bot)
- **Semantic Memory**: Facts, object properties, spatial layouts
- **Working Memory**: Current context, active goals
- **Vector Store**: FAISS/Milvus for similarity search

**Features**:
- Remember past interactions with users
- Learn object locations over time
- Retrieve relevant experiences for current situation
- Support RAG for NLP

**API**:
```python
POST /memory/store
{
  "type": "episodic",
  "event": "user_asked_to_fetch_mug",
  "timestamp": "2025-10-09T10:30:00Z",
  "context": {...},
  "outcome": "success"
}

POST /memory/query
{
  "query": "Where did I last see the red mug?",
  "type": "semantic",
  "k": 5
}

Response:
{
  "results": [
    {"location": "kitchen_counter", "timestamp": "...", "confidence": 0.92}
  ]
}
```

### 7. Safety Service

**Purpose**: Monitor and enforce safety constraints

**Components**:
- **Watchdog**: Monitor service health, restart if needed
- **Anomaly Detection**: Detect unexpected sensor readings or behaviors
- **Constraint Checker**: Ensure actions satisfy safety rules
- **Explainability**: Generate explanations for decisions
- **E-stop Handler**: Emergency stop coordination

**Features**:
- Real-time safety monitoring
- Pre-execution safety checks
- Post-execution validation
- Audit logging for all actions
- Human override interface

**API**:
```python
POST /safety/check
{
  "action": "grasp_object",
  "parameters": {...},
  "context": {"nearby_humans": true}
}

Response:
{
  "safe": true,
  "confidence": 0.94,
  "warnings": ["Human detected nearby - reduce speed"],
  "constraints": {"max_velocity": 0.5, "max_force": 10.0}
}
```

### 8. Orchestrator Service

**Purpose**: Coordinate all services and manage system state

**Responsibilities**:
- Route requests to appropriate services
- Manage service dependencies and sequencing
- Handle failures and retries
- Maintain global system state
- Provide unified API to robot control layer

**State Machine**:
```
IDLE → LISTENING → UNDERSTANDING → PLANNING → EXECUTING → IDLE
                      ↓                             ↓
                   ERROR ←──────────────────────── ERROR
                      ↓
                  RECOVERY → IDLE
```

---

## Data Flow

### Example: "Bring me the red mug from the kitchen"

```
1. [User Speech] → ASR (Whisper)
   ↓
2. [Text] → NLP Service (Intent + Entity Extraction)
   Intent: fetch_object
   Entities: {object: mug, color: red, location: kitchen}
   ↓
3. [Intent + Entities] → Orchestrator
   ↓
4. Orchestrator queries:
   a. Vision Service: "Find red mug in kitchen"
   b. Memory Service: "Where is kitchen? Any known red mugs?"
   c. Multimodal Service: "Ground 'red mug' in current scene"
   ↓
5. [Object Grounded] → Planning Service
   Generate task plan:
   - Navigate to kitchen
   - Detect red mug
   - Grasp red mug
   - Navigate to user
   - Hand over mug
   ↓
6. [Plan] → Safety Service: Check plan safety
   ↓
7. [Safe Plan] → Execute via ROS2 Action Servers
   Each step:
   - Vision provides feedback
   - Perception tracks robot pose
   - Safety monitors execution
   ↓
8. [Task Complete] → Memory Service: Store episode
   ↓
9. [Success] → NLP Service: Generate response via TTS
   "Here's your red mug!"
```

---

## Technology Stack

### Languages
- **Python**: Primary for AI/ML (PyTorch, TensorFlow)
- **C++**: Performance-critical ROS2 nodes, low-level control
- **CUDA**: Custom kernels for vision pipeline

### Deep Learning
- **PyTorch**: Training and inference
- **TensorRT**: Inference optimization (NVIDIA)
- **ONNX**: Model interchange format
- **Triton Inference Server**: Multi-model serving

### NLP
- **Transformers** (Hugging Face): BERT, GPT, Llama
- **LangChain**: RAG orchestration
- **Whisper**: ASR
- **VITS/Bark**: TTS
- **spaCy**: Lightweight NLP, NER fallback

### Computer Vision
- **YOLOv8/YOLO-NAS**: Object detection
- **SAM**: Segmentation
- **MediaPipe**: Pose estimation
- **OpenCV**: Image processing
- **DINOv2**: Feature extraction
- **ORB-SLAM3**: SLAM

### Robotics
- **ROS2 Humble**: Middleware
- **MoveIt2**: Motion planning
- **Nav2**: Navigation stack
- **ros2_control**: Low-level control interface

### MLOps
- **MLflow**: Experiment tracking, model registry
- **DVC**: Dataset versioning
- **Kubeflow**: ML pipeline orchestration
- **Weights & Biases**: Experiment visualization
- **Prometheus + Grafana**: Monitoring

### Data
- **MongoDB**: Document store (logs, sessions)
- **PostgreSQL**: Relational data (users, tasks)
- **FAISS/Milvus**: Vector database (embeddings)
- **Redis**: Caching, message queue

### Deployment
- **Docker**: Containerization
- **Kubernetes (K3s)**: Orchestration
- **Helm**: Package management
- **Terraform**: Infrastructure as Code
- **GitHub Actions**: CI/CD

---

## Deployment Architecture

### Development Environment
```
laptop/workstation
├── Code (VSCode/PyCharm)
├── Local Docker containers (services)
├── Simulation (Isaac Sim / Gazebo)
└── Remote training (cloud GPU)
```

### Edge Deployment (Robot)
```
NVIDIA Jetson Orin AGX (32GB)
├── Docker containers
│   ├── vision_service (TensorRT models)
│   ├── nlp_service (quantized LLM)
│   ├── perception_service (SLAM)
│   ├── orchestrator
│   └── safety_service
├── ROS2 nodes (native)
│   ├── Camera drivers
│   ├── Motor controllers
│   └── Sensor interfaces
└── Monitoring agent
```

### Cloud Deployment
```
Kubernetes Cluster (AWS/GCP/Azure)
├── Training namespace
│   ├── PyTorch DDP jobs (A100 pods)
│   ├── Data preprocessing
│   └── Hyperparameter tuning
├── Inference namespace
│   ├── Triton Inference Server
│   ├── Large LLM service
│   └── Model registry
├── MLOps namespace
│   ├── MLflow server
│   ├── DVC remote storage
│   └── Monitoring dashboards
└── API Gateway
    └── Robot fleet management
```

### Network Architecture
```
[Robot Edge] ←→ [VPN/Secure Tunnel] ←→ [Cloud]
    ↓                                       ↓
[Local WiFi]                          [Load Balancer]
    ↓                                       ↓
[User Devices]                        [Services]
```

---

## Integration Points

### ROS2 Integration
- Each service publishes/subscribes to ROS2 topics
- Action servers for long-running tasks
- Service calls for synchronous requests
- TF2 for coordinate transformations

### Hardware Integration
- Camera drivers (RealSense, Kinect)
- Motor controllers (via ros2_control)
- IMU, force/torque sensors
- Microphone arrays, speakers

### External APIs
- Weather, news (for user queries)
- Cloud LLMs (OpenAI, Anthropic)
- Map services (Google Maps)
- Smart home integration (optional)

---

## Performance Requirements

| Component | Latency Target | Throughput |
|-----------|----------------|------------|
| Vision (detection) | <50ms | 20 FPS |
| NLP (intent) | <200ms | N/A |
| Planning (motion) | <500ms | N/A |
| Safety check | <10ms | Real-time |
| SLAM | <100ms | 10 Hz |
| End-to-end | <2s | N/A |

---

## Next Steps

See implementation details in:
- `src/nlp/README.md` - NLP implementation
- `src/vision/README.md` - Vision implementation
- `services/README.md` - Microservices architecture
- `deployment/README.md` - Deployment guides

---

**Author**: Victor Ibhafidon  
**Organization**: Xtainless Technologies  
**Date**: October 9, 2025  
**License**: MIT

