# Getting Started - Humanoid Robot Assistant

## 🚀 Quick Start (5 Minutes)

This guide will help you set up the development environment and run your first test.

### Prerequisites

**Hardware Requirements:**
- **Development**: x86_64 laptop/workstation with NVIDIA GPU (8GB+ VRAM)
- **Edge Deployment**: NVIDIA Jetson Orin AGX (32GB recommended)
- **Cloud Training**: NVIDIA A100/H100 GPUs (via AWS/GCP/Azure)

**Software Requirements:**
- Ubuntu 22.04 LTS (recommended) or Ubuntu 20.04
- Python 3.10+
- Docker 24.0+
- NVIDIA Docker Runtime
- CUDA 11.8+ or 12.x
- ROS2 Humble (for hardware integration)

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd humaniod_robot_assitant
```

### Step 2: Environment Setup

#### Option A: Docker (Recommended for Development)

```bash
# Create .env file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or vim, code, etc.

# Required variables:
# - OPENAI_API_KEY (for cloud LLM)
# - MONGODB_URI (or use local via Docker)
# - WEATHER_API_KEY, NEWS_API_KEY (optional)

# Build and start services
cd deployment/docker
docker-compose up -d

# Check services are running
docker-compose ps
```

#### Option B: Local Python Environment

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install spaCy models
python -m spacy download en_core_web_sm

# Set environment variables
export PYTHONPATH=$(pwd):$PYTHONPATH
cp .env.example .env
# Edit .env with your keys
source .env
```

### Step 3: Download Models

```bash
# Create models directory
mkdir -p models/{nlp,vision,multimodal}

# Download NLP models (examples)
# Intent classifier
wget https://huggingface.co/robot-brain/intent-classifier-v1/resolve/main/intent_bert_v1.onnx \
  -O models/nlp/intent_bert_v1.onnx

# Or train your own (see Training Guide)
python training/nlp/train_intent_classifier.py
```

### Step 4: Run Your First Test

#### Test 1: Intent Classification

```bash
# Run intent classifier test
python src/nlp/intent/classifier.py
```

**Expected Output:**
```
========================================
Intent Classifier Test
========================================

Input: STOP RIGHT NOW
  Intent: emergency_stop
  Confidence: 1.00
  Safety Critical: True
  Used Fallback: True

Input: Hello! How are you?
  Intent: greeting
  Confidence: 0.90
  Safety Critical: False
  Used Fallback: True

...
```

#### Test 2: NLP Service (API)

```bash
# Start NLP service
python -m uvicorn services.nlp_service.src.main:app --reload --port 8001

# In another terminal, test the API
curl -X POST http://localhost:8001/nlp/understand \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Can you bring me a cup of water?",
    "session_id": "test_123"
  }'
```

**Expected Response:**
```json
{
  "intent": {
    "name": "fetch_object",
    "confidence": 0.94
  },
  "entities": {
    "object_type": "cup",
    "object_content": "water"
  },
  "emotion": {
    "primary": "neutral",
    "confidence": 0.87
  }
}
```

#### Test 3: End-to-End System (Simulation)

```bash
# Start all services (if using Docker)
docker-compose up -d

# Run integration test
pytest tests/integration/test_e2e_fetch_object.py -v

# Or run interactive CLI
python scripts/interactive_cli.py
```

---

## 📚 Next Steps

### 1. Explore the Architecture

Read these documents in order:
1. [`docs/architecture/SYSTEM_ARCHITECTURE.md`](architecture/SYSTEM_ARCHITECTURE.md) - Overall system design
2. [`src/nlp/README.md`](../src/nlp/README.md) - NLP module details
3. [`src/vision/README.md`](../src/vision/README.md) - Vision module details

### 2. Configure Your Robot

Edit configuration files:
- `configs/base/system_config.yaml` - System-wide settings
- `configs/development/robot_config.yaml` - Robot-specific parameters
- `configs/base/hardware_config.yaml` - Sensor and actuator definitions

### 3. Train Your First Model

Follow the training guide:
```bash
# Train intent classifier on custom data
python training/nlp/train_intent_classifier.py \
  --data data/datasets/intents_custom.csv \
  --output models/nlp/intent_custom_v1.onnx
```

See [`docs/training/NLP_TRAINING.md`](training/NLP_TRAINING.md) for details.

### 4. Deploy to Edge Device (Jetson)

```bash
# On your Jetson Orin
cd deployment/edge
./scripts/setup_jetson.sh

# Build optimized models
./scripts/build_tensorrt_models.sh

# Deploy services
docker-compose -f docker-compose.jetson.yml up -d
```

See [`docs/deployment/EDGE_DEPLOYMENT.md`](deployment/EDGE_DEPLOYMENT.md) for details.

---

## 🧪 Running Tests

### Unit Tests
```bash
# All unit tests
pytest tests/unit/ -v

# Specific module
pytest tests/unit/nlp/ -v
pytest tests/unit/vision/ -v
```

### Integration Tests
```bash
# All integration tests
pytest tests/integration/ -v

# End-to-end tests
pytest tests/e2e/ -v --slow
```

### Performance Benchmarks
```bash
# Benchmark NLP
python benchmarks/nlp_benchmark.py

# Benchmark vision
python benchmarks/vision_benchmark.py

# Full system benchmark
python benchmarks/end_to_end_benchmark.py
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA/GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 2. Docker GPU Access

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

#### 3. Model Loading Errors

```bash
# Check model files exist
ls -lh models/nlp/
ls -lh models/vision/

# Download missing models
python scripts/download_models.py --all

# Or download specific model
python scripts/download_models.py --model intent_classifier
```

#### 4. ROS2 Integration Issues

```bash
# Check ROS2 installation
ros2 --version

# Source ROS2
source /opt/ros/humble/setup.bash

# Add to ~/.bashrc for persistence
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

#### 5. Port Already in Use

```bash
# Find process using port
sudo lsof -i :8001

# Kill process
sudo kill -9 <PID>

# Or change port in config
# Edit configs/base/system_config.yaml
```

---

## 📖 Documentation Structure

```
docs/
├── architecture/          # System architecture
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── NLP_ARCHITECTURE.md
│   └── VISION_ARCHITECTURE.md
├── api/                   # API documentation
│   ├── NLP_API.md
│   ├── VISION_API.md
│   └── ORCHESTRATOR_API.md
├── training/              # Training guides
│   ├── NLP_TRAINING.md
│   ├── VISION_TRAINING.md
│   └── MULTIMODAL_TRAINING.md
├── deployment/            # Deployment guides
│   ├── EDGE_DEPLOYMENT.md
│   ├── CLOUD_DEPLOYMENT.md
│   └── KUBERNETES.md
├── research_paper/        # Academic paper
│   ├── 00_PAPER_OUTLINE.md
│   └── BIBLIOGRAPHY.md
└── governance/            # Ethics & safety
    ├── ETHICS.md
    ├── SAFETY.md
    └── PRIVACY.md
```

---

## 🤝 Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow coding standards (PEP 8, type hints)
- Add unit tests for new code
- Update documentation

### 3. Run Tests & Linters

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/
mypy src/

# Test
pytest tests/unit/ -v
```

### 4. Commit & Push

```bash
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

### 5. Create Pull Request

Open PR on GitHub/GitLab and request review.

---

## 🆘 Getting Help

- **Documentation**: [`docs/`](../docs/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: robot-ai-team@example.com
- **Slack**: #robot-brain-dev

---

## 🎯 Learning Path

**Week 1-2: Foundations**
- [ ] Set up development environment
- [ ] Run all basic tests
- [ ] Explore NLP module
- [ ] Explore vision module

**Week 3-4: Training**
- [ ] Train custom intent classifier
- [ ] Train object detection model
- [ ] Fine-tune LLM for robot tasks

**Week 5-6: Integration**
- [ ] Integrate with simulation
- [ ] Test multimodal fusion
- [ ] Deploy to Jetson

**Week 7-8: Advanced**
- [ ] Implement custom task planner
- [ ] Add new sensors
- [ ] Deploy to cloud
- [ ] Fleet management

---

## 🌟 Success Criteria

You're ready to move forward when you can:
- ✅ Run all services locally
- ✅ Send API requests to all services
- ✅ Train a simple model
- ✅ Run end-to-end tests
- ✅ Deploy to Docker containers
- ✅ Understand the architecture

---

**Happy Building! 🤖🚀**

For detailed architecture and research background, see the [Research Paper Outline](research_paper/00_PAPER_OUTLINE.md).

