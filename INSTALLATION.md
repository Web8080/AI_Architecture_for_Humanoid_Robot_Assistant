# Installation Guide - Humanoid Robot Assistant NLP Module

## Quick Start

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA 11.8+ (optional, will auto-detect)
- 16GB+ RAM (32GB recommended for full stack)
- PostgreSQL (optional, for session storage)
- Redis (optional, for dialogue management)

---

## 1. Basic Installation (CPU-only, Development)

```bash
# Clone repository
git clone https://github.com/Web8080/AI_Architecture_for_Humanoid_Robot_Assistant.git
cd humaniod_robot_assitant

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install basic dependencies
pip install --upgrade pip
pip install -r requirements-basic.txt
```

---

## 2. Component-by-Component Installation

### Entity Extraction
```bash
# Install transformers and spaCy
pip install transformers torch spacy

# Download spaCy model
python -m spacy download en_core_web_sm  # Small model (CPU-friendly)
# OR
python -m spacy download en_core_web_trf  # Transformer model (better accuracy)
```

### Dialogue Management
```bash
# Install Redis and state machine
pip install redis python-statemachine

# Start Redis (local)
# On macOS: brew services start redis
# On Linux: sudo systemctl start redis
# On Windows: Download from redis.io or use WSL
```

### Emotion Detection
```bash
# Install transformers (already done above)
# Install VADER for fallback
pip install vaderSentiment
```

### RAG System
```bash
# Install LangChain (primary)
pip install langchain langchain-community langchain-openai

# Install LlamaIndex (fallback)
pip install llama-index

# Install vector databases
pip install faiss-cpu  # For CPU
# OR
pip install faiss-gpu  # For GPU (requires CUDA)

# Optional: Qdrant
pip install qdrant-client

# Install sentence transformers for embeddings
pip install sentence-transformers
```

### LLM Integration
```bash
# Install OpenAI client
pip install openai

# Install Ollama Python client
pip install ollama

# Install Ollama (local LLM server)
# Visit: https://ollama.ai/download
# Then pull a model:
ollama pull llama3.2:3b
# OR
ollama pull phi3:mini
```

### ASR (Speech Recognition)
```bash
# Install Faster-Whisper (primary)
pip install faster-whisper

# OR OpenAI Whisper (alternative)
pip install openai-whisper

# Install Vosk (fallback for streaming)
pip install vosk

# Download Vosk model (optional)
# Visit: https://alphacephei.com/vosk/models
# Download vosk-model-small-en-us-0.15.zip
# Extract to: ./models/nlp/vosk-model-en-us
```

### TTS (Text-to-Speech)
```bash
# Install ElevenLabs (cloud, primary)
pip install elevenlabs

# Install Coqui TTS (local, fallback)
pip install TTS

# Install pyttsx3 (offline fallback)
pip install pyttsx3

# On Linux, pyttsx3 may need:
sudo apt-get install espeak espeak-data libespeak-dev
```

---

## 3. Full Installation (All Components)

```bash
# Install all requirements
pip install -r requirements.txt

# Download models
python scripts/setup/download_models.py

# Install spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf  # Optional, for better accuracy

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
#   OPENAI_API_KEY=your_key_here
#   ELEVENLABS_API_KEY=your_key_here
```

---

## 4. GPU Installation (NVIDIA Jetson / Cloud GPU)

### For NVIDIA Jetson Orin
```bash
# Install JetPack 5.1+ (includes CUDA, cuDNN, TensorRT)
# Follow: https://developer.nvidia.com/embedded/jetpack

# Install PyTorch for Jetson
# Follow: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

# Install requirements (some packages need ARM64 wheels)
pip install -r requirements-jetson.txt

# Use GPU-optimized packages
pip install faiss-gpu
pip install onnxruntime-gpu
```

### For Cloud GPU (x86_64 + CUDA)
```bash
# Ensure CUDA 11.8+ or 12.x is installed
nvidia-smi  # Check CUDA version

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install GPU-accelerated packages
pip install faiss-gpu
pip install onnxruntime-gpu

# Install full requirements
pip install -r requirements.txt
```

---

## 5. Verification

### Test Each Component

```bash
# Test Entity Extractor
python src/nlp/entities/extractor.py

# Test Dialogue Manager
python src/nlp/dialogue/manager.py

# Test Emotion Detector
python src/nlp/emotion/detector.py

# Test RAG System
python src/nlp/rag/retriever.py

# Test LLM Integration
python src/nlp/llm/integrator.py

# Test ASR
python src/nlp/asr/recognizer.py

# Test TTS
python src/nlp/tts/synthesizer.py

# Test Full NLP Service
python src/nlp/nlp_service.py
```

### Check System Status
```python
from src.nlp.nlp_service import NLPService

service = NLPService()
status = service.get_detailed_status()
print(status)
```

---

## 6. Configuration

Edit `configs/base/system_config.yaml` to customize:

```yaml
nlp:
  entity_extractor:
    tier1_model: "dslim/bert-base-NER"
    use_gpu: true  # or false for CPU
  
  llm:
    tier1_openai:
      enabled: true
      model: "gpt-4o-mini"
    tier2_ollama:
      enabled: true
      model: "llama3.2:3b"
```

---

## 7. Troubleshooting

### GPU Not Detected
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Redis Connection Failed
```bash
# Check if Redis is running
redis-cli ping

# Start Redis
sudo systemctl start redis  # Linux
brew services start redis   # macOS
```

### Ollama Connection Failed
```bash
# Check if Ollama is running
ollama list

# Start Ollama server
ollama serve

# Pull a model
ollama pull llama3.2:3b
```

### Models Not Found
```bash
# Download models
python scripts/setup/download_models.py

# Or manually download spaCy models
python -m spacy download en_core_web_sm
```

---

## 8. Minimal Installation (CPU-only, No External Services)

For testing without Redis, Ollama, or API keys:

```bash
# Minimal dependencies
pip install torch transformers spacy vaderSentiment pyttsx3

# Download spaCy model
python -m spacy download en_core_web_sm

# Test (will use all Tier 3 fallbacks)
python src/nlp/nlp_service.py
```

**This will work with:**
- ✓ Entity extraction (spaCy fallback)
- ✓ Dialogue management (in-memory fallback)
- ✓ Emotion detection (VADER fallback)
- ✓ TTS (pyttsx3 fallback)
- ✗ RAG (requires sentence-transformers)
- ✗ LLM (template-based only)
- ✗ ASR (needs whisper or vosk)

---

## 9. Docker Installation

```bash
# Build Docker image
cd deployment/docker
docker-compose build nlp_service

# Run NLP service
docker-compose up nlp_service

# Or run entire stack
docker-compose up
```

---

## 10. Environment Variables

Create `.env` file in project root:

```bash
# OpenAI (optional, for Tier 1 LLM)
OPENAI_API_KEY=sk-...

# ElevenLabs (optional, for Tier 1 TTS)
ELEVENLABS_API_KEY=...

# Redis (optional, defaults to localhost)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Database (optional)
MONGODB_URI=mongodb://localhost:27017
POSTGRES_HOST=localhost
POSTGRES_USER=robot
POSTGRES_PASSWORD=changeme
```

---

## Next Steps

After installation:
1. Read `src/nlp/README.md` for architecture overview
2. Check `docs/GETTING_STARTED.md` for usage examples
3. Review `configs/base/system_config.yaml` for configuration options
4. Run tests: `pytest tests/unit/nlp/`

---

**For issues, check:**
- GitHub Issues: https://github.com/Web8080/AI_Architecture_for_Humanoid_Robot_Assistant/issues
- Documentation: `docs/`

