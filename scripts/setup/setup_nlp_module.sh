#!/bin/bash

# ============================================================================
# NLP Module Setup Script
# Automates installation of all NLP components
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "Humanoid Robot Assistant - NLP Module Setup"
echo "============================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo -e "${RED}✗ Python 3.10+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install core dependencies
echo ""
echo "Installing core NLP dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # CPU version
pip install transformers sentence-transformers

# Install entity extraction dependencies
echo ""
echo "Installing entity extraction (spaCy)..."
pip install spacy
python -m spacy download en_core_web_sm
echo -e "${GREEN}✓ spaCy model downloaded${NC}"

# Install dialogue management
echo ""
echo "Installing dialogue management..."
pip install redis python-statemachine

# Install emotion detection
echo ""
echo "Installing emotion detection..."
pip install vaderSentiment

# Install RAG dependencies
echo ""
echo "Installing RAG components (LangChain, FAISS)..."
pip install langchain langchain-community langchain-openai
pip install faiss-cpu  # Use faiss-gpu if you have NVIDIA GPU
pip install llama-index  # Fallback framework

# Install LLM clients
echo ""
echo "Installing LLM clients..."
pip install openai
pip install ollama

# Install ASR
echo ""
echo "Installing ASR (Faster-Whisper, Vosk)..."
pip install faster-whisper
pip install vosk

# Install TTS
echo ""
echo "Installing TTS (ElevenLabs, Coqui, pyttsx3)..."
pip install elevenlabs
pip install TTS
pip install pyttsx3

# Install utilities
echo ""
echo "Installing utilities..."
pip install soundfile pydub
pip install python-dotenv pyyaml
pip install loguru

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python3 -c "import torch; print('✓ GPU Available' if torch.cuda.is_available() else '✗ No GPU detected (using CPU)')"

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p data/vector_store/faiss
mkdir -p audio_output
mkdir -p models/nlp
mkdir -p logs
echo -e "${GREEN}✓ Directories created${NC}"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file..."
    cat > .env << EOF
# OpenAI API Key (optional - for Tier 1 LLM)
OPENAI_API_KEY=

# ElevenLabs API Key (optional - for Tier 1 TTS)
ELEVENLABS_API_KEY=

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Database Configuration
MONGODB_URI=mongodb://localhost:27017
POSTGRES_HOST=localhost
POSTGRES_USER=robot
POSTGRES_PASSWORD=changeme
EOF
    echo -e "${GREEN}✓ .env file created${NC}"
    echo -e "${YELLOW}⚠ Edit .env file and add your API keys${NC}"
fi

# Optional: Install Ollama
echo ""
echo "============================================================================"
echo "Optional: Install Ollama for local LLM (Tier 2)"
echo "============================================================================"
echo "Visit: https://ollama.ai/download"
echo "After installing, run:"
echo "  ollama pull llama3.2:3b"
echo ""

# Test installation
echo ""
echo "============================================================================"
echo "Testing Installation"
echo "============================================================================"
echo ""

echo "Testing Entity Extractor..."
python3 -c "from src.nlp.entities import EntityExtractor; e = EntityExtractor(); print('✓ Entity Extractor OK')"

echo "Testing Dialogue Manager..."
python3 -c "from src.nlp.dialogue import DialogueManager; d = DialogueManager(); print('✓ Dialogue Manager OK')"

echo "Testing Emotion Detector..."
python3 -c "from src.nlp.emotion import EmotionDetector; e = EmotionDetector(); print('✓ Emotion Detector OK')"

echo ""
echo "============================================================================"
echo -e "${GREEN}✓ NLP Module Setup Complete!${NC}"
echo "============================================================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env file and add API keys (optional)"
echo "  2. Install Ollama if you want local LLM: https://ollama.ai"
echo "  3. Start Redis if using dialogue management: redis-server"
echo "  4. Test the system: python src/nlp/nlp_service.py"
echo ""
echo "For full documentation, see:"
echo "  - INSTALLATION.md (this file's documentation)"
echo "  - src/nlp/README.md (NLP module overview)"
echo "  - docs/GETTING_STARTED.md (general guide)"
echo ""

