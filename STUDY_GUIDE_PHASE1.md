# Study Guide - Phase 1: Core NLP Module

**Author**: Victor Ibhafidon  
**Organization**: Xtainless Technologies  
**Phase**: 1 (Core NLP Module - Weeks 5-12)  
**Estimated Study Time**: 2-3 weeks before heavy coding

---

## 🎯 Learning Objectives

By the end of this study period, you should be able to:
- ✅ Understand transformer architectures (BERT, GPT, Llama)
- ✅ Fine-tune models for custom intent classification
- ✅ Implement RAG (Retrieval-Augmented Generation) systems
- ✅ Optimize models for edge deployment (quantization, TensorRT)
- ✅ Build production-ready APIs with FastAPI
- ✅ Understand your existing chapo-bot architecture deeply

---

## 📚 Week 1: Foundational Concepts

### Day 1-2: Review Your Chapo-Bot Architecture

**Study your existing code**:
```bash
cd /Users/user/chapo-bot-backend/backend/

# Study these files in order:
1. intent/intent_router.py          # Understand intent routing
2. services/nlp.py                   # Wit.ai integration
3. chapo_engines/core_conversation_engine.py  # Dialogue management
4. chapo_engines/emotion_detector_engine.py   # Emotion detection
5. multi_turn_manager.py             # Multi-turn conversations
```

**Key Concepts to Extract**:
- How intent normalization works
- Session memory with TTL
- Entity extraction patterns
- Multi-engine architecture
- Fallback mechanisms (spaCy when Wit.ai fails)

**Action**: Create a diagram mapping how a user query flows through chapo-bot

### Day 3-4: Transformer Fundamentals

**Resources**:

1. **The Illustrated Transformer** (30 min read)
   - URL: http://jalammar.github.io/illustrated-transformer/
   - Focus: Self-attention mechanism, encoder-decoder architecture
   - Why: Foundation for all modern NLP (BERT, GPT, Llama)

2. **The Illustrated BERT** (20 min read)
   - URL: http://jalammar.github.io/illustrated-bert/
   - Focus: How BERT works, fine-tuning process
   - Why: We'll use BERT for intent classification

3. **Hugging Face Transformers Course** (2-3 hours)
   - URL: https://huggingface.co/learn/nlp-course/chapter1/1
   - Focus: Chapters 1-3 (Transformer models, Using Transformers, Fine-tuning)
   - Hands-on: Run the code examples in Colab
   - Why: Our primary framework for NLP

**Practical Exercise**:
```python
# Install and try:
pip install transformers torch

# Run this script:
from transformers import pipeline

# Try sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love building robots!")
print(result)

# Try zero-shot classification (similar to intent classification)
classifier = pipeline("zero-shot-classification")
result = classifier(
    "Bring me the red cup",
    candidate_labels=["fetch_object", "navigate", "greeting"]
)
print(result)
```

### Day 5-7: PyTorch Essentials

**Resources**:

1. **PyTorch 60 Minute Blitz** (2-3 hours)
   - URL: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
   - Focus: Tensors, Autograd, Neural Networks, Training
   - Why: Our primary deep learning framework

2. **PyTorch for NLP** (1-2 hours)
   - URL: https://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html
   - Focus: Word embeddings, text classification
   - Why: Directly applicable to intent classification

**Practical Exercise**:
```python
# Build a simple intent classifier from scratch:
import torch
import torch.nn as nn

class SimpleIntentClassifier(nn.Module):
    def __init__(self, vocab_size, num_intents):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, num_intents)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :])

# Train this on a simple dataset
```

---

## 📚 Week 2: Advanced NLP & RAG

### Day 8-10: Fine-Tuning Transformers

**Resources**:

1. **Fine-tuning a pretrained model** (Hugging Face)
   - URL: https://huggingface.co/docs/transformers/training
   - Focus: Dataset preparation, Trainer API, evaluation
   - Why: We need to fine-tune BERT for robotics intents

2. **Text Classification Tutorial** (1-2 hours)
   - URL: https://huggingface.co/docs/transformers/tasks/sequence_classification
   - Hands-on: Fine-tune BERT on IMDb or similar
   - Why: Same process as intent classification

**Practical Exercise**:
```python
# Fine-tune BERT on a small dataset:
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=10)

# Prepare your data (start with public dataset, then use your own)
dataset = load_dataset("emotion")

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Train
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)
trainer.train()

# This is the same process we'll use for intent classification!
```

### Day 11-12: RAG (Retrieval-Augmented Generation)

**Resources**:

1. **LangChain RAG Tutorial** (2-3 hours)
   - URL: https://python.langchain.com/docs/tutorials/rag/
   - Focus: Vector stores, embeddings, retrieval, LLM integration
   - Why: Core component of our NLP system

2. **Building RAG with FAISS** (1 hour)
   - URL: https://www.pinecone.io/learn/retrieval-augmented-generation/
   - Focus: Vector similarity search, embedding generation
   - Why: We'll use FAISS for vector storage

**Practical Exercise**:
```python
# Simple RAG implementation:
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create knowledge base
documents = [
    "The kitchen is located on the first floor",
    "Cups are stored in the upper cabinet",
    "The robot can navigate autonomously"
]

# Create embeddings
embeddings = model.encode(documents)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))

# Query
query = "Where are cups located?"
query_embedding = model.encode([query])
distances, indices = index.search(query_embedding.astype('float32'), k=2)

print("Top results:", [documents[i] for i in indices[0]])
```

### Day 13-14: Model Optimization for Edge

**Resources**:

1. **ONNX Conversion** (1-2 hours)
   - URL: https://onnx.ai/get-started.html
   - Focus: Converting PyTorch → ONNX
   - Why: First step for TensorRT optimization

2. **Model Quantization** (1-2 hours)
   - URL: https://pytorch.org/docs/stable/quantization.html
   - Focus: INT8 quantization, QAT (Quantization-Aware Training)
   - Why: Reduce model size for Jetson deployment

**Practical Exercise**:
```python
# Convert your model to ONNX:
import torch

model = YourModel()  # Load your trained model
dummy_input = torch.randn(1, 128)  # Adjust to your input shape

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)

# Load and test ONNX model:
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
output = session.run(None, {"input": dummy_input.numpy()})
```

---

## 📚 Week 3: Production APIs & MLOps

### Day 15-17: FastAPI for ML Services

**Resources**:

1. **FastAPI Official Tutorial** (3-4 hours)
   - URL: https://fastapi.tiangolo.com/tutorial/
   - Focus: Path operations, request/response models, async
   - Why: Our primary API framework

2. **Deploying ML Models with FastAPI** (2 hours)
   - URL: https://testdriven.io/blog/fastapi-machine-learning/
   - Focus: Model serving, async inference, error handling
   - Why: Exact pattern we'll use

**Practical Exercise**:
```python
# Create ML API:
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    intent: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Load your model and predict
    intent = "fetch_object"  # Replace with actual prediction
    confidence = 0.95
    return PredictionResponse(intent=intent, confidence=confidence)

# Run: uvicorn main:app --reload
```

### Day 18-19: Docker for ML

**Resources**:

1. **Docker for Data Science** (2-3 hours)
   - URL: https://docker-curriculum.com/
   - Focus: Basics, Dockerfiles, multi-stage builds
   - Why: All our services will be containerized

**Practical Exercise**:
```dockerfile
# Create a simple ML service Dockerfile:
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Day 20-21: MLflow Basics

**Resources**:

1. **MLflow Quickstart** (1-2 hours)
   - URL: https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html
   - Focus: Tracking experiments, logging parameters, saving models
   - Why: Our experiment tracking tool

**Practical Exercise**:
```python
import mlflow

# Track your training:
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    
    # Train model...
    accuracy = 0.96
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Save model
    mlflow.pytorch.log_model(model, "intent_classifier")
```

---

## 🎓 Optional (But Recommended)

### For Robotics Understanding (if time permits):

1. **ROS2 Basics** (3-4 hours)
   - URL: https://docs.ros.org/en/humble/Tutorials.html
   - Focus: Concepts (nodes, topics, services), not deep implementation yet
   - Why: Understanding how robot middleware works

2. **Understanding SLAM** (1 hour)
   - Video: "SLAM Explained" on YouTube
   - Focus: Localization, mapping, loop closure concepts
   - Why: Context for perception module later

### For Computer Vision (upcoming Phase 2):

**Bookmark these for later**:
1. YOLOv8 Documentation: https://docs.ultralytics.com/
2. OpenCV Tutorials: https://docs.opencv.org/4.x/d9/df8/tutorial_root.html
3. PyTorch Vision: https://pytorch.org/vision/stable/index.html

---

## 🛠️ Hands-On Practice (MOST IMPORTANT)

### Project 1: Build a Simple Intent Classifier (Weekend Project)

**Goal**: Understand the full pipeline before scaling up

```python
# Steps:
1. Collect 100 sample robot commands (or use public dataset)
2. Label them with intents (navigate, fetch, greeting, etc.)
3. Fine-tune BERT on your data
4. Convert to ONNX
5. Build FastAPI endpoint
6. Test with Docker
7. Track with MLflow

# This mini-project covers 80% of what we'll build!
```

### Project 2: RAG Prototype

**Goal**: Understand retrieval-augmented generation

```python
# Steps:
1. Create a small knowledge base (robot facts, object locations)
2. Generate embeddings with sentence-transformers
3. Store in FAISS
4. Build query system
5. Integrate with a simple LLM (GPT-2 or small Llama)
6. Compare responses with vs. without RAG
```

---

## 📖 Key Papers to Read (From Our Bibliography)

**Priority 1 (Must Read)**:
1. **BERT** (Devlin et al., 2019) - 30 min
   - Focus: How BERT is pretrained and fine-tuned
   - Why: Our intent classifier will be BERT-based

2. **SayCan** (Ahn et al., 2022) - 45 min
   - URL: https://arxiv.org/abs/2204.01691
   - Focus: How LLMs ground in robot affordances
   - Why: Informs our LLM integration strategy

**Priority 2 (Recommended)**:
3. **RT-2** (Brohan et al., 2023) - 45 min
   - Focus: Vision-language-action models
   - Why: Multimodal fusion inspiration

4. **RAG** (Lewis et al., 2020) - 30 min
   - Focus: Retrieval-augmented generation
   - Why: We're implementing this

**Priority 3 (Context)**:
5. Browse our `docs/research_paper/BIBLIOGRAPHY.md` for relevant papers

---

## 💻 Setup Your Development Environment

### Essential Tools

```bash
# 1. Create virtual environment
cd /Users/user/humaniod_robot_assitant
python3.10 -m venv venv
source venv/bin/activate

# 2. Install core dependencies
pip install torch transformers sentence-transformers
pip install fastapi uvicorn pydantic
pip install onnx onnxruntime
pip install mlflow
pip install faiss-cpu  # or faiss-gpu if you have GPU
pip install spacy
python -m spacy download en_core_web_sm

# 3. Test installations
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "from sentence_transformers import SentenceTransformer; print('✓ Sentence Transformers')"
```

### Recommended IDE Setup

**VS Code Extensions**:
- Python
- Pylance
- Jupyter
- Docker
- YAML

**PyCharm Plugins** (if using PyCharm):
- Python
- Docker
- Markdown

---

## 📝 Study Checklist

### Week 1: Foundations
- [ ] Review chapo-bot architecture (4 hours)
- [ ] Read "Illustrated Transformer" (30 min)
- [ ] Read "Illustrated BERT" (20 min)
- [ ] Complete Hugging Face Course Chapters 1-3 (3 hours)
- [ ] PyTorch 60 Minute Blitz (3 hours)
- [ ] PyTorch NLP tutorial (2 hours)
- [ ] Build simple classifier from scratch (4 hours)

**Total**: ~16 hours (2 hours/day for 1 week)

### Week 2: Advanced Topics
- [ ] Fine-tuning tutorial (2 hours)
- [ ] Text classification hands-on (3 hours)
- [ ] LangChain RAG tutorial (3 hours)
- [ ] Build RAG prototype (4 hours)
- [ ] ONNX conversion tutorial (2 hours)
- [ ] Quantization basics (2 hours)
- [ ] Convert model to ONNX (2 hours)

**Total**: ~18 hours (2-3 hours/day for 1 week)

### Week 3: Production Skills
- [ ] FastAPI tutorial (4 hours)
- [ ] ML model serving (2 hours)
- [ ] Build FastAPI ML endpoint (3 hours)
- [ ] Docker tutorial (3 hours)
- [ ] Containerize ML service (2 hours)
- [ ] MLflow quickstart (2 hours)
- [ ] Track your experiments (2 hours)

**Total**: ~18 hours (2-3 hours/day for 1 week)

---

## 🎯 Conceptual Understanding Priorities

### Must Understand (Critical):

1. **How BERT works**
   - Masked language modeling
   - Fine-tuning for classification
   - Tokenization process

2. **Intent Classification Pipeline**
   - Input text → Tokenization → BERT → Classification head → Intent
   - How to prepare training data
   - Evaluation metrics (accuracy, F1)

3. **RAG Architecture**
   - Query → Embedding → Vector search → Context + Query → LLM → Response
   - Why it prevents hallucination
   - When to use vs. direct LLM

4. **ONNX & TensorRT**
   - Why we convert (speed, optimization)
   - What gets optimized (layer fusion, quantization)
   - Trade-offs (speed vs. accuracy)

5. **Microservices for ML**
   - Why separate services (NLP, Vision, etc.)
   - gRPC vs REST
   - Service orchestration

### Good to Understand (Important):

6. Model quantization (INT8, 4-bit)
7. Attention mechanisms in transformers
8. Embeddings and vector similarity
9. Docker containerization
10. CI/CD for ML

### Nice to Have (Bonus):

11. ROS2 basics
12. Kubernetes fundamentals
13. Distributed training
14. Advanced prompt engineering

---

## 🧪 Practical Mini-Projects Before We Code

### Mini-Project 1: Chapo-Bot Extension (2-4 hours)

**Goal**: Get comfortable with your existing architecture

```python
# Add a new intent to chapo-bot:
# 1. Add "play_music" intent to INTENT_NORMALIZATION_MAP
# 2. Create a simple MusicEngine (similar to WeatherEngine)
# 3. Route it in intent_router.py
# 4. Test end-to-end

# This familiarizes you with the architecture we're evolving
```

### Mini-Project 2: BERT Intent Classifier (4-6 hours)

**Goal**: Build what we'll actually use

```python
# 1. Create dataset: 100-200 robot commands with labels
# 2. Fine-tune bert-base-uncased
# 3. Evaluate on test set
# 4. Save model
# 5. Load and predict on new examples

# This is 70% of our intent classifier!
```

### Mini-Project 3: Simple RAG System (3-4 hours)

**Goal**: Understand retrieval before scaling

```python
# 1. Create knowledge base (10-20 facts about robot capabilities)
# 2. Generate embeddings
# 3. Store in FAISS
# 4. Query with user questions
# 5. Print top-3 relevant facts

# No LLM needed yet - just understand retrieval
```

---

## 📊 Progress Tracking

Create a study log:

```markdown
# Study Log

## Week 1
- [x] Day 1: Reviewed intent_router.py, understood routing
- [x] Day 2: Studied emotion_detector_engine.py
- [ ] Day 3: Read Illustrated Transformer
- [ ] Day 4: Completed HuggingFace Chapter 1
...
```

---

## 🎯 What You Should Be Able to Do After 3 Weeks

### Coding Skills:
- ✅ Fine-tune BERT for any classification task
- ✅ Build RAG system with FAISS
- ✅ Convert PyTorch → ONNX
- ✅ Create FastAPI endpoints
- ✅ Containerize ML services
- ✅ Track experiments with MLflow

### Conceptual Understanding:
- ✅ How transformers work (attention, self-attention)
- ✅ Why RAG prevents hallucination
- ✅ How to optimize models for edge
- ✅ Microservices architecture for AI
- ✅ Production ML best practices

### Confidence Level:
- ✅ Comfortable with Hugging Face ecosystem
- ✅ Can debug transformer training issues
- ✅ Understand trade-offs in model optimization
- ✅ Ready to build production NLP systems

---

## 🚀 After 3 Weeks, We'll Build:

**Week 4 Onward (Phase 1 Implementation)**:

1. **Entity Extractor** (`src/nlp/entities/extractor.py`)
2. **Dialogue Manager** (`src/nlp/dialogue/manager.py`)
3. **Emotion Detector** (`src/nlp/emotion/detector.py`)
4. **RAG System** (`src/nlp/rag/`)
5. **LLM Integration** (`src/nlp/llm/`)
6. **NLP Service API** (`services/nlp_service/`)

You'll be ready because you'll have:
- ✅ Built similar components during study
- ✅ Understood the underlying concepts
- ✅ Practiced the tools and workflows
- ✅ Created mini-versions of each component

---

## 📌 Key Resources Summary

### Primary Learning Platforms:
1. **Hugging Face**: https://huggingface.co/learn
2. **PyTorch Tutorials**: https://pytorch.org/tutorials/
3. **FastAPI Docs**: https://fastapi.tiangolo.com/
4. **LangChain Docs**: https://python.langchain.com/

### Your Local Resources:
1. **Chapo-Bot**: `/Users/user/chapo-bot-backend/`
2. **This Project**: `/Users/user/humaniod_robot_assitant/`
3. **Our Docs**: `docs/` folder
4. **Research Papers**: `docs/research_paper/BIBLIOGRAPHY.md`

---

## 🎓 Study Schedule Template

### Full-Time Study (4-6 hours/day):
- **Week 1**: Foundations (transformers, PyTorch)
- **Week 2**: Advanced NLP (fine-tuning, RAG)
- **Week 3**: Production (FastAPI, Docker, MLOps)

### Part-Time Study (2 hours/day):
- **Weeks 1-2**: Foundations
- **Weeks 3-4**: Advanced NLP
- **Weeks 5-6**: Production skills

### Weekend-Only (8 hours/weekend):
- **Weekends 1-2**: Foundations
- **Weekends 3-4**: Advanced NLP
- **Weekends 5-6**: Production skills

---

## ✅ Ready to Start Building When You Can:

### Demonstrate:
- [ ] Explain how BERT fine-tuning works
- [ ] Build a simple intent classifier (100 lines)
- [ ] Create a basic RAG system
- [ ] Deploy a model with FastAPI
- [ ] Containerize with Docker
- [ ] Convert PyTorch → ONNX

### Have:
- [ ] Development environment set up
- [ ] Chapo-bot architecture understood
- [ ] Completed 2-3 mini-projects
- [ ] Read 3-4 key papers

---

## 🚀 First Coding Session (After Study)

When ready, we'll start with:

```python
# Session 1: Entity Extractor
# File: src/nlp/entities/extractor.py
# Goal: Build NER system for robotics entities

# You'll know:
# - How to fine-tune BERT for token classification
# - How to handle entity spans
# - How to integrate spaCy fallback
# - How to optimize for edge
```

---

## 📞 Questions During Study?

**Save them and we'll address when you start coding**. Or:
- Check our docs: `docs/` folder
- Review similar code in chapo-bot
- Google with specific error messages
- Stack Overflow for common issues

---

**Start with Week 1, Day 1-2: Review your chapo-bot code! Understanding what you've already built is the fastest path to success.** 🚀

**Take your time - solid foundations lead to faster development later.**

---

**Study Guide Version**: 1.0  
**Author**: Victor Ibhafidon  
**Organization**: Xtainless Technologies  
**Next Update**: After you complete Week 1

