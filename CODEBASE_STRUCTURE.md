# Codebase Structure & Organization

**Author:** Victor Ibhafidon  
**Date:** October 2025

## Overview

This document explains the complete codebase structure, where to find each script, and how everything connects together in the ML/robotics pipeline.

---

## Directory Structure

```
humaniod_robot_assitant/
│
├── src/                          # Core source code (production)
│   ├── nlp/                     # Natural Language Processing module
│   │   ├── intent/              # Intent classification
│   │   ├── entities/            # Entity extraction (NER)
│   │   ├── emotion/             # Emotion detection
│   │   ├── dialogue/            # Dialogue management
│   │   ├── rag/                 # RAG system
│   │   ├── llm/                 # LLM integration
│   │   ├── asr/                 # Speech recognition
│   │   ├── tts/                 # Text-to-speech
│   │   └── nlp_service.py       # Unified NLP service
│   │
│   └── vision/                  # Computer Vision module
│       ├── object_detection/    # Object detector
│       ├── segmentation/        # Segmentation
│       ├── depth/               # Depth estimation
│       ├── pose/                # Pose estimation
│       ├── face/                # Face recognition
│       ├── scene/               # Scene understanding
│       └── vision_service.py    # Unified Vision service
│
├── tests/                        # All testing code
│   ├── unit/                    # Unit tests (per component)
│   │   ├── test_nlp_components.py
│   │   └── test_cv_components.py
│   │
│   ├── integration/             # Integration & end-to-end tests
│   │   ├── test_nlp_comprehensive.py    # Full NLP pipeline test
│   │   └── test_vision_comprehensive.py # Full CV pipeline test
│   │
│   └── simulation/              # Simulation-based testing
│       ├── sim_environment.py   # Generates synthetic scenarios
│       └── README.md            # Simulation testing guide
│
├── training/                     # Model training & fine-tuning
│   ├── finetune_yolov11.py     # YOLOv11 fine-tuning pipeline
│   ├── finetune_sam.py         # SAM fine-tuning (future)
│   ├── configs/                 # Training configurations
│   │   └── robot_objects.yaml  # Dataset config template
│   └── runs/                    # Training runs & checkpoints
│
├── evaluation/                   # Model evaluation & metrics
│   ├── metrics.py              # CV metrics (mAP, IoU, etc.)
│   ├── nlp_metrics.py          # NLP metrics (F1, accuracy)
│   └── benchmark_results/       # Evaluation results
│
├── data_collection/             # Data collection strategies
│   ├── collection_strategy.py  # 10 robust scenarios
│   ├── annotation_guidelines.md
│   └── README.md
│
├── benchmarks/                  # Performance benchmarking
│   ├── latency_benchmark.py    # Measure inference speed
│   ├── accuracy_benchmark.py   # Measure model accuracy
│   └── tier_comparison.py      # Compare Tier 1/2/3
│
├── configs/                     # Configuration files
│   └── base/
│       └── system_config.yaml  # Master system configuration
│
├── scripts/                     # Utility scripts only
│   └── setup/
│       └── setup_nlp_module.sh # Environment setup
│
├── docs/                        # Documentation
│   ├── architecture/
│   ├── research_paper/
│   └── PROJECT_ROADMAP.md
│
├── deployment/                  # Deployment configs
│   └── docker/
│       └── docker-compose.yml
│
├── data/                        # Data storage (gitignored)
│   ├── collected/              # Real-world collected data
│   ├── synthetic/              # Simulation-generated data
│   └── processed/              # Preprocessed datasets
│
└── models/                      # Trained models (gitignored)
    ├── yolov11_robot_v1.pt
    ├── yolov11_robot_v2.pt
    └── model_registry.json
```

---

## Where to Find Scripts

### Testing Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| NLP Integration Test | `tests/integration/test_nlp_comprehensive.py` | Tests all 8 NLP components end-to-end |
| CV Integration Test | `tests/integration/test_vision_comprehensive.py` | Tests all 6 CV components end-to-end |
| Simulation Environment | `tests/simulation/sim_environment.py` | Generates synthetic test scenarios |

**When to use:**
- Run integration tests before committing code
- Use simulation for rapid iteration without hardware
- Run full test suite before deployment

### Training Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| YOLOv11 Fine-Tuning | `training/finetune_yolov11.py` | Fine-tune object detection on robot data |
| Training Config Template | `training/configs/robot_objects.yaml` | Dataset configuration template |

**When to use:**
- After collecting 1000+ robot-specific images
- When pre-trained models underperform
- For domain adaptation to robot scenarios

### Evaluation Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| CV Metrics | `evaluation/metrics.py` | Calculate mAP, IoU, precision, recall |
| Benchmarking | `benchmarks/latency_benchmark.py` | Measure inference speed |
| Tier Comparison | `benchmarks/tier_comparison.py` | Compare multi-tier performance |

**When to use:**
- After training to measure improvement
- For A/B testing model versions
- To validate multi-tier fallback

### Data Collection Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| Collection Strategy | `data_collection/collection_strategy.py` | 10 comprehensive scenarios |

**When to use:**
- Planning data collection campaigns
- Generating collection schedules
- Ensuring scenario coverage

---

## Pipeline Flow

### 1. Development Flow

```
Code Changes → Unit Tests → Integration Tests → Commit
     ↓              ↓              ↓
  src/nlp/    tests/unit/   tests/integration/
  src/vision/
```

### 2. Training Flow

```
Data Collection → Annotation → Training → Evaluation → Deployment
      ↓              ↓            ↓           ↓           ↓
data_collection/  data/     training/  evaluation/   models/
                collected/   runs/     results/
```

### 3. Testing Flow

```
Synthetic Testing → Real Testing → Production Testing
       ↓                ↓                 ↓
tests/simulation/  tests/integration/  benchmarks/
```

---

## How Scripts Connect

### Example: Complete Fine-Tuning Workflow

1. **Plan Data Collection**
   ```bash
   python data_collection/collection_strategy.py
   # Outputs: 8-week collection plan
   ```

2. **Generate Synthetic Data** (Optional)
   ```bash
   python tests/simulation/sim_environment.py
   # Outputs: data/synthetic/scenarios/
   ```

3. **Collect Real Data**
   ```
   # Manual: Follow collection_strategy.py scenarios
   # Output: data/collected/images/
   ```

4. **Fine-Tune Model**
   ```bash
   python training/finetune_yolov11.py --data configs/robot_objects.yaml --epochs 100
   # Outputs: models/yolov11_robot_v1.pt
   ```

5. **Evaluate Model**
   ```bash
   python evaluation/metrics.py --model models/yolov11_robot_v1.pt
   # Outputs: evaluation/benchmark_results/
   ```

6. **Test Integration**
   ```bash
   python tests/integration/test_vision_comprehensive.py
   # Validates: All CV components with new model
   ```

7. **Deploy**
   ```bash
   # Update configs/base/system_config.yaml with new model path
   # System automatically uses Tier 1 with new model
   ```

---

## Quick Reference

### I want to...

**Test if my code changes broke anything:**
```bash
python tests/integration/test_nlp_comprehensive.py
python tests/integration/test_vision_comprehensive.py
```

**Generate test data without hardware:**
```bash
python tests/simulation/sim_environment.py
```

**Train a model on robot data:**
```bash
python training/finetune_yolov11.py --data robot_objects.yaml
```

**Measure model performance:**
```bash
python evaluation/metrics.py --model path/to/model.pt
```

**Compare Tier 1 vs Tier 2 vs Tier 3:**
```bash
python benchmarks/tier_comparison.py
```

**Plan data collection:**
```bash
python data_collection/collection_strategy.py
```

---

## File Naming Conventions

### Scripts
- `test_*.py` - Test scripts (in tests/)
- `finetune_*.py` - Training scripts (in training/)
- `*_strategy.py` - Strategy/planning scripts (in data_collection/)
- `*_metrics.py` - Evaluation scripts (in evaluation/)
- `*_benchmark.py` - Benchmarking scripts (in benchmarks/)

### Configs
- `*_config.yaml` - Configuration files (in configs/)
- `*.yaml` - Dataset configs (in training/configs/)

### Models
- `*_v1.pt, *_v2.pt` - Model checkpoints (in models/)
- `best.pt, last.pt` - Training checkpoints (in training/runs/)

---

## Best Practices

### Before Committing
1. Run integration tests
2. Check linting (if applicable)
3. Update relevant docs

### Before Training
1. Validate dataset (check image count, annotations)
2. Create dataset YAML config
3. Set up experiment tracking (MLflow)

### Before Deployment
1. Evaluate on held-out test set
2. Run benchmarks (latency, accuracy)
3. Test tier fallback behavior
4. Update system config

---

## Common Workflows

### Workflow 1: Add New CV Component

1. Implement in `src/vision/new_component/`
2. Add unit test in `tests/unit/test_new_component.py`
3. Update `tests/integration/test_vision_comprehensive.py`
4. Update `src/vision/vision_service.py` to integrate
5. Run full test suite
6. Update docs

### Workflow 2: Improve Model Accuracy

1. Collect more data (`data_collection/collection_strategy.py`)
2. Annotate data
3. Fine-tune (`training/finetune_yolov11.py`)
4. Evaluate (`evaluation/metrics.py`)
5. If improved, deploy to Tier 1
6. Monitor production metrics

### Workflow 3: Debug Failure Case

1. Reproduce with simulation (`tests/simulation/sim_environment.py`)
2. Add to test suite (`tests/integration/`)
3. Fix in source (`src/`)
4. Verify fix with tests
5. Collect real-world data of failure case
6. Retrain if needed

---

## Related Documents

- **TRAINING_AND_EVALUATION_STRATEGY.md** - ML best practices
- **CV_IMPLEMENTATION_SUMMARY.md** - Computer Vision details
- **NLP_IMPLEMENTATION_SUMMARY.md** - NLP details
- **PROJECT_ROADMAP.md** - Development timeline
- **START_HERE.md** - Getting started guide

---

**Last Updated:** October 2025  
**Maintained By:** Victor Ibhafidon

