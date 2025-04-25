# Enhanced Code Generation Model

This repository contains an enhanced implementation of a code generation model using RLHF (Reinforcement Learning from Human Feedback). The model is designed for high accuracy and reliable code generation.

## Key Features

- **Advanced Reward Model**: Dual-head architecture for both reward prediction and confidence estimation
- **Comprehensive Code Quality Metrics**:
  - Syntax correctness
  - Cyclomatic complexity
  - Code style (using pylint)
  - Combined quality score
- **Enhanced Training Process**:
  - Early stopping with patience
  - Learning rate scheduling with warmup
  - Gradient clipping for stability
  - Best model checkpointing
  - Wandb integration for monitoring

## Dataset

The model uses a combination of high-quality code datasets:
- CodeAlpaca-20k
- HumanEval
- MBPP (Mostly Basic Programming Problems)
- CodeParrot (clean subset)
- DeepMind Code Contests

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure Weights & Biases:
```bash
wandb login
```

3. Start training:
```bash
python main_training.py
```

## Model Architecture

The model uses a dual-head architecture:
- Base model: DeepSeek Coder 6.7B
- Reward head: Quality prediction
- Confidence head: Uncertainty estimation

## Training Process

1. **Dataset Preparation**:
   - Load and combine multiple code datasets
   - Calculate quality metrics for each code sample
   - Tokenize and prepare batches

2. **Training Loop**:
   - Cosine learning rate scheduling with warmup
   - Gradient clipping at 1.0
   - Early stopping with 5-epoch patience
   - Best model checkpointing based on validation loss

3. **Monitoring**:
   - Training and validation loss
   - Reward predictions
   - Confidence scores
   - Learning rate changes
   - Model gradients

## Performance Optimization

- Gradient checkpointing for memory efficiency
- Mixed precision training (float16)
- Efficient data loading with proper batching
- Automated early stopping to prevent overfitting

## Files

- `main.py`: Core model architecture and components
- `main_training.py`: Training loop and dataset handling
- `training_utils.py`: Training utilities and metrics calculation
- `requirements.txt`: Project dependencies
