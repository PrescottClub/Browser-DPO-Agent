# Browser-DPO-Agent

A production-ready implementation of Direct Preference Optimization (DPO) for browser automation agents, featuring automated preference data collection and robust training pipelines.

## Overview

Browser-DPO-Agent implements a complete DPO training pipeline for web automation tasks using the MiniWoB++ environment. The system automatically collects preference data from environment feedback and uses it to improve agent performance through direct preference optimization.

## Key Features

- **Automated Preference Collection**: Generates preference pairs from environment success/failure signals
- **Modular Architecture**: Clean separation between SFT, DPO, and inference modules
- **Production Ready**: Comprehensive testing, error handling, and MLflow integration
- **Optimized Training**: Carefully tuned hyperparameters to prevent overfitting
- **Extensible Design**: Easy to adapt to new environments and tasks

## Architecture

```
src/
├── agent/           # Core agent modules
│   ├── base_model.py      # Foundation model class
│   ├── sft_module.py      # Supervised fine-tuning
│   ├── dpo_module.py      # Direct preference optimization
│   ├── inference_module.py # Text generation and inference
│   └── model.py           # High-level agent interface
├── environment/     # Environment interface
├── miniwob/        # MiniWoB++ integration
└── utils/          # Utilities and configuration
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install poetry
poetry install
```

### 2. Configuration

The system uses `config.yaml` for all training parameters:

```yaml
model:
  base_model_name: "Qwen/Qwen2-7B-Instruct"

training:
  sft:
    learning_rate: 2.0e-4
    max_steps: 100
    batch_size: 1
  dpo:
    learning_rate: 1.0e-6
    max_steps: 10
    batch_size: 1
    beta: 0.1
```

### 3. Training Pipeline

Run the complete training pipeline:

```bash
# Full pipeline execution
python scripts/run_pipeline.py

# Or run individual steps:
python scripts/01_sft_training.py     # Supervised fine-tuning
python scripts/02_collect_preferences.py  # Preference data collection
python scripts/03_dpo_training.py     # DPO training
python scripts/04_evaluate_agent.py   # Performance evaluation
```

## Training Process

1. **Supervised Fine-Tuning (SFT)**: Train on golden examples for basic competency
2. **Preference Collection**: Automatically generate preference pairs from environment feedback
3. **DPO Training**: Optimize agent behavior using collected preferences
4. **Evaluation**: Measure performance improvements on test tasks

## Configuration Details

### DPO Training Parameters

The DPO configuration has been optimized to prevent overfitting:

- **Learning Rate**: 1.0e-6 (reduced from 5.0e-6)
- **Max Steps**: 10 (reduced from 50)  
- **Early Stopping**: Enabled with evaluation-based best model selection
- **Data Size**: Expanded to 24 preference samples

### Data Format

Preference data uses the standard DPO format:
```json
{
  "prompt": "Click the button.",
  "chosen": "I need to click the button...\nAction: CLICK(selector=\"#button-1\")",
  "rejected": "I need to click something...\nAction: CLICK(selector=\"#wrong-element\")"
}
```

## Testing

Run the comprehensive test suite:

```bash
# All tests
python -m pytest tests/

# Specific test categories
python -m pytest tests/test_config.py
python -m pytest tests/test_modular_architecture.py
python -m pytest tests/test_environment.py
```

## MLflow Integration

The system includes comprehensive MLflow tracking:

```bash
# Start MLflow UI
python start_mlflow_ui.py
# Navigate to http://localhost:5000
```

Tracked metrics include:
- Training loss and learning curves
- System resources and performance
- Model checkpoints and artifacts
- Git state and reproducibility info

## Project Structure

```
Browser-DPO-Agent/
├── config.yaml              # Main configuration
├── data/                    # Training and preference data
├── models/                  # Saved model adapters  
├── scripts/                 # Training pipeline scripts
├── src/                     # Source code
├── tests/                   # Test suite
└── README.md               # This file
```

## Troubleshooting

### Common Issues

1. **Configuration Loading Error**: Ensure config.yaml uses UTF-8 encoding without special characters
2. **DPO Parameter Conflicts**: The system automatically adjusts eval_steps to be compatible with save_steps
3. **Memory Issues**: Reduce batch_size or use gradient accumulation for larger models

### Performance Optimization

- Use smaller learning rates for DPO (1e-6 or lower)
- Limit training steps when working with small datasets
- Enable early stopping to prevent overfitting
- Monitor evaluation metrics during training

## Contributing

1. Follow the existing code structure and patterns
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass before submitting

## License

MIT License - see LICENSE file for details.

---

**Note**: This implementation focuses on production readiness and includes extensive error handling, testing, and monitoring capabilities for real-world deployment scenarios.