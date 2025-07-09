# DPO-Driver: Web Automation Agent via Direct Preference Optimization

[![Status](https://img.shields.io/badge/status-Phase%200%20Complete-brightgreen)](./DEVELOPMENT_GUIDE.md)
[![Model](https://img.shields.io/badge/model-Qwen2--7B-blue)](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
[![Environment](https://img.shields.io/badge/environment-MiniWoB++-orange)]()
[![Framework](https://img.shields.io/badge/framework-Poetry%20%7C%20PyTorch%20%7C%20TRL-violet)](./pyproject.toml)

**DPO-Driver is a research project to build a highly capable web automation agent by fine-tuning a Large Language Model (Qwen2-7B) using Direct Preference Optimization (DPO).**

This project strictly follows the technical specifications and development plan outlined in the [**DEVELOPMENT_GUIDE.md**](./DEVELOPMENT_GUIDE.md).

---

### Core Principles

1.  **Risk-First:** Always prioritize solving the highest-risk technical challenges first.
2.  **Keep It Simple (KISS):** V1.0 focuses on validating the core methodology, not building a complex system.
3.  **Iterative Loop:** Each development phase must form a testable, verifiable closed loop. Make it work, then make it right.

### System Architecture

The system follows a "brain-body" separation design:

-   **Agent (`src/agent/`):** The "brain." It loads the Qwen2 model and encapsulates SFT and DPO training logic.
-   **Environment (`src/environment/`):** The "body." It handles all interactions with the MiniWoB++ simulation environment, translating abstract actions into concrete Selenium executions.
-   **Scripts (`scripts/`):** The execution layer, orchestrating training, evaluation, and data collection processes.

### Project Status

**Phase 0: Risk Mitigation & Environment Validation - ✅ COMPLETE**

-   [x] Project structure and dependencies initialized.
-   [x] `EnvironmentInterface` for MiniWoB++ is functional.
-   [x] **DPO Pressure Test Passed:** Successfully ran a DPO training step with the Qwen2-7B model on an 8GB VRAM GPU (RTX 4060) **without quantization**. The core hardware risk is fully mitigated.

**Next Up: Phase 1 - SFT Baseline Loop**

### Getting Started

1.  **Configure Poetry:**
    If you want the virtual environment to be created in the project directory, run:
    ```bash
    poetry config virtualenvs.in-project true
    ```

2.  **Install Dependencies:**
    This project uses [Poetry](https://python-poetry.org/) for dependency management.
    ```bash
    poetry install
    ```
    *Note: The installation will set up a local `.venv` folder for the environment.*

3.  **Run Tests:**
    To verify the environment setup:
    ```bash
    # Verify CUDA and PyTorch
    poetry run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

    # Run the environment unit test
    poetry run pytest
    ```

### Usage

The `scripts/` directory contains the main execution scripts for different project phases:

-   `00_dpo_pressure_test.py`: Validates hardware and environment setup (already passed).
-   `01_sft_training.py`: (Upcoming) Script to run Supervised Fine-Tuning.
-   `02_collect_preferences.py`: (Upcoming) Script to collect data for DPO.
-   `03_dpo_training.py`: (Upcoming) Script to run DPO training.
-   `04_evaluate_agent.py`: (Upcoming) Script to evaluate the final agent. 