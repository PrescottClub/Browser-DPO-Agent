#!/usr/bin/env python3
"""
Project environment verification script
Verifies that all dependencies and core components are correctly installed
"""

import os
import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_dependencies():
    """Check core dependencies"""
    print("[CHECKING] Checking dependencies...")

    try:
        import torch

        print(f"[SUCCESS] PyTorch: {torch.__version__}")
    except ImportError:
        print("[ERROR] PyTorch not installed")
        return False

    try:
        import transformers

        print(f"[SUCCESS] Transformers: {transformers.__version__}")
    except ImportError:
        print("[ERROR] Transformers not installed")
        return False

    try:
        import trl

        print(f"[SUCCESS] TRL: {trl.__version__}")
    except ImportError:
        print("[ERROR] TRL not installed")
        return False

    try:
        import peft

        print(f"[SUCCESS] PEFT: {peft.__version__}")
    except ImportError:
        print("[ERROR] PEFT not installed")
        return False

    try:
        import selenium

        print(f"[SUCCESS] Selenium: {selenium.__version__}")
    except ImportError:
        print("[ERROR] Selenium not installed")
        return False

    return True


def check_project_structure():
    """Check project structure"""
    print("\n[CHECKING] Checking project structure...")

    required_dirs = [
        "src/agent",
        "src/environment",
        "src/miniwob",
        "scripts",
        "data",
        "models",
        "tests",
    ]

    required_files = [
        "src/agent/__init__.py",
        "src/agent/model.py",
        "src/environment/__init__.py",
        "src/environment/interface.py",
        "scripts/01_sft_training.py",
        "scripts/02_collect_preferences.py",
        "scripts/03_dpo_training.py",
        "scripts/04_evaluate_agent.py",
        "pyproject.toml",
        "README.md",
    ]

    all_good = True

    for dir_path in required_dirs:
        if (project_root / dir_path).exists():
            print(f"[SUCCESS] {dir_path}/")
        else:
            print(f"[ERROR] {dir_path}/ does not exist")
            all_good = False

    for file_path in required_files:
        if (project_root / file_path).exists():
            print(f"[SUCCESS] {file_path}")
        else:
            print(f"[ERROR] {file_path} does not exist")
            all_good = False

    return all_good


def check_imports():
    """Check core module imports"""
    print("\n[CHECKING] Testing module imports...")

    try:
        # Check AgentModel
        sys.path.insert(0, str(project_root / "src"))
        from agent.model import AgentModel

        print("[SUCCESS] AgentModel imported successfully")
    except ImportError as e:
        print(f"[ERROR] AgentModel import failed: {e}")
        return False

    try:
        # Check EnvironmentInterface
        from environment.interface import EnvironmentInterface

        print("[SUCCESS] EnvironmentInterface imported successfully")
    except ImportError as e:
        print(f"[ERROR] EnvironmentInterface import failed: {e}")
        return False

    return True


def main():
    """Main verification process"""
    print("[RUNNING] DPO-Driver Environment Verification")
    print("=" * 50)

    # Check dependencies
    deps_ok = check_dependencies()

    # Check project structure
    structure_ok = check_project_structure()

    # Check module imports
    imports_ok = check_imports()

    print("\n" + "=" * 50)

    if deps_ok and structure_ok and imports_ok:
        print("[SUCCESS] All checks passed! Project environment is configured correctly.")
        print("\n[NEXT STEPS] Next steps:")
        print("1. Run SFT training: poetry run python scripts/01_sft_training.py")
        print("2. Collect preference data: poetry run python scripts/02_collect_preferences.py")
        print("3. Run DPO training: poetry run python scripts/03_dpo_training.py")
        print("4. Evaluate performance: poetry run python scripts/04_evaluate_agent.py")
        return True
    else:
        print("[ERROR] Environment setup has issues, please check the error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
