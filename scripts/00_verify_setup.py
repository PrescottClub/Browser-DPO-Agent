#!/usr/bin/env python3
"""
项目环境验证脚本
验证所有依赖和核心组件是否正确安装
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def check_dependencies():
    """检查核心依赖"""
    print("🔍 检查依赖包...")

    try:
        import torch

        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch 未安装")
        return False

    try:
        import transformers

        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers 未安装")
        return False

    try:
        import trl

        print(f"✅ TRL: {trl.__version__}")
    except ImportError:
        print("❌ TRL 未安装")
        return False

    try:
        import peft

        print(f"✅ PEFT: {peft.__version__}")
    except ImportError:
        print("❌ PEFT 未安装")
        return False

    try:
        import selenium

        print(f"✅ Selenium: {selenium.__version__}")
    except ImportError:
        print("❌ Selenium 未安装")
        return False

    return True


def check_project_structure():
    """检查项目结构"""
    print("\n📁 检查项目结构...")

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
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ 不存在")
            all_good = False

    for file_path in required_files:
        if (project_root / file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} 不存在")
            all_good = False

    return all_good


def check_imports():
    """检查核心模块导入"""
    print("\n🔧 检查核心模块...")

    try:
        # 检查AgentModel
        sys.path.insert(0, str(project_root / "src"))
        from agent.model import AgentModel

        print("✅ AgentModel 导入成功")
    except ImportError as e:
        print(f"❌ AgentModel 导入失败: {e}")
        return False

    try:
        # 检查EnvironmentInterface
        from environment.interface import EnvironmentInterface

        print("✅ EnvironmentInterface 导入成功")
    except ImportError as e:
        print(f"❌ EnvironmentInterface 导入失败: {e}")
        return False

    return True


def main():
    """主验证流程"""
    print("🚀 DPO-Driver 环境验证")
    print("=" * 50)

    # 检查依赖
    deps_ok = check_dependencies()

    # 检查项目结构
    structure_ok = check_project_structure()

    # 检查模块导入
    imports_ok = check_imports()

    print("\n" + "=" * 50)

    if deps_ok and structure_ok and imports_ok:
        print("🎉 所有检查通过！项目环境配置正确。")
        print("\n📋 下一步:")
        print("1. 运行 SFT 训练: poetry run python scripts/01_sft_training.py")
        print("2. 收集偏好数据: poetry run python scripts/02_collect_preferences.py")
        print("3. 运行 DPO 训练: poetry run python scripts/03_dpo_training.py")
        print("4. 评估性能: poetry run python scripts/04_evaluate_agent.py")
        return True
    else:
        print("❌ 环境配置存在问题，请检查上述错误信息。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
