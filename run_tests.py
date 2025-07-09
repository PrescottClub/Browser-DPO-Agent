#!/usr/bin/env python3
"""
DPO-Driver 测试运行器
"""

import argparse
import subprocess
import sys


def run_fast_tests():
    """运行快速测试（跳过慢速集成测试）"""
    print("🚀 运行快速测试套件...")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-m", "not slow", "-v", "--tb=short"]
    )

    return result.returncode == 0


def run_all_tests():
    """运行所有测试（包括慢速集成测试）"""
    print("🚀 运行完整测试套件（包括慢速测试）...")
    print("⚠️  警告：这可能需要几分钟时间")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    )

    return result.returncode == 0


def run_unit_tests():
    """只运行单元测试"""
    print("🧪 运行单元测试...")
    print("=" * 60)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_config.py",
            "tests/test_action_parser.py",
            "-v",
            "--tb=short",
        ]
    )

    return result.returncode == 0


def run_integration_tests():
    """只运行集成测试"""
    print("🔗 运行集成测试...")
    print("=" * 60)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_pipeline_integration.py",
            "-v",
            "--tb=short",
        ]
    )

    return result.returncode == 0


def run_smoke_test():
    """运行端到端冒烟测试"""
    print("💨 运行端到端冒烟测试...")
    print("⚠️  警告：这将实际运行训练脚本，可能需要很长时间")
    print("=" * 60)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_pipeline_integration.py::test_full_pipeline_smoke_test",
            "-v",
            "--tb=short",
            "-s",
        ]
    )

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="DPO-Driver 测试运行器")
    parser.add_argument("--fast", action="store_true", help="运行快速测试（默认）")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    parser.add_argument("--unit", action="store_true", help="只运行单元测试")
    parser.add_argument("--integration", action="store_true", help="只运行集成测试")
    parser.add_argument("--smoke", action="store_true", help="运行端到端冒烟测试")

    args = parser.parse_args()

    success = False

    if args.all:
        success = run_all_tests()
    elif args.unit:
        success = run_unit_tests()
    elif args.integration:
        success = run_integration_tests()
    elif args.smoke:
        success = run_smoke_test()
    else:  # 默认运行快速测试
        success = run_fast_tests()

    print("\n" + "=" * 60)
    if success:
        print("✅ 所有测试通过！")
        print("\n📋 测试覆盖范围:")
        print("• 配置系统验证")
        print("• 动作解析器测试")
        print("• 脚本导入和参数验证")
        print("• 环境初始化测试")

        if not args.unit:
            print("\n🚀 下一步:")
            print("1. 运行训练流程: poetry run python scripts/01_sft_training.py")
            print("2. 查看 MLflow UI: poetry run python start_mlflow_ui.py")
            print("3. 运行完整测试: python run_tests.py --all")
    else:
        print("❌ 部分测试失败")
        print("请检查上面的错误信息并修复问题")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
