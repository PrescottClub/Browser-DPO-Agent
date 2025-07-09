# tests/test_pipeline_integration.py
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


# 将这个测试标记为"慢速"，以便在快速测试时可以跳过它
# 运行 pytest -m "not slow" 来跳过
@pytest.mark.slow
def test_full_pipeline_smoke_test(tmp_path: Path):
    """
    一个端到端的冒烟测试，使用微型参数跑通整个流程。
    这会花费几分钟时间。
    """
    print(f"--- 开始端到端集成冒烟测试，所有文件将生成在: {tmp_path} ---")

    # 1. 创建一个临时的、微型的配置文件
    test_config = {
        "model": {
            "base_model_name": "Qwen/Qwen2-1.5B-Instruct"
        },  # 使用最小的模型以加速
        "paths": {
            "sft_data": "data/sft_golden_samples.jsonl",  # 路径相对于项目根目录
            "preference_data": str(tmp_path / "prefs.jsonl"),
            "sft_adapter_path": str(tmp_path / "sft_adapter"),
            "dpo_adapter_path": str(tmp_path / "dpo_adapter"),
        },
        "training": {
            "sft": {
                "learning_rate": 2e-4,
                "max_steps": 1,
                "batch_size": 1,
                "grad_accumulation_steps": 1,
            },
            "dpo": {
                "learning_rate": 5e-6,
                "max_steps": 1,
                "batch_size": 1,
                "grad_accumulation_steps": 1,
                "beta": 0.1,
            },
        },
        "evaluation": {
            "num_episodes_per_task": 1,
            "tasks": ["miniwob/click-button-v1"],
        },
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)

    # 2. 手动创建假的偏好数据（因为收集偏好数据需要真实的环境交互，太慢）
    fake_pref_path = tmp_path / "prefs.jsonl"
    with open(fake_pref_path, "w") as f:
        f.write(
            '{"prompt": "Click the button", "chosen": "I will click the button. action: CLICK(selector=\\"#button\\")", "rejected": "I will do nothing."}\n'
        )

    # 3. 运行SFT训练
    print("运行SFT训练...")
    sft_result = subprocess.run(
        [
            sys.executable,
            "scripts/01_sft_training.py",
            "--config_path",
            str(config_path),
        ],
        capture_output=True,
        text=True,
        cwd=".",
    )

    print("SFT训练输出:")
    print(sft_result.stdout)
    if sft_result.stderr:
        print("SFT训练错误:")
        print(sft_result.stderr)

    assert (
        sft_result.returncode == 0
    ), f"SFT 脚本运行失败! 返回码: {sft_result.returncode}"
    assert (tmp_path / "sft_adapter").exists(), "SFT adapter 未能成功创建!"

    # 4. 运行DPO训练
    print("运行DPO训练...")
    dpo_result = subprocess.run(
        [
            sys.executable,
            "scripts/03_dpo_training.py",
            "--config_path",
            str(config_path),
        ],
        capture_output=True,
        text=True,
        cwd=".",
    )

    print("DPO训练输出:")
    print(dpo_result.stdout)
    if dpo_result.stderr:
        print("DPO训练错误:")
        print(dpo_result.stderr)

    assert (
        dpo_result.returncode == 0
    ), f"DPO 脚本运行失败! 返回码: {dpo_result.returncode}"
    assert (tmp_path / "dpo_adapter").exists(), "DPO adapter 未能成功创建!"

    print("--- 端到端集成冒烟测试成功！---")


@pytest.mark.slow
def test_config_parameter_passing():
    """
    测试配置参数传递功能，确保脚本能正确接收和使用自定义配置。
    """
    # 创建临时配置文件
    test_config = {
        "model": {"base_model_name": "test-model"},
        "paths": {
            "sft_data": "test.jsonl",
            "preference_data": "test_pref.jsonl",
            "sft_adapter_path": "./test_sft",
            "dpo_adapter_path": "./test_dpo",
        },
        "training": {
            "sft": {
                "learning_rate": 1e-4,
                "max_steps": 1,
                "batch_size": 1,
                "grad_accumulation_steps": 1,
            },
            "dpo": {
                "learning_rate": 1e-6,
                "max_steps": 1,
                "batch_size": 1,
                "grad_accumulation_steps": 1,
                "beta": 0.1,
            },
        },
        "evaluation": {"num_episodes_per_task": 1, "tasks": ["test-task"]},
    }

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(test_config, f)
        temp_config_path = f.name

    try:
        # 测试脚本能否正确解析配置参数
        # 这里我们只测试参数解析，不实际运行训练
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"""
import sys
sys.path.append('.')
import argparse
from src.utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="config.yaml")
args = parser.parse_args(["--config_path", "{temp_config_path}"])

config = load_config(args.config_path)
assert config.model.base_model_name == 'test-model'
assert config.training.sft.learning_rate == 1e-4
print("配置参数传递测试成功!")
""",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"配置参数测试失败: {result.stderr}"
        assert "配置参数传递测试成功!" in result.stdout

    finally:
        import os

        os.unlink(temp_config_path)


def test_scripts_import():
    """
    测试所有脚本能否正常导入（语法检查）。
    """
    scripts = [
        "scripts/01_sft_training.py",
        "scripts/02_collect_preferences.py",
        "scripts/03_dpo_training.py",
        "scripts/04_evaluate_agent.py",
    ]

    for script in scripts:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", script], capture_output=True, text=True
        )

        assert result.returncode == 0, f"脚本 {script} 编译失败: {result.stderr}"


def test_help_messages():
    """
    测试所有脚本的帮助信息是否正常显示。
    """
    scripts = [
        "scripts/01_sft_training.py",
        "scripts/02_collect_preferences.py",
        "scripts/03_dpo_training.py",
        "scripts/04_evaluate_agent.py",
    ]

    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"], capture_output=True, text=True
        )

        assert result.returncode == 0, f"脚本 {script} 帮助信息显示失败"
        assert (
            "--config_path" in result.stdout
        ), f"脚本 {script} 缺少 --config_path 参数"
