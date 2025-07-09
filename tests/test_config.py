# tests/test_config.py
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import get_config, load_config


def test_load_default_config():
    """测试能否成功加载默认的config.yaml文件。"""
    config = get_config()
    assert config is not None
    assert config.model.base_model_name == "Qwen/Qwen2-7B-Instruct"


def test_get_nested_value():
    """测试能否正确获取嵌套的配置值。"""
    config = get_config()
    learning_rate = config.training.dpo.learning_rate
    assert learning_rate == 5.0e-6


def test_config_validation():
    """测试配置验证功能。"""
    config = get_config()

    # 测试必需字段存在
    assert hasattr(config, "model")
    assert hasattr(config, "paths")
    assert hasattr(config, "training")
    assert hasattr(config, "evaluation")

    # 测试训练配置
    assert hasattr(config.training, "sft")
    assert hasattr(config.training, "dpo")

    # 测试数值范围
    assert config.training.sft.learning_rate > 0
    assert config.training.dpo.learning_rate > 0
    assert config.training.sft.max_steps > 0
    assert config.training.dpo.max_steps > 0
    assert config.training.dpo.beta > 0
    assert config.evaluation.num_episodes_per_task > 0


def test_config_file_not_found():
    """测试当配置文件不存在时是否会抛出FileNotFoundError。"""
    # 重置全局配置状态以强制重新加载
    from src.utils import config

    original_config = config._config
    config._config = None

    try:
        with pytest.raises(FileNotFoundError):
            load_config("non_existent_config.yaml")
    finally:
        # 恢复原始配置状态
        config._config = original_config


def test_load_custom_config():
    """测试加载自定义配置文件。"""
    # 创建临时配置文件
    test_config_data = {
        "model": {"base_model_name": "test-model"},
        "paths": {
            "sft_data": "test_sft.jsonl",
            "preference_data": "test_pref.jsonl",
            "sft_adapter_path": "./test_sft",
            "dpo_adapter_path": "./test_dpo",
        },
        "training": {
            "sft": {
                "learning_rate": 1e-4,
                "max_steps": 50,
                "batch_size": 2,
                "grad_accumulation_steps": 2,
            },
            "dpo": {
                "learning_rate": 1e-6,
                "max_steps": 25,
                "batch_size": 1,
                "grad_accumulation_steps": 1,
                "beta": 0.2,
            },
        },
        "evaluation": {"num_episodes_per_task": 5, "tasks": ["test-task"]},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(test_config_data, f)
        temp_config_path = f.name

    try:
        # 重置全局配置
        from src.utils import config

        original_config = config._config
        config._config = None

        # 加载自定义配置
        custom_config = load_config(temp_config_path)

        # 验证自定义配置
        assert custom_config.model.base_model_name == "test-model"
        assert custom_config.training.sft.learning_rate == 1e-4
        assert custom_config.training.dpo.beta == 0.2
        assert custom_config.evaluation.num_episodes_per_task == 5

    finally:
        # 清理
        os.unlink(temp_config_path)
        # 恢复原始配置
        from src.utils import config

        config._config = original_config


def test_config_invalid_yaml():
    """测试无效的YAML文件。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content: [")
        temp_config_path = f.name

    try:
        from src.utils import config

        original_config = config._config
        config._config = None

        with pytest.raises(yaml.YAMLError):
            load_config(temp_config_path)

    finally:
        os.unlink(temp_config_path)
        from src.utils import config

        config._config = original_config


def test_config_missing_required_fields():
    """测试缺少必需字段的配置文件。"""
    incomplete_config = {
        "model": {"base_model_name": "test-model"},
        # 缺少其他必需字段
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(incomplete_config, f)
        temp_config_path = f.name

    try:
        from src.utils import config

        original_config = config._config
        config._config = None

        with pytest.raises(Exception):  # Pydantic会抛出验证错误
            load_config(temp_config_path)

    finally:
        os.unlink(temp_config_path)
        from src.utils import config

        config._config = original_config
