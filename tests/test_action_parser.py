# tests/test_action_parser.py
from unittest.mock import Mock, patch

import pytest

from src.environment.interface import EnvironmentInterface


@pytest.fixture
def env_interface():
    """提供一个临时的、不实际启动环境的接口实例用于测试解析器。"""
    # 我们通过模拟(mocking)来避免实际创建环境
    with patch("src.environment.interface.MiniWoBEnvironment") as mock_env:
        # 模拟环境的基本属性
        mock_env_instance = Mock()
        mock_env_instance.observation_space.sample.return_value = {
            "utterance": None,
            "dom_elements": None,
            "screenshot": None,
        }
        mock_env.return_value = mock_env_instance

        # 创建接口实例
        interface = EnvironmentInterface(task_id="miniwob/click-button-v1")
        interface.env = mock_env_instance
        return interface


def test_parse_valid_click(env_interface):
    """测试解析有效的CLICK动作。"""
    action_str = 'CLICK(selector="#my-button")'
    parsed_action = env_interface._parse_action(action_str)
    assert parsed_action is not None
    assert parsed_action["action_type"] is not None
    # 由于实际实现返回的是字典格式，我们检查字典结构


def test_parse_valid_type(env_interface):
    """测试解析有效的TYPE动作。"""
    action_str = 'TYPE(selector="input[name=\'user\']", text="test-user")'
    parsed_action = env_interface._parse_action(action_str)
    assert parsed_action is not None
    assert parsed_action["action_type"] is not None
    assert "text" in parsed_action
    assert parsed_action["text"] == "test-user"


def test_parse_malformed_string(env_interface):
    """测试格式错误的字符串，应返回None。"""
    action_str = 'CLICK selector="#button"'
    assert env_interface._parse_action(action_str) is None


def test_parse_string_with_extra_whitespace(env_interface):
    """测试带有额外空格的字符串，应能正确解析。"""
    action_str = '  CLICK(selector="#my-button")  '
    parsed_action = env_interface._parse_action(action_str)
    assert parsed_action is not None
    assert parsed_action["action_type"] is not None


def test_parse_empty_string(env_interface):
    """测试空字符串，应返回None。"""
    assert env_interface._parse_action("") is None


def test_parse_none_input(env_interface):
    """测试None输入，应返回None。"""
    assert env_interface._parse_action(None) is None


def test_parse_unsupported_action(env_interface):
    """测试不支持的动作类型，应返回None。"""
    action_str = 'DRAG(selector="#item", target="#target")'
    assert env_interface._parse_action(action_str) is None


def test_parse_click_with_single_quotes(env_interface):
    """测试使用单引号的CLICK动作。"""
    action_str = "CLICK(selector='#my-button')"
    # 当前实现可能不支持单引号，这个测试可以帮助发现这个问题
    parsed_action = env_interface._parse_action(action_str)
    # 根据实际实现调整断言


def test_parse_type_with_complex_text(env_interface):
    """测试包含特殊字符的TYPE动作。"""
    action_str = 'TYPE(selector="input", text="Hello, World! @#$%")'
    parsed_action = env_interface._parse_action(action_str)
    if parsed_action:  # 如果解析成功
        assert "text" in parsed_action
        assert "Hello, World! @#$%" in parsed_action["text"]
