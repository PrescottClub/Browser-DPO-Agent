# tests/test_environment.py

import sys
import os
import pytest

# 将 src 目录添加到 Python 搜索路径中
# 这使得我们可以像导入已安装的包一样导入 src 下的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from environment.interface import EnvironmentInterface

# 定义一个基础的测试任务ID
TEST_TASK_ID = 'miniwob/click-button'

def test_environment_initialization_and_reset():
    """
    测试EnvironmentInterface是否能成功初始化并重置。
    这是验证环境通路的关键。
    """
    print("开始测试环境初始化与重置...")
    
    env_interface = None
    try:
        # 1. 测试初始化
        env_interface = EnvironmentInterface(task_id=TEST_TASK_ID)
        assert env_interface is not None, "环境接口对象未能成功创建！"
        # 由于我们不再使用gym.make，所以env的检查方式改变
        assert hasattr(env_interface, 'env'), "env_interface对象缺少'env'属性！"
        print(f"成功初始化环境接口，任务ID: {TEST_TASK_ID}")

        # 2. 测试重置
        observation, info = env_interface.reset()
        
        # 3. 验证返回值
        assert isinstance(observation, dict), f"观察值类型错误，应为dict，实际为{type(observation)}"
        assert 'utterance' in observation, "观察值中缺少'utterance'字段"
        assert 'dom_elements' in observation, "观察值中缺少'dom_elements'字段"
        print("成功重置环境，并获得有效观察值。")

    except Exception as e:
        pytest.fail(f"环境接口测试过程中发生意外错误: {e}")

    finally:
        # 确保环境被关闭
        if env_interface:
            env_interface.close()
        print("测试结束，环境已清理。") 