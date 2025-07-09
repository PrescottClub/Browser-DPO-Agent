# src/environment/interface.py

import gymnasium as gym
from src.miniwob.action import ActionTypes, ActionSpaceConfig
from src.miniwob import MiniWoBEnvironment
from selenium.webdriver.common.by import By
from typing import Dict, Any, Tuple

# 首先，配置Action Space，我们使用基础的元素操作
# 这段代码应该在类的外部，作为模块级别的配置
ACTION_SPACE_CONFIG = ActionSpaceConfig(
    action_types=[
        ActionTypes.CLICK_ELEMENT,
        ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT,
    ]
)

# 注册 MiniWoB++ 环境
# 由于我们是本地导入，所以不需要再调用 register_envs()

class EnvironmentInterface:
    """
    封装与MiniWoB++环境的交互，为Agent提供一个干净的接口。
    """

    def __init__(self, task_id: str):
        """
        初始化环境。

        Args:
            task_id (str): MiniWoB++的任务ID (例如 'miniwob/click-button-v1').
        """
        self.task_id = task_id
        # 注意：由于我们是本地使用，创建环境的方式略有不同
        self.env = MiniWoBEnvironment(
            subdomain=task_id.split('/')[1],
            action_space_config=ACTION_SPACE_CONFIG,
            render_mode='human'
        )
        print(f"环境 {self.task_id} 已初始化。")

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        重置环境，开始一个新的回合。

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: 返回初始的观察（observation）和信息（info）。
        """
        observation, info = self.env.reset()
        print("环境已重置。")
        return observation, info

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        在环境中执行一个动作。

        Args:
            action (str): 一个格式化的动作字符串，例如 'CLICK(selector="#button-1")'

        Returns:
            Tuple: observation, reward, terminated, truncated, info
        """
        # Phase 0阶段暂时不实现此方法
        print(f"执行动作（暂未实现）: {action}")
        pass

    def close(self):
        """
        关闭环境，释放资源。
        """
        print("关闭环境。")
        self.env.close() 