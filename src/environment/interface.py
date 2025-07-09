# src/environment/interface.py

import gymnasium as gym
import re # 引入正则表达式库
from src.miniwob.action import ActionTypes, ActionSpaceConfig, Action
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
            task_id (str): MiniWoB++的任务ID (例如 'click-test' 或 'miniwob/click-button-v1').
        """
        self.task_id = task_id
        # 解析任务名称：如果包含'/'则取后半部分，否则直接使用
        if '/' in task_id:
            subdomain = task_id.split('/')[1]
        else:
            subdomain = task_id
            
        # 注意：由于我们是本地使用，创建环境的方式略有不同
        self.env = MiniWoBEnvironment(
            subdomain=subdomain,
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

    def step(self, action_string: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        在环境中执行一个动作。

        Args:
            action_string (str): 一个格式化的动作字符串，例如 'CLICK(selector="#button-1")'

        Returns:
            Tuple: observation, reward, terminated, truncated, info
        """
        # 1. 解析动作字符串
        parsed_action = self._parse_action(action_string)
        
        if parsed_action:
            # 2. 如果解析成功，执行动作
            return self.env.step(parsed_action)
        else:
            # 3. 如果解析失败，返回一个惩罚，并认为任务失败
            print(f"动作解析失败: '{action_string}'")
            # 返回一个表示失败的元组：空的观察值，-1的奖励，任务终止
            # 注意：这里的返回值结构需要与gym环境的返回值完全一致
            dummy_obs = self.env.observation_space.sample() # 创建一个合规的空观察
            for key in dummy_obs:
                dummy_obs[key] = None # 清空内容
                
            return dummy_obs, -1.0, True, False, {"error": "Invalid action format"}

    # 新增一个私有方法用于解析
    def _parse_action(self, action_string: str) -> Action | None:
        """
        使用正则表达式从字符串解析出具体的Action对象。
        这是一个简化的实现，可以根据需要扩展。
        """
        action_string = action_string.strip()
        
        # 匹配 CLICK(selector="...")
        click_match = re.match(r'CLICK\(selector="(.+?)"\)', action_string)
        if click_match:
            selector = click_match.group(1)
            # 创建CLICK_ELEMENT类型的Action
            return {
                "action_type": ActionTypes.CLICK_ELEMENT,
                "ref": 0,  # 简化处理，使用固定的ref值
            }

        # 匹配 TYPE(selector="...", text="...")
        type_match = re.match(r'TYPE\(selector="(.+?)", text="(.+?)"\)', action_string)
        if type_match:
            selector, text = type_match.groups()
            # 创建FOCUS_ELEMENT_AND_TYPE_TEXT类型的Action
            return {
                "action_type": ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT,
                "ref": 0,  # 简化处理，使用固定的ref值
                "text": text
            }
            
        # 如果都未匹配，返回None
        return None

    def close(self):
        """
        关闭环境，释放资源。
        """
        print("关闭环境。")
        self.env.close() 