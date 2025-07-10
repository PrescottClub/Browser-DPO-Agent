# src/environment/interface.py

import re  # 引入正则表达式库

import gymnasium as gym

try:
    from src.miniwob import MiniWoBEnvironment
    from src.miniwob.action import Action, ActionSpaceConfig, ActionTypes
except ImportError:
    from miniwob.action import ActionTypes, ActionSpaceConfig, Action
    from miniwob import MiniWoBEnvironment

from typing import Any, Dict, Tuple, Optional, Union

from selenium.webdriver.common.by import By

# 导入新的日志系统
from src.utils.logger import get_global_logger, log_action_parsing_error

# 首先，配置Action Space，我们使用基础的元素操作
# 这段代码应该在类的外部，作为模块级别的配置
ACTION_SPACE_CONFIG = ActionSpaceConfig(
    action_types=[
        ActionTypes.CLICK_ELEMENT,
        ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT,
    ]
)

# 注册 MiniWoB++ 环境
try:
    from src.miniwob.registration import register_miniwob_envs
except ImportError:
    from miniwob.registration import register_miniwob_envs
register_miniwob_envs()


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
        # 初始化日志系统
        from src.utils.logger import init_global_logger
        init_global_logger()
        
        self.task_id = task_id
        # 解析任务名称：如果包含'/'则取后半部分，否则直接使用
        if "/" in task_id:
            subdomain = task_id.split("/")[1]
        else:
            subdomain = task_id

        # 注意：由于我们是本地使用，创建环境的方式略有不同
        self.env = MiniWoBEnvironment(
            subdomain=subdomain,
            action_space_config=ACTION_SPACE_CONFIG,
            render_mode="human",
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

    def step(
        self, action_string: str
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        在环境中执行一个动作。

        Args:
            action_string (str): 一个格式化的动作字符串，例如 'CLICK(selector="#button-1")'

        Returns:
            Tuple: observation, reward, terminated, truncated, info
        """
        logger = get_global_logger()
        
        # 1. 解析动作字符串
        parsed_action, error_message = self._parse_action(action_string)

        if parsed_action is not None:
            # 2. 如果解析成功，执行动作
            logger.debug(f"Executing parsed action: {parsed_action}")
            return self.env.step(parsed_action)
        else:
            # 3. 如果解析失败，返回一个惩罚，并认为任务失败
            logger.warning(f"Action parsing failed for: '{action_string}' - {error_message}")
            print(f"动作解析失败: '{action_string}' - {error_message}")
            
            # 返回一个表示失败的元组：空的观察值，-1的奖励，任务终止
            # 注意：这里的返回值结构需要与gym环境的返回值完全一致
            dummy_obs = self.env.observation_space.sample()  # 创建一个合规的空观察
            for key in dummy_obs:
                dummy_obs[key] = None  # 清空内容

            return dummy_obs, -1.0, True, False, {
                "error": "Invalid action format",
                "error_message": error_message,
                "original_action": action_string
            }

    # 新增一个私有方法用于解析
    def _parse_action(self, action_string: str) -> Tuple[Optional[Action], Optional[str]]:
        """
        使用正则表达式从字符串解析出具体的Action对象。
        
        Args:
            action_string (str): 待解析的动作字符串
            
        Returns:
            Tuple[Optional[Action], Optional[str]]: 返回 (解析后的Action对象, 错误信息)
            - 如果解析成功: (Action对象, None)
            - 如果解析失败: (None, 错误信息字符串)
            
        Examples:
            >>> interface._parse_action('CLICK(selector="#button-1")')
            ({'action_type': ActionTypes.CLICK_ELEMENT, 'ref': 0}, None)
            
            >>> interface._parse_action('TYPE(selector="input", text="hello")')
            ({'action_type': ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT, 'ref': 0, 'text': 'hello'}, None)
            
            >>> interface._parse_action('INVALID_ACTION')
            (None, 'Unknown action format: INVALID_ACTION')
        """
        logger = get_global_logger()
        
        # 输入验证
        if action_string is None:
            error_msg = "Action string is None"
            log_action_parsing_error(
                logger=logger,
                action_str=action_string,
                error_reason=error_msg,
                expected_format="CLICK(selector=\"...\") or TYPE(selector=\"...\", text=\"...\")"
            )
            return None, error_msg
        
        # 预处理
        action_string = action_string.strip()
        
        if not action_string:
            error_msg = "Action string is empty"
            log_action_parsing_error(
                logger=logger,
                action_str=action_string,
                error_reason=error_msg,
                expected_format="CLICK(selector=\"...\") or TYPE(selector=\"...\", text=\"...\")"
            )
            return None, error_msg
        
        logger.debug(f"Parsing action string: {repr(action_string)}")
        
        try:
            # 匹配 CLICK(selector="...")
            click_match = re.match(r'CLICK\(selector="(.+?)"\)', action_string)
            if click_match:
                selector = click_match.group(1)
                
                # 验证selector是否有效
                if not selector or selector.isspace():
                    error_msg = "Empty or whitespace-only selector in CLICK action"
                    log_action_parsing_error(
                        logger=logger,
                        action_str=action_string,
                        error_reason=error_msg,
                        expected_format="CLICK(selector=\"valid_css_selector\")"
                    )
                    return None, error_msg
                
                logger.debug(f"Successfully parsed CLICK action with selector: {selector}")
                # 创建CLICK_ELEMENT类型的Action
                return {
                    "action_type": ActionTypes.CLICK_ELEMENT,
                    "ref": 0,  # 简化处理，使用固定的ref值
                }, None

            # 匹配 TYPE(selector="...", text="...")
            type_match = re.match(r'TYPE\(selector="(.+?)", text="(.+?)"\)', action_string)
            if type_match:
                selector, text = type_match.groups()
                
                # 验证selector和text是否有效
                if not selector or selector.isspace():
                    error_msg = "Empty or whitespace-only selector in TYPE action"
                    log_action_parsing_error(
                        logger=logger,
                        action_str=action_string,
                        error_reason=error_msg,
                        expected_format="TYPE(selector=\"valid_css_selector\", text=\"some_text\")"
                    )
                    return None, error_msg
                
                if not text:  # 允许空文本，但不能是None
                    logger.warning(f"Empty text in TYPE action: {action_string}")
                
                logger.debug(f"Successfully parsed TYPE action with selector: {selector}, text: {repr(text)}")
                # 创建FOCUS_ELEMENT_AND_TYPE_TEXT类型的Action
                return {
                    "action_type": ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT,
                    "ref": 0,  # 简化处理，使用固定的ref值
                    "text": text,
                }, None

            # 如果都未匹配，记录详细的错误信息
            error_msg = f"Unknown action format: {action_string}"
            
            # 提供更具体的错误建议
            if "CLICK" in action_string.upper():
                expected_format = "CLICK(selector=\"css_selector_here\")"
            elif "TYPE" in action_string.upper():
                expected_format = "TYPE(selector=\"css_selector_here\", text=\"text_here\")"
            else:
                expected_format = "CLICK(selector=\"...\") or TYPE(selector=\"...\", text=\"...\")"
            
            log_action_parsing_error(
                logger=logger,
                action_str=action_string,
                error_reason=error_msg,
                expected_format=expected_format
            )
            
            return None, error_msg
            
        except re.error as e:
            error_msg = f"Regular expression error while parsing action: {str(e)}"
            log_action_parsing_error(
                logger=logger,
                action_str=action_string,
                error_reason=error_msg,
                expected_format="CLICK(selector=\"...\") or TYPE(selector=\"...\", text=\"...\")"
            )
            return None, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error while parsing action: {str(e)}"
            log_action_parsing_error(
                logger=logger,
                action_str=action_string,
                error_reason=error_msg,
                expected_format="CLICK(selector=\"...\") or TYPE(selector=\"...\", text=\"...\")"
            )
            return None, error_msg

    def close(self):
        """
        关闭环境，释放资源。
        """
        print("关闭环境。")
        self.env.close()
