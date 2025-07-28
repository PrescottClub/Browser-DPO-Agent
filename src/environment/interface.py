# src/environment/interface.py

import re  # 引入正则表达式库

import gymnasium as gym

# Import warning filters early to suppress noisy warnings
from src.utils.warning_filters import configure_warnings_for_dpo_driver

try:
    from src.miniwob import MiniWoBEnvironment
    from src.miniwob.action import Action, ActionSpaceConfig, ActionTypes
except ImportError:
    from miniwob.action import ActionTypes, ActionSpaceConfig, Action
    from miniwob import MiniWoBEnvironment

from typing import Any, Dict, Tuple, Optional, Union, List

from selenium.webdriver.common.by import By

# 导入新的日志系统
from src.utils.logger import get_global_logger, log_action_parsing_error


class ElementReferenceManager:
    """
    元素引用管理器，负责元素定位和引用管理。
    """

    def __init__(self):
        self.element_cache: Dict[int, Any] = {}
        self.ref_counter: int = 0
        self.selector_cache: Dict[str, int] = {}

    def get_element_ref(self, selector: str, page_context=None) -> int:
        """
        根据选择器获取元素引用。

        Args:
            selector (str): CSS选择器
            page_context: 页面上下文（在实际环境中使用）

        Returns:
            int: 元素引用ID

        Raises:
            ElementNotFoundError: 当元素未找到时
            ElementLocationError: 当元素定位失败时
        """
        # 检查缓存
        if selector in self.selector_cache:
            cached_ref = self.selector_cache[selector]
            if cached_ref in self.element_cache:
                return cached_ref

        try:
            # 在实际环境中，这里会使用page_context来查找元素
            # 目前为了兼容性，我们生成一个新的引用
            self.ref_counter += 1
            element_ref = self.ref_counter

            # 缓存元素引用
            self.element_cache[element_ref] = {
                "selector": selector,
                "timestamp": self._get_timestamp(),
                "valid": True
            }
            self.selector_cache[selector] = element_ref

            return element_ref

        except Exception as e:
            raise ElementLocationError(f"Failed to locate element with selector '{selector}': {e}")

    def invalidate_cache(self):
        """清理缓存"""
        self.element_cache.clear()
        self.selector_cache.clear()

    def is_valid_ref(self, ref: int) -> bool:
        """检查引用是否有效"""
        return ref in self.element_cache and self.element_cache[ref].get("valid", False)

    def _get_timestamp(self):
        """获取当前时间戳"""
        import time
        return time.time()


class ElementNotFoundError(Exception):
    """元素未找到异常"""
    pass


class ElementLocationError(Exception):
    """元素定位异常"""
    pass

# 首先，配置Action Space，我们使用基础的元素操作
# 这段代码应该在类的外部，作为模块级别的配置
ACTION_SPACE_CONFIG = ActionSpaceConfig(
    action_types=[
        ActionTypes.CLICK_ELEMENT,
        ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT,
        # 添加更多动作类型支持
    ]
)

# 注册 MiniWoB++ 环境
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Overriding environment.*")

try:
    from src.miniwob.registration import register_miniwob_envs
except ImportError:
    from miniwob.registration import register_miniwob_envs
    
# 临时抑制警告进行环境注册
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
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

        # 初始化元素引用管理器
        self.element_manager = ElementReferenceManager()

        print(f"环境 {self.task_id} 已初始化。")

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        重置环境，开始一个新的回合。

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: 返回初始的观察（observation）和信息（info）。
        """
        # 清理元素引用缓存
        self.element_manager.invalidate_cache()

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
        增强的动作解析器，支持多种选择器格式和动作类型。

        Args:
            action_string (str): 待解析的动作字符串

        Returns:
            Tuple[Optional[Action], Optional[str]]: 返回 (解析后的Action对象, 错误信息)

        Supported formats:
            - CLICK(selector="css_selector")
            - TYPE(selector="css_selector", text="text")
            - SELECT(selector="css_selector", value="value")
            - CHECK(selector="css_selector")

        Supported selector formats:
            - Standard CSS: "#id", ".class", "tag[attr='value']"
            - jQuery-style: "tag:contains('text')"
            - Attribute selectors: "[data-test='value']"
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
            # 解析CLICK动作
            click_match = re.match(r'CLICK\(selector="(.+?)"\)', action_string)
            if click_match:
                selector = click_match.group(1)
                converted_selector = self._convert_selector(selector)

                if not converted_selector:
                    error_msg = f"Invalid or unsupported selector: {selector}"
                    log_action_parsing_error(
                        logger=logger,
                        action_str=action_string,
                        error_reason=error_msg,
                        expected_format="CLICK(selector=\"valid_css_selector\")"
                    )
                    return None, error_msg

                # 获取真正的元素引用
                try:
                    element_ref = self.element_manager.get_element_ref(converted_selector)
                    logger.debug(f"Successfully parsed CLICK action with selector: {converted_selector}, ref: {element_ref}")
                    return {
                        "action_type": ActionTypes.CLICK_ELEMENT,
                        "ref": element_ref,  # 使用真正的元素引用
                        "selector": converted_selector,  # 保存转换后的选择器
                    }, None
                except (ElementNotFoundError, ElementLocationError) as e:
                    error_msg = f"Failed to locate element for CLICK action: {e}"
                    log_action_parsing_error(
                        logger=logger,
                        action_str=action_string,
                        error_reason=error_msg,
                        expected_format="CLICK(selector=\"valid_css_selector\")"
                    )
                    return None, error_msg

            # 解析TYPE动作
            type_match = re.match(r'TYPE\(selector="(.+?)", text="(.+?)"\)', action_string)
            if type_match:
                selector, text = type_match.groups()
                converted_selector = self._convert_selector(selector)

                if not converted_selector:
                    error_msg = f"Invalid or unsupported selector in TYPE action: {selector}"
                    log_action_parsing_error(
                        logger=logger,
                        action_str=action_string,
                        error_reason=error_msg,
                        expected_format="TYPE(selector=\"valid_css_selector\", text=\"some_text\")"
                    )
                    return None, error_msg

                if text is None:  # 允许空文本，但不能是None
                    logger.warning(f"Empty text in TYPE action: {action_string}")
                    text = ""

                # 获取真正的元素引用
                try:
                    element_ref = self.element_manager.get_element_ref(converted_selector)
                    logger.debug(f"Successfully parsed TYPE action with selector: {converted_selector}, text: {repr(text)}, ref: {element_ref}")
                    return {
                        "action_type": ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT,
                        "ref": element_ref,  # 使用真正的元素引用
                        "text": text,
                        "selector": converted_selector,
                    }, None
                except (ElementNotFoundError, ElementLocationError) as e:
                    error_msg = f"Failed to locate element for TYPE action: {e}"
                    log_action_parsing_error(
                        logger=logger,
                        action_str=action_string,
                        error_reason=error_msg,
                        expected_format="TYPE(selector=\"valid_css_selector\", text=\"some_text\")"
                    )
                    return None, error_msg

            # 解析SELECT动作 (新增支持)
            select_match = re.match(r'SELECT\(selector="(.+?)", value="(.+?)"\)', action_string)
            if select_match:
                selector, value = select_match.groups()
                converted_selector = self._convert_selector(selector)

                if not converted_selector:
                    error_msg = f"Invalid or unsupported selector in SELECT action: {selector}"
                    return None, error_msg

                # 获取真正的元素引用
                try:
                    element_ref = self.element_manager.get_element_ref(converted_selector)
                    logger.debug(f"Successfully parsed SELECT action with selector: {converted_selector}, value: {value}, ref: {element_ref}")
                    # 注意：SELECT可能需要特殊处理，这里先转换为CLICK
                    return {
                        "action_type": ActionTypes.CLICK_ELEMENT,
                        "ref": element_ref,
                        "selector": converted_selector,
                        "select_value": value,  # 额外信息
                    }, None
                except (ElementNotFoundError, ElementLocationError) as e:
                    error_msg = f"Failed to locate element for SELECT action: {e}"
                    return None, error_msg

            # 解析CHECK动作 (新增支持)
            check_match = re.match(r'CHECK\(selector="(.+?)"\)', action_string)
            if check_match:
                selector = check_match.group(1)
                converted_selector = self._convert_selector(selector)

                if not converted_selector:
                    error_msg = f"Invalid or unsupported selector in CHECK action: {selector}"
                    return None, error_msg

                # 获取真正的元素引用
                try:
                    element_ref = self.element_manager.get_element_ref(converted_selector)
                    logger.debug(f"Successfully parsed CHECK action with selector: {converted_selector}, ref: {element_ref}")
                    return {
                        "action_type": ActionTypes.CLICK_ELEMENT,
                        "ref": element_ref,
                        "selector": converted_selector,
                        "action_subtype": "check",  # 标记为复选框操作
                    }, None
                except (ElementNotFoundError, ElementLocationError) as e:
                    error_msg = f"Failed to locate element for CHECK action: {e}"
                    return None, error_msg

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

    def _convert_selector(self, selector: str) -> str:
        """
        将各种格式的选择器转换为标准CSS选择器。

        Args:
            selector (str): 原始选择器

        Returns:
            str: 转换后的标准CSS选择器，如果无法转换则返回None
        """
        if not selector or selector.isspace():
            return None

        # 移除多余的空格
        selector = selector.strip()

        # 处理jQuery风格的:contains()选择器
        if ':contains(' in selector:
            return self._convert_contains_selector(selector)

        # 处理复合选择器（逗号分隔的多个选择器）
        if ',' in selector:
            selectors = [s.strip() for s in selector.split(',')]
            converted = []
            for s in selectors:
                conv = self._convert_selector(s)  # 递归转换
                if conv:
                    converted.append(conv)
            return ', '.join(converted) if converted else None

        # 验证标准CSS选择器格式
        if self._is_valid_css_selector(selector):
            return selector

        return None

    def _convert_contains_selector(self, selector: str) -> str:
        """
        将:contains()选择器转换为XPath或属性选择器。

        Examples:
            "button:contains('Login')" -> "button[text()='Login']" (XPath style)
            "a:contains('Home')" -> "a[text()='Home']"
        """
        # 匹配 tag:contains('text') 格式
        contains_pattern = r"([^:]+):contains\(['\"]([^'\"]+)['\"]\)"
        match = re.search(contains_pattern, selector)

        if match:
            tag, text = match.groups()
            # 转换为属性选择器的近似形式
            # 注意：这是一个简化的转换，实际应用中可能需要更复杂的逻辑
            return f"{tag}[title*='{text}'], {tag}[aria-label*='{text}'], {tag}[data-text*='{text}']"

        return selector

    def _is_valid_css_selector(self, selector: str) -> bool:
        """
        验证是否为有效的CSS选择器。

        Args:
            selector (str): 待验证的选择器

        Returns:
            bool: 是否为有效的CSS选择器
        """
        # 基本的CSS选择器格式验证
        # 这是一个简化的验证，实际应用中可能需要更严格的验证

        # 检查空字符串和纯空白字符
        if not selector or selector.isspace():
            return False

        # 检查是否包含基本的CSS选择器字符
        valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_#.[]()=:*^$|~+> \'"')
        if not all(c in valid_chars for c in selector):
            return False

        # 检查基本格式 - 扩展验证逻辑
        if (selector.startswith(('.', '#')) or
            selector.isalpha() or
            '[' in selector or
            '>' in selector or
            '+' in selector or
            '~' in selector or
            ' ' in selector):  # 包含空格的组合选择器
            return True

        return False

    def close(self):
        """
        关闭环境，释放资源。
        """
        print("关闭环境。")
        self.env.close()
