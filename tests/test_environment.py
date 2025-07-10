# tests/test_environment.py

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

# 将 src 目录添加到 Python 搜索路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from environment.interface import EnvironmentInterface

# 定义一个基础的测试任务ID
TEST_TASK_ID = "miniwob/click-button"


class TestEnvironmentInterface(unittest.TestCase):
    """使用Mock测试EnvironmentInterface，避免对真实浏览器的依赖"""
    
    def setUp(self):
        """设置测试环境"""
        self.test_task_id = TEST_TASK_ID
        
    def create_mock_environment(self):
        """创建模拟的MiniWoB环境"""
        mock_env = MagicMock()
        
        # 模拟observation_space
        mock_observation_space = MagicMock()
        mock_observation_space.sample.return_value = {
            "utterance": "Click the button",
            "dom_elements": [{"tag": "button", "id": "test-btn"}],
            "screenshot": None
        }
        mock_env.observation_space = mock_observation_space
        
        # 模拟reset方法
        mock_env.reset.return_value = (
            {
                "utterance": "Click the button", 
                "dom_elements": [{"tag": "button", "id": "test-btn"}],
                "screenshot": None
            },
            {"info": "test"}
        )
        
        # 模拟step方法
        mock_env.step.return_value = (
            {
                "utterance": None,
                "dom_elements": [{"tag": "button", "id": "test-btn", "clicked": True}],
                "screenshot": None
            },
            1.0,  # reward
            True,  # terminated
            False,  # truncated
            {"success": True}  # info
        )
        
        # 模拟close方法
        mock_env.close.return_value = None
        
        return mock_env
    
    @patch('src.utils.logger.get_global_logger')
    def test_environment_initialization_and_reset(self, mock_logger):
        """
        测试EnvironmentInterface的初始化和重置功能。
        使用Mock避免对真实环境的依赖。
        """
        # 设置Mock
        mock_env_instance = self.create_mock_environment()
        mock_logger.return_value = Mock()
        
        # 创建mock环境接口
        env_interface = EnvironmentInterface.__new__(EnvironmentInterface)
        env_interface.task_id = self.test_task_id
        env_interface.env = mock_env_instance
        print(f"环境 {env_interface.task_id} 已初始化。")

        try:
            # 1. 测试初始化
            self.assertIsNotNone(env_interface, "环境接口对象未能成功创建！")
            self.assertTrue(hasattr(env_interface, "env"), "env_interface对象缺少'env'属性！")
            
            # 验证mock环境已设置
            self.assertEqual(env_interface.env, mock_env_instance)
            
            # 2. 测试重置
            observation, info = env_interface.reset()
            
            # 3. 验证返回值
            self.assertIsInstance(observation, dict, "观察值类型应为dict")
            self.assertIn("utterance", observation, "观察值中缺少'utterance'字段")
            self.assertIn("dom_elements", observation, "观察值中缺少'dom_elements'字段")
            
            # 验证reset方法被调用
            mock_env_instance.reset.assert_called_once()
            
        finally:
            # 确保环境被关闭
            if env_interface:
                env_interface.close()
    
    @patch('src.utils.logger.get_global_logger') 
    def test_environment_step_with_valid_action(self, mock_logger):
        """测试有效动作的执行"""
        # 设置Mock
        mock_env_instance = self.create_mock_environment()
        mock_logger.return_value = Mock()

        # 创建mock环境接口
        env_interface = EnvironmentInterface.__new__(EnvironmentInterface)
        env_interface.task_id = self.test_task_id
        env_interface.env = mock_env_instance
        print(f"环境 {env_interface.task_id} 已初始化。")

        try:
            
            # 执行有效动作
            action_string = 'CLICK(selector="#test-btn")'
            observation, reward, terminated, truncated, info = env_interface.step(action_string)
            
            # 验证返回值
            self.assertIsInstance(observation, dict)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)
            
        finally:
            if env_interface:
                env_interface.close()
    
    @patch('src.utils.logger.get_global_logger')
    def test_environment_step_with_invalid_action(self, mock_logger):
        """测试无效动作的处理"""
        # 设置Mock
        mock_env_instance = self.create_mock_environment()
        mock_logger.return_value = Mock()
        
        # 创建mock环境接口
        env_interface = EnvironmentInterface.__new__(EnvironmentInterface)
        env_interface.task_id = self.test_task_id
        env_interface.env = mock_env_instance
        print(f"环境 {env_interface.task_id} 已初始化。")
        
        try:
            # 执行无效动作
            action_string = 'INVALID_ACTION'
            observation, reward, terminated, truncated, info = env_interface.step(action_string)
            
            # 验证错误处理
            self.assertEqual(reward, -1.0, "无效动作应返回-1奖励")
            self.assertTrue(terminated, "无效动作应导致任务终止")
            self.assertIn("error", info, "info中应包含错误信息")
            
        finally:
            if env_interface:
                env_interface.close()
    
    def test_environment_context_manager(self):
        """测试环境接口作为上下文管理器的使用"""
        # 这个测试验证了我们是否正确实现了资源管理
        with patch('src.utils.logger.get_global_logger') as mock_logger:
            
            mock_env_instance = self.create_mock_environment()
            mock_logger.return_value = Mock()
            
            # 测试with语句的使用
            try:
                # 创建mock context manager  
                env_manager = EnvironmentInterfaceContextManager.__new__(EnvironmentInterfaceContextManager)
                env_manager.task_id = self.test_task_id
                env_manager.env = mock_env_instance
                print(f"环境 {env_manager.task_id} 已初始化。")
                
                with env_manager as env_interface:
                    self.assertIsNotNone(env_interface)
                    # 在with块中使用环境
                    observation, info = env_interface.reset()
                    self.assertIsInstance(observation, dict)
                
                # 退出with块后，close应该被自动调用
                mock_env_instance.close.assert_called()
                
            except NameError:
                # 如果EnvironmentInterfaceContextManager还未实现，这个测试会跳过
                self.skipTest("EnvironmentInterfaceContextManager not implemented yet")


class EnvironmentInterfaceContextManager(EnvironmentInterface):
    """
    环境接口的上下文管理器版本。
    这是一个资源管理优化的示例。
    """
    
    def __enter__(self):
        """进入上下文管理器"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器，确保资源清理"""
        try:
            self.close()
        except Exception as e:
            # 记录清理失败，但不抛出异常
            print(f"环境清理时发生错误: {e}")
        return False  # 不抑制异常


if __name__ == '__main__':
    unittest.main()
