# tests/test_modular_architecture.py

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch

from src.agent import Agent, BaseModel, SFTModule, DPOModule, InferenceModule


class TestBaseModel(unittest.TestCase):
    """测试BaseModel基础类"""
    
    def setUp(self):
        self.model_name = "microsoft/DialoGPT-small"  # 使用较小的测试模型
        
    @patch('src.agent.base_model.AutoModelForCausalLM.from_pretrained')
    @patch('src.agent.base_model.AutoTokenizer.from_pretrained')
    def test_base_model_initialization(self, mock_tokenizer, mock_model):
        """测试BaseModel初始化"""
        # Mock返回值
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        
        base_model = BaseModel(self.model_name)
        
        # 测试延迟加载 - 模型还未真正加载
        self.assertIsNone(base_model._model)
        self.assertIsNone(base_model._tokenizer)
        
        # 访问属性时才会加载
        _ = base_model.model
        _ = base_model.tokenizer
        
        # 验证加载被调用
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()
    
    def test_get_model_info(self):
        """测试模型信息获取"""
        with patch('src.agent.base_model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.agent.base_model.AutoTokenizer.from_pretrained'):
            
            base_model = BaseModel(self.model_name)
            info = base_model.get_model_info()
            
            self.assertEqual(info["model_name"], self.model_name)
            self.assertFalse(info["quantization_enabled"])
            self.assertEqual(info["device_map"], "auto")
    
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.is_available', return_value=True)
    def test_clear_cache(self, mock_cuda_available, mock_empty_cache):
        """测试GPU缓存清理"""
        with patch('src.agent.base_model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.agent.base_model.AutoTokenizer.from_pretrained'):
            
            base_model = BaseModel(self.model_name)
            base_model.clear_cache()
            
            mock_empty_cache.assert_called_once()


class TestSFTModule(unittest.TestCase):
    """测试SFTModule类"""
    
    def setUp(self):
        self.model_name = "microsoft/DialoGPT-small"
        
    @patch('src.agent.base_model.AutoModelForCausalLM.from_pretrained')
    @patch('src.agent.base_model.AutoTokenizer.from_pretrained')
    def test_sft_module_initialization(self, mock_tokenizer, mock_model):
        """测试SFTModule初始化"""
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        
        sft_module = SFTModule(self.model_name)
        
        self.assertEqual(sft_module.model_name, self.model_name)
        self.assertIsNone(sft_module.current_trainer)
    
    def test_create_lora_config(self):
        """测试LoRA配置创建"""
        with patch('src.agent.base_model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.agent.base_model.AutoTokenizer.from_pretrained'):
            
            sft_module = SFTModule(self.model_name)
            lora_config = sft_module.create_lora_config()
            
            self.assertEqual(lora_config.r, 16)
            self.assertEqual(lora_config.lora_alpha, 32)
            self.assertEqual(lora_config.task_type, "CAUSAL_LM")
    
    def test_create_training_args(self):
        """测试训练参数创建"""
        with patch('src.agent.base_model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.agent.base_model.AutoTokenizer.from_pretrained'):
            
            sft_module = SFTModule(self.model_name)
            training_args = sft_module.create_training_args(
                output_dir="./test_output",
                max_steps=10,
                learning_rate=1e-4
            )
            
            self.assertEqual(training_args.output_dir, "./test_output")
            self.assertEqual(training_args.max_steps, 10)
            self.assertEqual(training_args.learning_rate, 1e-4)
    
    def test_get_trainer_metrics_no_trainer(self):
        """测试无训练器时的指标获取"""
        with patch('src.agent.base_model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.agent.base_model.AutoTokenizer.from_pretrained'):
            
            sft_module = SFTModule(self.model_name)
            metrics = sft_module.get_trainer_metrics()
            
            self.assertEqual(metrics["status"], "no_trainer")


class TestDPOModule(unittest.TestCase):
    """测试DPOModule类"""
    
    def setUp(self):
        self.model_name = "microsoft/DialoGPT-small"
        
    @patch('src.agent.base_model.AutoModelForCausalLM.from_pretrained')
    @patch('src.agent.base_model.AutoTokenizer.from_pretrained')
    def test_dpo_module_initialization(self, mock_tokenizer, mock_model):
        """测试DPOModule初始化"""
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        
        dpo_module = DPOModule(self.model_name)
        
        self.assertEqual(dpo_module.model_name, self.model_name)
        self.assertIsNone(dpo_module.current_trainer)
    
    def test_create_dpo_config(self):
        """测试DPO配置创建"""
        with patch('src.agent.base_model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.agent.base_model.AutoTokenizer.from_pretrained'):
            
            dpo_module = DPOModule(self.model_name)
            dpo_config = dpo_module.create_dpo_config(
                output_dir="./test_output",
                beta=0.2,
                max_steps=20
            )
            
            self.assertEqual(dpo_config.output_dir, "./test_output")
            self.assertEqual(dpo_config.beta, 0.2)
            self.assertEqual(dpo_config.max_steps, 20)
    
    def test_get_preference_analysis_no_trainer(self):
        """测试无训练器时的偏好分析"""
        with patch('src.agent.base_model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.agent.base_model.AutoTokenizer.from_pretrained'):
            
            dpo_module = DPOModule(self.model_name)
            analysis = dpo_module.get_preference_analysis()
            
            self.assertEqual(analysis["status"], "no_data")


class TestInferenceModule(unittest.TestCase):
    """测试InferenceModule类"""
    
    def setUp(self):
        self.model_name = "microsoft/DialoGPT-small"
        
    @patch('src.agent.base_model.AutoModelForCausalLM.from_pretrained')
    @patch('src.agent.base_model.AutoTokenizer.from_pretrained')
    def test_inference_module_initialization(self, mock_tokenizer, mock_model):
        """测试InferenceModule初始化"""
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        
        inference_module = InferenceModule(self.model_name)
        
        self.assertEqual(inference_module.model_name, self.model_name)
        self.assertIn("max_new_tokens", inference_module.default_generation_config)
    
    def test_format_prompt(self):
        """测试prompt格式化"""
        with patch('src.agent.base_model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.agent.base_model.AutoTokenizer.from_pretrained'):
            
            inference_module = InferenceModule(self.model_name)
            formatted = inference_module.format_prompt("Click the button")
            
            self.assertIn("### Instruction:", formatted)
            self.assertIn("Click the button", formatted)
            self.assertIn("### Response:", formatted)
    
    def test_parse_response(self):
        """测试响应解析"""
        with patch('src.agent.base_model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.agent.base_model.AutoTokenizer.from_pretrained'):
            
            inference_module = InferenceModule(self.model_name)
            
            # 测试正常解析
            response = "Thought: I need to click the button\nAction: CLICK(selector='#button')"
            parsed = inference_module.parse_response(response)
            
            self.assertEqual(parsed["thought"], "I need to click the button")
            self.assertEqual(parsed["action"], "CLICK(selector='#button')")
            
            # 测试空响应
            empty_parsed = inference_module.parse_response("")
            self.assertEqual(empty_parsed["thought"], "")
            self.assertEqual(empty_parsed["action"], "")
    
    def test_update_generation_config(self):
        """测试生成配置更新"""
        with patch('src.agent.base_model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.agent.base_model.AutoTokenizer.from_pretrained'):
            
            inference_module = InferenceModule(self.model_name)
            
            # 更新配置
            inference_module.update_generation_config(
                max_new_tokens=512,
                temperature=0.8
            )
            
            self.assertEqual(inference_module.default_generation_config["max_new_tokens"], 512)
            self.assertEqual(inference_module.default_generation_config["temperature"], 0.8)


class TestAgent(unittest.TestCase):
    """测试Agent协调器类"""
    
    def setUp(self):
        self.model_name = "microsoft/DialoGPT-small"
        
    def test_agent_initialization(self):
        """测试Agent初始化"""
        agent = Agent(self.model_name)
        
        self.assertEqual(agent.model_name, self.model_name)
        self.assertIsNone(agent._sft_module)
        self.assertIsNone(agent._dpo_module)
        self.assertIsNone(agent._inference_module)
        self.assertIsNone(agent._current_module)
    
    @patch('src.agent.agent.SFTModule')
    def test_sft_module_lazy_loading(self, mock_sft_class):
        """测试SFT模块的延迟加载"""
        mock_sft_instance = Mock()
        mock_sft_class.return_value = mock_sft_instance
        
        agent = Agent(self.model_name)
        
        # 第一次访问时创建模块
        sft_module = agent.sft_module
        
        mock_sft_class.assert_called_once_with(
            self.model_name, None, "auto"
        )
        self.assertEqual(agent._current_module, "sft")
        self.assertIn("sft", agent._module_history)
    
    @patch('src.agent.agent.DPOModule')
    def test_dpo_module_lazy_loading(self, mock_dpo_class):
        """测试DPO模块的延迟加载"""
        mock_dpo_instance = Mock()
        mock_dpo_class.return_value = mock_dpo_instance
        
        agent = Agent(self.model_name)
        
        # 第一次访问时创建模块
        dpo_module = agent.dpo_module
        
        mock_dpo_class.assert_called_once_with(
            self.model_name, None, "auto"
        )
        self.assertEqual(agent._current_module, "dpo")
        self.assertIn("dpo", agent._module_history)
    
    @patch('src.agent.agent.InferenceModule')
    def test_inference_module_lazy_loading(self, mock_inference_class):
        """测试推理模块的延迟加载"""
        mock_inference_instance = Mock()
        mock_inference_class.return_value = mock_inference_instance
        
        agent = Agent(self.model_name)
        
        # 第一次访问时创建模块
        inference_module = agent.inference_module
        
        mock_inference_class.assert_called_once_with(
            self.model_name, None, "auto"
        )
        self.assertEqual(agent._current_module, "inference")
        self.assertIn("inference", agent._module_history)
    
    def test_get_agent_status(self):
        """测试Agent状态获取"""
        agent = Agent(self.model_name)
        status = agent.get_agent_status()
        
        self.assertEqual(status["model_name"], self.model_name)
        self.assertIsNone(status["current_module"])
        self.assertEqual(status["initialized_modules"], [])
        self.assertFalse(status["quantization_enabled"])
    
    def test_context_manager(self):
        """测试上下文管理器功能"""
        with Agent(self.model_name) as agent:
            self.assertIsInstance(agent, Agent)
        
        # 测试__exit__是否正确清理资源
        self.assertIsNone(agent._current_module)
    
    def test_repr(self):
        """测试字符串表示"""
        agent = Agent(self.model_name)
        repr_str = repr(agent)
        
        self.assertIn(self.model_name, repr_str)
        self.assertIn("Agent", repr_str)


if __name__ == '__main__':
    unittest.main() 