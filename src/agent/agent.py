# src/agent/agent.py

from typing import Optional, Dict, Any
from transformers import BitsAndBytesConfig

from .sft_module import SFTModule
from .dpo_module import DPOModule
from .inference_module import InferenceModule


class Agent:
    """
    Agent协调器类，负责组合和协调各个专业化模块。
    
    使用组合模式而非继承，遵循"组合优于继承"的设计原则。
    提供统一的外部接口，同时保持向后兼容性。
    """

    def __init__(
        self, 
        model_name: str, 
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device_map: str = "auto"
    ):
        """
        初始化Agent协调器。

        Args:
            model_name (str): Hugging Face上的模型名称
            quantization_config (BitsAndBytesConfig, optional): 量化配置
            device_map (str): 设备映射策略
        """
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.device_map = device_map
        
        # 延迟初始化各个模块
        self._sft_module = None
        self._dpo_module = None
        self._inference_module = None
        
        # 当前活动模块跟踪
        self._current_module = None
        self._module_history = []
    
    @property
    def sft_module(self) -> SFTModule:
        """延迟加载SFT模块"""
        if self._sft_module is None:
            print("初始化SFT训练模块...")
            self._sft_module = SFTModule(
                self.model_name, 
                self.quantization_config, 
                self.device_map
            )
            self._current_module = "sft"
            self._module_history.append("sft")
        return self._sft_module
    
    @property
    def dpo_module(self) -> DPOModule:
        """延迟加载DPO模块"""
        if self._dpo_module is None:
            print("初始化DPO训练模块...")
            self._dpo_module = DPOModule(
                self.model_name, 
                self.quantization_config, 
                self.device_map
            )
            self._current_module = "dpo"
            self._module_history.append("dpo")
        return self._dpo_module
    
    @property
    def inference_module(self) -> InferenceModule:
        """延迟加载推理模块"""
        if self._inference_module is None:
            print("初始化推理模块...")
            self._inference_module = InferenceModule(
                self.model_name, 
                self.quantization_config, 
                self.device_map
            )
            self._current_module = "inference"
            self._module_history.append("inference")
        return self._inference_module
    
    # === 向后兼容的接口 ===
    
    def train_sft(self, dataset, adapter_path: str, config=None):
        """
        SFT训练接口（向后兼容）。

        Args:
            dataset: 训练数据集
            adapter_path (str): LoRA adapter权重保存路径
            config: 训练配置对象
            
        Returns:
            训练结果对象
        """
        return self.sft_module.train(dataset, adapter_path, config)
    
    def train_dpo(self, dataset, dpo_adapter_path: str, config=None):
        """
        DPO训练接口（向后兼容）。

        Args:
            dataset: DPO偏好数据集
            dpo_adapter_path (str): DPO adapter权重保存路径
            config: DPO训练配置对象
            
        Returns:
            训练结果对象
        """
        return self.dpo_module.train(dataset, dpo_adapter_path, config)
    
    def generate_completion(self, prompt: str) -> str:
        """
        生成完成文本接口（向后兼容）。

        Args:
            prompt (str): 输入prompt
            
        Returns:
            str: 生成的文本
        """
        return self.inference_module.generate_completion(prompt)
    
    def predict(self, instruction: str) -> Dict[str, str]:
        """
        预测接口，返回thought和action。

        Args:
            instruction (str): 任务指令
            
        Returns:
            dict: 包含thought和action的字典
        """
        return self.inference_module.predict(instruction)
    
    # === 类方法（向后兼容） ===
    
    @classmethod
    def from_sft_adapter(
        cls,
        base_model_name: str,
        adapter_path: str,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ):
        """
        从SFT adapter加载Agent（向后兼容）。
        
        Args:
            base_model_name (str): 基础模型名称
            adapter_path (str): adapter路径
            quantization_config: 量化配置
            
        Returns:
            Agent: 加载了SFT adapter的Agent实例
        """
        agent = cls(base_model_name, quantization_config)
        
        # 加载adapter到推理模块
        agent.inference_module.load_adapter(adapter_path)
        
        return agent
    
    # === 新增的协调功能 ===
    
    def load_adapter_to_module(self, adapter_path: str, module_type: str = "inference"):
        """
        将adapter加载到指定模块。
        
        Args:
            adapter_path (str): adapter路径
            module_type (str): 目标模块类型 ("sft", "dpo", "inference")
        """
        if module_type == "sft":
            self.sft_module.load_adapter(adapter_path)
        elif module_type == "dpo":
            self.dpo_module.load_adapter(adapter_path)
        elif module_type == "inference":
            self.inference_module.load_adapter(adapter_path)
        else:
            raise ValueError(f"不支持的模块类型: {module_type}")
    
    def transfer_model_between_modules(self, from_module: str, to_module: str):
        """
        在模块间转移模型状态。
        
        Args:
            from_module (str): 源模块
            to_module (str): 目标模块
        """
        # 这是一个高级功能，需要更复杂的实现
        # 目前提供基础框架
        print(f"转移模型状态: {from_module} -> {to_module}")
        
        # 获取源模块的模型
        if from_module == "sft" and self._sft_module is not None:
            source_model = self._sft_module.model
        elif from_module == "dpo" and self._dpo_module is not None:
            source_model = self._dpo_module.model
        elif from_module == "inference" and self._inference_module is not None:
            source_model = self._inference_module.model
        else:
            raise ValueError(f"源模块未初始化或不存在: {from_module}")
        
        # 这里可以实现模型状态的复制逻辑
        # 当前只是占位符
        print("模型状态转移完成")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        获取Agent的整体状态。
        
        Returns:
            dict: Agent状态信息
        """
        status = {
            "model_name": self.model_name,
            "current_module": self._current_module,
            "module_history": self._module_history.copy(),
            "initialized_modules": [],
            "quantization_enabled": self.quantization_config is not None,
        }
        
        if self._sft_module is not None:
            status["initialized_modules"].append("sft")
            status["sft_metrics"] = self._sft_module.get_trainer_metrics()
        
        if self._dpo_module is not None:
            status["initialized_modules"].append("dpo")
            status["dpo_metrics"] = self._dpo_module.get_trainer_metrics()
        
        if self._inference_module is not None:
            status["initialized_modules"].append("inference")
            status["inference_stats"] = self._inference_module.get_generation_stats()
        
        return status
    
    def cleanup_all_modules(self):
        """清理所有模块和释放资源"""
        if self._sft_module is not None:
            self._sft_module.cleanup()
            self._sft_module = None
        
        if self._dpo_module is not None:
            self._dpo_module.cleanup()
            self._dpo_module = None
        
        if self._inference_module is not None:
            self._inference_module.clear_cache()
            self._inference_module = None
        
        self._current_module = None
        print("所有模块已清理完毕")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源清理"""
        self.cleanup_all_modules()
    
    def __repr__(self):
        return f"Agent(model='{self.model_name}', current_module='{self._current_module}')" 