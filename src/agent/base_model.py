# src/agent/base_model.py

from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft.peft_model import PeftModel


class BaseModel:
    """
    基础模型类，负责模型和tokenizer的加载、设备管理等通用功能。
    
    这个类遵循单一职责原则，只负责模型的基础设施管理，
    不涉及具体的训练或推理逻辑。
    """

    def __init__(
        self, 
        model_name: str, 
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device_map: str = "auto"
    ):
        """
        初始化基础模型和tokenizer。

        Args:
            model_name (str): Hugging Face上的模型名称
            quantization_config (BitsAndBytesConfig, optional): 量化配置
            device_map (str): 设备映射策略，默认为"auto"
        """
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.device_map = device_map
        
        # 初始化时不立即加载，采用延迟加载策略
        self._model = None
        self._tokenizer = None
        
    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """延迟加载tokenizer"""
        if self._tokenizer is None:
            self._load_tokenizer()
        return self._tokenizer
    
    def _load_model(self):
        """加载模型"""
        print(f"正在加载模型: {self.model_name}")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.quantization_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=self.device_map,
        )
        print("模型加载完毕。")
    
    def _load_tokenizer(self):
        """加载tokenizer"""
        print(f"正在加载tokenizer: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        
        # 设置pad_token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            
        print("Tokenizer加载完毕。")
    
    def load_adapter(self, adapter_path: str):
        """
        加载LoRA adapter权重。
        
        Args:
            adapter_path (str): adapter权重文件路径
            
        Returns:
            BaseModel: 返回自身以支持链式调用
        """
        print(f"正在从 {adapter_path} 加载LoRA adapter...")
        self._model = PeftModel.from_pretrained(self.model, adapter_path)
        print("Adapter加载完毕。")
        return self
    
    def get_device(self):
        """获取模型所在设备"""
        if self._model is not None:
            return next(self.model.parameters()).device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_model_info(self) -> dict:
        """
        获取模型基本信息。
        
        Returns:
            dict: 包含模型名称、参数量、设备等信息的字典
        """
        info = {
            "model_name": self.model_name,
            "quantization_enabled": self.quantization_config is not None,
            "device_map": self.device_map,
        }
        
        if self._model is not None:
            info["device"] = str(self.get_device())
            info["num_parameters"] = sum(p.numel() for p in self.model.parameters())
            info["trainable_parameters"] = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        
        return info
    
    def clear_cache(self):
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU缓存已清理。") 