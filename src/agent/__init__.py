# src/agent/__init__.py

# 新的模块化架构
from .base_model import BaseModel
from .sft_module import SFTModule
from .dpo_module import DPOModule
from .inference_module import InferenceModule
from .agent import Agent

# 向后兼容：保留原有的AgentModel导入
from .model import AgentModel

# 主要导出接口
__all__ = [
    # 新架构
    "Agent",
    "BaseModel", 
    "SFTModule",
    "DPOModule", 
    "InferenceModule",
    
    # 向后兼容
    "AgentModel",
]

# 为了向后兼容，设置Agent为默认导入
# 这样现有代码可以无缝迁移：
# from src.agent import Agent as AgentModel
