# src/utils/reproducibility.py

import random
import numpy as np
import torch
import os


def set_seed(seed: int) -> None:
    """
    设置所有随机数生成器的种子以确保可复现性。
    
    Args:
        seed (int): 随机种子值
    """
    print(f"设置全局随机种子为: {seed}")
    
    # Python内置random模块
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保PyTorch使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量以确保其他库的确定性
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print("随机种子设置完成，实验具备可复现性")


def get_seed_from_config(config) -> int:
    """
    从配置中获取随机种子值。
    
    Args:
        config: 配置对象
        
    Returns:
        int: 随机种子值
    """
    return config.project.seed 