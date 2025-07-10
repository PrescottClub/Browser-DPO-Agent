# src/agent/sft_module.py

from typing import Optional
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer

from .base_model import BaseModel


class SFTModule(BaseModel):
    """
    SFT训练模块，专门负责监督微调（Supervised Fine-Tuning）的逻辑。
    
    继承自BaseModel，专注于SFT训练的具体实现，
    遵循单一职责原则。
    """

    def __init__(
        self, 
        model_name: str, 
        quantization_config=None,
        device_map: str = "auto"
    ):
        """
        初始化SFT训练模块。

        Args:
            model_name (str): Hugging Face上的模型名称
            quantization_config: 量化配置
            device_map (str): 设备映射策略
        """
        super().__init__(model_name, quantization_config, device_map)
        self.current_trainer = None
    
    def create_lora_config(
        self, 
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: str = "all-linear"
    ) -> LoraConfig:
        """
        创建LoRA配置。
        
        Args:
            r (int): LoRA rank
            lora_alpha (int): LoRA alpha参数
            lora_dropout (float): LoRA dropout率
            target_modules (str): 目标模块
            
        Returns:
            LoraConfig: LoRA配置对象
        """
        return LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
    
    def create_training_args(
        self,
        output_dir: str,
        learning_rate: float = 2e-4,
        max_steps: int = 100,
        batch_size: int = 1,
        grad_accumulation_steps: int = 4,
        logging_steps: int = 10,
        save_strategy: str = "steps",
        save_steps: Optional[int] = None
    ) -> TrainingArguments:
        """
        创建训练参数配置。
        
        Args:
            output_dir (str): 输出目录
            learning_rate (float): 学习率
            max_steps (int): 最大训练步数
            batch_size (int): 批次大小
            grad_accumulation_steps (int): 梯度累积步数
            logging_steps (int): 日志记录步数
            save_strategy (str): 保存策略
            save_steps (int, optional): 保存步数间隔
            
        Returns:
            TrainingArguments: 训练参数对象
        """
        if save_steps is None:
            save_steps = max_steps // 2
            
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            max_steps=max_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            report_to="none",
        )
    
    def train(
        self, 
        dataset, 
        adapter_path: str, 
        config=None,
        lora_config: Optional[LoraConfig] = None
    ):
        """
        执行SFT训练。

        Args:
            dataset: 训练数据集
            adapter_path (str): LoRA adapter权重保存路径
            config: 训练配置对象（可选）
            lora_config (LoraConfig, optional): LoRA配置（可选）
            
        Returns:
            训练结果对象
        """
        # 解析配置参数
        if config:
            learning_rate = getattr(config, 'learning_rate', 2e-4)
            max_steps = getattr(config, 'max_steps', 100)
            batch_size = getattr(config, 'batch_size', 1)
            grad_accumulation_steps = getattr(config, 'grad_accumulation_steps', 4)
        else:
            # 默认值
            learning_rate = 2e-4
            max_steps = 100
            batch_size = 1
            grad_accumulation_steps = 4
        
        # 创建LoRA配置
        if lora_config is None:
            lora_config = self.create_lora_config()
        
        # 创建训练参数
        training_args = self.create_training_args(
            output_dir=adapter_path,
            learning_rate=learning_rate,
            max_steps=max_steps,
            batch_size=batch_size,
            grad_accumulation_steps=grad_accumulation_steps
        )
        
        # 创建训练器
        self.current_trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            peft_config=lora_config,
        )
        
        print("--- 开始SFT训练 ---")
        train_result = self.current_trainer.train()
        print(f"--- SFT训练完成，Adapter已保存至 {adapter_path} ---")
        
        return train_result
    
    def get_trainer_metrics(self) -> dict:
        """
        获取训练器的指标信息。
        
        Returns:
            dict: 训练指标字典
        """
        if self.current_trainer is None:
            return {"status": "no_trainer"}
        
        # 获取训练状态
        state = self.current_trainer.state
        return {
            "status": "active" if self.current_trainer.is_in_train else "completed",
            "global_step": state.global_step,
            "epoch": state.epoch,
            "max_steps": state.max_steps,
            "log_history": state.log_history[-5:] if state.log_history else []  # 最近5条日志
        }
    
    def cleanup(self):
        """清理训练器和释放资源"""
        if self.current_trainer is not None:
            del self.current_trainer
            self.current_trainer = None
        self.clear_cache() 