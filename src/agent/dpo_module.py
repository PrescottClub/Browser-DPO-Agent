# src/agent/dpo_module.py

from typing import Optional
from trl import DPOConfig, DPOTrainer

from .base_model import BaseModel


class DPOModule(BaseModel):
    """
    DPO训练模块，专门负责直接偏好优化（Direct Preference Optimization）的逻辑。
    
    继承自BaseModel，专注于DPO训练的具体实现，
    遵循单一职责原则。
    """

    def __init__(
        self, 
        model_name: str, 
        quantization_config=None,
        device_map: str = "auto"
    ):
        """
        初始化DPO训练模块。

        Args:
            model_name (str): Hugging Face上的模型名称
            quantization_config: 量化配置
            device_map (str): 设备映射策略
        """
        super().__init__(model_name, quantization_config, device_map)
        self.current_trainer = None
    
    def create_dpo_config(
        self,
        output_dir: str,
        learning_rate: float = 5e-6,
        max_steps: int = 50,
        batch_size: int = 1,
        grad_accumulation_steps: int = 2,
        beta: float = 0.1,
        logging_steps: int = 5,
        save_strategy: str = "steps",
        save_steps: Optional[int] = None,
        max_prompt_length: int = 512,
        max_length: int = 1024,
        eval_strategy: str = "steps",
        eval_steps: Optional[int] = None,
        early_stopping_patience: int = 3
    ) -> DPOConfig:
        """
        创建DPO训练配置。
        
        Args:
            output_dir (str): 输出目录
            learning_rate (float): 学习率（DPO通常需要更小的学习率）
            max_steps (int): 最大训练步数
            batch_size (int): 批次大小
            grad_accumulation_steps (int): 梯度累积步数
            beta (float): DPO的beta参数，控制偏好强度
            logging_steps (int): 日志记录步数
            save_strategy (str): 保存策略
            save_steps (int, optional): 保存步数间隔
            max_prompt_length (int): 最大prompt长度
            max_length (int): 最大序列长度
            eval_strategy (str): 评估策略
            eval_steps (int, optional): 评估步数间隔
            early_stopping_patience (int): 早停耐心值
            
        Returns:
            DPOConfig: DPO配置对象
        """
        if save_steps is None:
            save_steps = max(max_steps // 2, 1)
        if eval_steps is None:
            eval_steps = max(max_steps // 3, 1)
            
        # Ensure eval_steps is compatible with save_steps for load_best_model_at_end
        if save_steps % eval_steps != 0:
            # Adjust eval_steps to be a divisor of save_steps
            eval_steps = save_steps
            
        return DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            max_steps=max_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            beta=beta,
            max_prompt_length=max_prompt_length,
            max_length=max_length,
        )
    
    def train(
        self, 
        dataset, 
        dpo_adapter_path: str, 
        config=None,
        ref_model=None
    ):
        """
        执行DPO训练。

        Args:
            dataset: DPO偏好数据集（包含prompt, chosen, rejected）
            dpo_adapter_path (str): 新的DPO LoRA adapter权重保存路径
            config: DPO训练配置对象（可选）
            ref_model: 参考模型（可选，如果None则TRL会自动创建）
            
        Returns:
            训练结果对象
        """
        # 强制要求配置，确保参数一致性
        if config is None:
            raise ValueError(
                "DPO training config is required. "
                "Please provide a valid DPOTrainingConfig object."
            )

        # 直接使用配置值，不提供默认值
        learning_rate = config.learning_rate
        max_steps = config.max_steps
        batch_size = config.batch_size
        grad_accumulation_steps = config.grad_accumulation_steps
        beta = config.beta
        
        # 创建DPO配置
        dpo_config = self.create_dpo_config(
            output_dir=dpo_adapter_path,
            learning_rate=learning_rate,
            max_steps=max_steps,
            batch_size=batch_size,
            grad_accumulation_steps=grad_accumulation_steps,
            beta=beta
        )
        
        # 确保模型处于训练模式
        self.model.train()
        
        # 创建DPO训练器
        self.current_trainer = DPOTrainer(
            model=self.model,  # 应该是已加载SFT adapter的PeftModel
            ref_model=ref_model,  # TRL会自动创建参考模型（如果为None）
            args=dpo_config,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        
        print("--- 开始DPO训练 ---")
        train_result = self.current_trainer.train()
        print(f"--- DPO训练完成，Adapter已保存至 {dpo_adapter_path} ---")
        
        return train_result
    
    def get_trainer_metrics(self) -> dict:
        """
        获取DPO训练器的指标信息。
        
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
            "log_history": state.log_history[-5:] if state.log_history else [],  # 最近5条日志
            "dpo_metrics": self._extract_dpo_metrics(state.log_history)
        }
    
    def _extract_dpo_metrics(self, log_history: list) -> dict:
        """
        从日志历史中提取DPO特有的指标。
        
        Args:
            log_history (list): 训练日志历史
            
        Returns:
            dict: DPO特有指标
        """
        if not log_history:
            return {}
        
        latest_log = log_history[-1] if log_history else {}
        
        dpo_metrics = {}
        for key, value in latest_log.items():
            if 'rewards' in key.lower() or 'chosen' in key.lower() or 'rejected' in key.lower():
                dpo_metrics[key] = value
        
        return dpo_metrics
    
    def get_preference_analysis(self) -> dict:
        """
        分析偏好学习的效果。
        
        Returns:
            dict: 偏好分析结果
        """
        metrics = self.get_trainer_metrics()
        
        if metrics["status"] == "no_trainer":
            return {"status": "no_data"}
        
        dpo_metrics = metrics.get("dpo_metrics", {})
        
        analysis = {
            "training_progress": f"{metrics.get('global_step', 0)}/{metrics.get('max_steps', 0)}",
            "preference_learning_active": len(dpo_metrics) > 0,
            "latest_metrics": dpo_metrics
        }
        
        return analysis
    
    def cleanup(self):
        """清理训练器和释放资源"""
        if self.current_trainer is not None:
            del self.current_trainer
            self.current_trainer = None
        self.clear_cache() 