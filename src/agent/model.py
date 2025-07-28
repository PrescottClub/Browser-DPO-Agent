# src/agent/model.py

from typing import Dict, Optional

import torch
from peft import LoraConfig, get_peft_model
from peft.peft_model import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer, SFTTrainer


# 自定义异常类
class ModelLoadError(Exception):
    """模型加载异常"""
    pass


class TokenizerLoadError(Exception):
    """分词器加载异常"""
    pass


class AdapterLoadError(Exception):
    """Adapter加载异常"""
    pass


class AgentModel:
    """
    封装了语言模型的加载、SFT训练和推理功能。
    """

    def __init__(
        self, model_name: str, quantization_config: Optional[BitsAndBytesConfig] = None
    ):
        """
        初始化模型和分词器。

        Args:
            model_name (str): Hugging Face上的模型名称。
            quantization_config (BitsAndBytesConfig, optional): 量化配置。默认为None。

        Raises:
            ModelLoadError: 当模型加载失败时
            TokenizerLoadError: 当分词器加载失败时
        """
        print(f"正在加载模型: {model_name}")

        # 验证模型名称格式（在try块外，让ValueError直接抛出）
        if not model_name or not isinstance(model_name, str):
            raise ValueError(f"Invalid model name: {model_name}")

        try:
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )

            # 验证模型加载成功
            if self.model is None:
                raise ModelLoadError(f"Failed to load model: {model_name}")

        except Exception as e:
            raise ModelLoadError(f"Failed to load model '{model_name}': {e}")

        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            # 验证分词器加载成功
            if self.tokenizer is None:
                raise TokenizerLoadError(f"Failed to load tokenizer: {model_name}")

            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:
            raise TokenizerLoadError(f"Failed to load tokenizer '{model_name}': {e}")

        print("模型和分词器加载完毕。")

    def train_sft(self, dataset, adapter_path: str, config=None):
        """
        使用SFTTrainer对模型进行监督微调。

        Args:
            dataset (Dataset): 用于训练的数据集。
            adapter_path (str): LoRA adapter权重保存的路径。
            config (SFTTrainingConfig, optional): 训练配置。如果None，使用默认值。
        """
        # 从配置中获取参数，如果没有配置则使用默认值
        if config:
            learning_rate = config.learning_rate
            max_steps = config.max_steps
            batch_size = config.batch_size
            grad_accumulation_steps = config.grad_accumulation_steps
        else:
            # 默认值（与原来保持一致）
            learning_rate = 2e-4
            max_steps = 100
            batch_size = 1
            grad_accumulation_steps = 4
            
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

        training_args = TrainingArguments(
            output_dir=adapter_path,
            per_device_train_batch_size=batch_size,  # 保持为1以节约显存
            gradient_accumulation_steps=grad_accumulation_steps,  # 通过累积梯度模拟更大的batch size
            learning_rate=learning_rate,
            logging_steps=10,
            max_steps=max_steps,  # 初始训练100步观察效果
            save_strategy="steps",
            save_steps=max_steps // 2,  # 在中间保存一次
            report_to="none",
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
        )

        print("--- 开始SFT训练 ---")
        train_result = trainer.train()
        print(f"--- SFT训练完成，Adapter已保存至 {adapter_path} ---")

        return train_result

    # 新增一个类方法，用于从已保存的adapter加载模型
    @classmethod
    def from_sft_adapter(
        cls,
        base_model_name: str,
        adapter_path: str,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ):
        """
        从已保存的SFT adapter创建模型实例。

        Args:
            base_model_name (str): 基础模型名称
            adapter_path (str): adapter权重路径
            quantization_config (BitsAndBytesConfig, optional): 量化配置

        Returns:
            AgentModel: 加载了adapter的模型实例

        Raises:
            FileNotFoundError: 当adapter路径不存在时
            AdapterLoadError: 当adapter加载失败时
            ValueError: 当参数无效时
        """
        # 验证输入参数
        if not base_model_name or not isinstance(base_model_name, str):
            raise ValueError(f"Invalid base model name: {base_model_name}")

        if not adapter_path or not isinstance(adapter_path, str):
            raise ValueError(f"Invalid adapter path: {adapter_path}")

        # 验证adapter路径存在
        from pathlib import Path
        adapter_path_obj = Path(adapter_path)
        if not adapter_path_obj.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

        if not adapter_path_obj.is_dir():
            raise ValueError(f"Adapter path is not a directory: {adapter_path}")

        # 验证adapter文件完整性
        required_files = ['adapter_config.json']
        optional_files = ['adapter_model.bin', 'adapter_model.safetensors']

        # 检查必需文件
        for file in required_files:
            if not (adapter_path_obj / file).exists():
                raise AdapterLoadError(f"Missing required adapter file: {file}")

        # 检查至少有一个权重文件
        weight_files_exist = any((adapter_path_obj / file).exists() for file in optional_files)
        if not weight_files_exist:
            raise AdapterLoadError(f"No adapter weight files found. Expected one of: {optional_files}")

        try:
            # 先用和训练时相同的配置加载基础模型
            model_instance = cls(
                model_name=base_model_name, quantization_config=quantization_config
            )
        except (ModelLoadError, TokenizerLoadError) as e:
            raise AdapterLoadError(f"Failed to load base model for adapter: {e}")

        try:
            # 加载LoRA adapter
            print(f"正在从 {adapter_path} 加载LoRA adapter...")
            model_instance.model = PeftModel.from_pretrained(
                model_instance.model, adapter_path
            )

            # 验证adapter加载成功
            if not hasattr(model_instance.model, 'peft_config'):
                raise AdapterLoadError("Adapter loading failed: no peft_config found")

            print("Adapter加载并合并完毕。模型已准备好进行推理。")

        except Exception as e:
            raise AdapterLoadError(f"Failed to load adapter from {adapter_path}: {e}")

        return model_instance

    # 新增推理方法
    def generate_completion(self, prompt: str) -> str:
        """
        根据给定的prompt，生成模型的响应（包含thought和action）。

        Args:
            prompt (str): 来自环境的任务指令。

        Returns:
            str: 模型生成的完整响应文本。
        """
        # 格式化输入
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

        # 对输入进行分词
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(
            self.model.device
        )

        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # 使用贪心解码避免数值不稳定
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 解码并清理输出
        # 我们只取生成的部分，不包括输入的prompt
        response_ids = outputs[0][inputs.input_ids.shape[1] :]
        completion = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return completion.strip()

    def train_dpo(self, dataset, dpo_adapter_path: str, config):
        """
        使用DPOTrainer对模型进行直接偏好优化。

        Args:
            dataset (Dataset): DPO偏好数据集 (包含prompt, chosen, rejected)。
            dpo_adapter_path (str): 新的DPO LoRA adapter权重保存路径。
            config (DPOTrainingConfig): 训练配置，必须提供。
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
            
        # 使用DPOConfig来配置训练参数，包括beta
        dpo_config = DPOConfig(
            output_dir=dpo_adapter_path,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accumulation_steps,
            learning_rate=learning_rate,  # DPO的学习率通常需要更小
            logging_steps=5,
            max_steps=max_steps,  # DPO训练50步观察效果
            save_strategy="steps",
            save_steps=max_steps // 2,  # 在中间保存一次
            report_to="none",
            beta=beta,  # DPO的beta参数
            max_prompt_length=512,
            max_length=1024,
        )

        # 确保模型参数可以训练
        self.model.train()

        # 注意：DPOTrainer不需要peft_config，因为它会从已有的PeftModel中自动处理
        trainer = DPOTrainer(
            model=self.model,  # self.model此时应为已加载SFT adapter的PeftModel
            ref_model=None,  # TRL会自动创建参考模型
            args=dpo_config,
            train_dataset=dataset,
            processing_class=self.tokenizer,  # 使用processing_class而不是tokenizer
        )

        print("--- 开始DPO训练 ---")
        train_result = trainer.train()
        print(f"--- DPO训练完成，Adapter已保存至 {dpo_adapter_path} ---")

        return train_result
