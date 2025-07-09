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
        """
        print(f"正在加载模型: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("模型和分词器加载完毕。")

    def train_sft(self, dataset, adapter_path: str, config=None):
        """
        使用SFTTrainer对模型进行监督微调。

        Args:
            dataset (Dataset): 用于训练的数据集。
            adapter_path (str): LoRA adapter权重保存的路径。
            config (dict, optional): 训练配置。如果None，使用默认值。
        """
        # 从配置中获取参数，如果没有配置则使用默认值
        if config:
            learning_rate = config.get('training.sft.learning_rate', 2e-4)
            max_steps = config.get('training.sft.max_steps', 100)
            batch_size = config.get('training.sft.batch_size', 1)
            grad_accumulation_steps = config.get('training.sft.grad_accumulation_steps', 4)
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
        加载基础模型，并应用SFT阶段训练好的LoRA adapter权重。
        """
        # 先用和训练时相同的配置加载基础模型
        model_instance = cls(
            model_name=base_model_name, quantization_config=quantization_config
        )

        # 加载LoRA adapter
        print(f"正在从 {adapter_path} 加载LoRA adapter...")
        model_instance.model = PeftModel.from_pretrained(
            model_instance.model, adapter_path
        )
        print("Adapter加载并合并完毕。模型已准备好进行推理。")

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

    def train_dpo(self, dataset, dpo_adapter_path: str, config=None):
        """
        使用DPOTrainer对模型进行直接偏好优化。

        Args:
            dataset (Dataset): DPO偏好数据集 (包含prompt, chosen, rejected)。
            dpo_adapter_path (str): 新的DPO LoRA adapter权重保存路径。
            config (dict, optional): 训练配置。如果None，使用默认值。
        """
        # 从配置中获取参数，如果没有配置则使用默认值
        if config:
            learning_rate = config.get('training.dpo.learning_rate', 5e-6)
            max_steps = config.get('training.dpo.max_steps', 50) 
            batch_size = config.get('training.dpo.batch_size', 1)
            grad_accumulation_steps = config.get('training.dpo.grad_accumulation_steps', 2)
            beta = config.get('training.dpo.beta', 0.1)
        else:
            # 默认值（与原来保持一致）
            learning_rate = 5e-6
            max_steps = 50
            batch_size = 1
            grad_accumulation_steps = 2
            beta = 0.1
            
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
