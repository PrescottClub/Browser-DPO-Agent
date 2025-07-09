# src/agent/model.py

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from typing import Dict, Optional

class AgentModel:
    """
    封装了语言模型的加载、SFT训练和推理功能。
    """
    def __init__(self, model_name: str, quantization_config: Optional[BitsAndBytesConfig] = None):
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("模型和分词器加载完毕。")

    def train_sft(self, dataset, adapter_path: str):
        """
        使用SFTTrainer对模型进行监督微调。

        Args:
            dataset (Dataset): 用于训练的数据集。
            adapter_path (str): LoRA adapter权重保存的路径。
        """
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
            per_device_train_batch_size=1, # 保持为1以节约显存
            gradient_accumulation_steps=4, # 通过累积梯度模拟更大的batch size
            learning_rate=2e-4,
            logging_steps=10,
            max_steps=100, # 初始训练100步观察效果
            save_strategy="steps",
            save_steps=50,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
        )

        print("--- 开始SFT训练 ---")
        trainer.train()
        print(f"--- SFT训练完成，Adapter已保存至 {adapter_path} ---") 