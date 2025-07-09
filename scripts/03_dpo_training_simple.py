# scripts/03_dpo_training_simple.py
# 简化版DPO训练脚本，用于测试基本功能

import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model

# --- 配置 ---
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
PREFERENCE_DATA_PATH = "data/preferences/dpo_v1_data.jsonl"
DPO_ADAPTER_PATH = "./models/dpo_v1_adapter"

def main():
    print("--- 启动简化版DPO训练流程 ---")

    # 1. 加载基础模型和分词器
    print(f"正在加载模型: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("模型和分词器加载完毕。")

    # 2. 配置LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    
    # 应用LoRA到模型
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. 加载偏好数据集
    print(f"从 {PREFERENCE_DATA_PATH} 加载偏好数据集...")
    preference_dataset = load_dataset("json", data_files=PREFERENCE_DATA_PATH, split="train")
    print("数据集加载完毕。")

    # 4. 配置DPO训练
    dpo_config = DPOConfig(
        output_dir=DPO_ADAPTER_PATH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        logging_steps=5,
        max_steps=10,  # 先用10步测试
        save_strategy="steps",
        save_steps=5,
        report_to="none",
        beta=0.1,
        max_prompt_length=512,
        max_length=1024,
    )

    # 5. 初始化DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL会自动创建参考模型
        args=dpo_config,
        train_dataset=preference_dataset,
        processing_class=tokenizer,
    )

    # 6. 开始训练
    print("--- 开始DPO训练 ---")
    try:
        trainer.train()
        print(f"--- DPO训练完成，Adapter已保存至 {DPO_ADAPTER_PATH} ---")
        print("🎉 DPO训练成功完成！")
    except Exception as e:
        print(f"❌ DPO训练失败: {e}")
        print("这可能是由于显存不足或其他配置问题。")

    print("--- DPO训练流程结束 ---")

if __name__ == "__main__":
    main()
