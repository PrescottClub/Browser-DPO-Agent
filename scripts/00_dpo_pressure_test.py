# scripts/00_dpo_pressure_test.py

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

def run_dpo_pressure_test():
    """
    一个最小化的DPO训练脚本，用于测试在资源受限设备上的可行性。
    此版本不使用量化，直接测试RTX 4060的8GB显存极限。
    """
    print("--- 开始DPO极限压力测试 (无量化版本) ---")

    # 1. 模型和分词器加载
    model_name = "Qwen/Qwen2-7B-Instruct"
    print(f"加载模型: {model_name}")

    # 加载模型和分词器 (无量化)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 使用bfloat16以节省显存
        low_cpu_mem_usage=True,      # 启用低CPU内存使用
        trust_remote_code=True,
    )
    
    # 手动移动到GPU
    print("将模型移动到GPU...")
    model = model.to("cuda")
    print(f"模型现在位于: {next(model.parameters()).device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Qwen2 tokenizer没有默认的pad_token，我们将其设置为eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("模型和分词器加载完毕。")

    # 2. 创建一个虚假的最小化数据集
    print("创建虚假数据集...")
    dummy_data = {
        'prompt': ["Tell me a story about a brave knight."],
        'chosen': ["The brave knight fought the dragon and saved the princess."],
        'rejected': ["The knight was scared and ran away."]
    }
    dummy_dataset = Dataset.from_dict(dummy_data)
    print("数据集创建完毕。")
    
    # 3. LoRA配置
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    
    # 4. DPO训练参数配置
    dpo_config = DPOConfig(
        output_dir="./dpo_test_output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        logging_steps=1,
        max_steps=2, # 只训练2步，足以验证是否OOM
        save_strategy="no", # 不保存模型
        report_to="none", # 关闭wandb等报告
        beta=0.1,
        max_prompt_length=512,
        max_length=1024,
    )

    print("初始化DPOTrainer...")
    # 5. 初始化DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # TRL会自动为我们创建参考模型
        args=dpo_config,
        train_dataset=dummy_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    print("DPOTrainer初始化完毕。")

    # 6. 开始训练！
    print("\n--- 开始训练，请密切关注显存占用 ---")
    print("可以在另一个终端窗口运行 'nvidia-smi' 来监控GPU。")
    
    try:
        trainer.train()
        print("\n--- 训练步骤成功完成！---")
        print("🚀 结论：DPO训练在当前硬件和配置下是可行的！项目核心风险已解除。")
        print("💡 重要：这是在无量化的情况下成功，说明我们的硬件完全满足要求！")
    
    except torch.cuda.OutOfMemoryError:
        print("\n--- 捕获到CUDA OutOfMemoryError！---")
        print("❌ 结论：在当前配置下，Qwen2-7B模型的DPO训练显存不足。")
        print("下一步行动：我们需要启用量化或切换到更小的模型（如Qwen2-1.5B）。")
    
    except Exception as e:
        print(f"\n--- 发生未知错误：{e} ---")
        print("❌ 结论：测试未能成功，需要调试。请将完整错误信息报告。")

if __name__ == "__main__":
    run_dpo_pressure_test() 