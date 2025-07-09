# scripts/01_sft_training.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from src.agent.model import AgentModel
# 假设我们将在src/utils/settings.py中创建配置加载器
# from src.utils.settings import load_config 

def main():
    """
    SFT训练主流程
    """
    print("--- 启动SFT基线模型训练流程 ---")
    
    # 在未来的步骤中，我们将从config.yaml加载这些值
    # 现在为了简单，我们先硬编码
    MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
    SFT_DATA_PATH = "data/sft_golden_samples.jsonl"
    ADAPTER_PATH = "./models/sft_v1_adapter"

    # 1. 加载数据集
    print(f"从 {SFT_DATA_PATH} 加载数据集...")
    sft_dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
    
    # 格式化数据集 - 将prompt和completion组合成text字段
    def format_example(example):
        text = f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['completion']}"
        return {"text": text}
    
    sft_dataset = sft_dataset.map(format_example)
    print("数据集加载完毕。")

    # 2. 初始化Agent模型
    # 注意：我们暂时不使用量化，因为压力测试表明我们有足够显存
    agent = AgentModel(model_name=MODEL_NAME, quantization_config=None)

    # 3. 开始训练
    agent.train_sft(dataset=sft_dataset, adapter_path=ADAPTER_PATH)

    print("--- SFT训练流程全部结束 ---")

if __name__ == "__main__":
    main() 