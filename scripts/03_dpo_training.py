# scripts/03_dpo_training.py

import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from src.agent.model import AgentModel

# --- 配置 ---
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
SFT_ADAPTER_PATH = "./models/sft_v1_adapter/checkpoint-100" # 从SFT训练好的模型开始
PREFERENCE_DATA_PATH = "data/preferences/dpo_v1_data.jsonl"
DPO_ADAPTER_PATH = "./models/dpo_v1_adapter" # DPO训练结果的保存路径

def main():
    print("--- 启动DPO强化训练流程 ---")

    # 1. 加载SFT Agent作为DPO训练的起点
    # 这是关键一步，DPO是在SFT的基础上进行强化
    agent = AgentModel.from_sft_adapter(base_model_name=MODEL_NAME, adapter_path=SFT_ADAPTER_PATH)

    # 2. 加载偏好数据集
    print(f"从 {PREFERENCE_DATA_PATH} 加载偏好数据集...")
    preference_dataset = load_dataset("json", data_files=PREFERENCE_DATA_PATH, split="train")
    print("数据集加载完毕。")

    # 3. 开始DPO训练
    agent.train_dpo(dataset=preference_dataset, dpo_adapter_path=DPO_ADAPTER_PATH)

    print("--- DPO训练流程全部结束 ---")

if __name__ == "__main__":
    main()
