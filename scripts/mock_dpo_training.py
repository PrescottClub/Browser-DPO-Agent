# scripts/mock_dpo_training.py
# 模拟DPO训练完成，创建一个假的DPO adapter用于测试评估流程

import os
import shutil

def main():
    print("--- 模拟DPO训练完成 ---")
    
    # 源路径和目标路径
    sft_adapter_path = "./models/sft_v1_adapter/checkpoint-100"
    dpo_adapter_path = "./models/dpo_v1_adapter"
    
    # 检查源路径是否存在
    if not os.path.exists(sft_adapter_path):
        print(f"❌ SFT adapter路径不存在: {sft_adapter_path}")
        return
    
    # 创建目标目录
    os.makedirs(dpo_adapter_path, exist_ok=True)
    
    # 复制SFT adapter作为"DPO训练结果"
    print(f"正在复制 {sft_adapter_path} 到 {dpo_adapter_path}...")
    
    # 复制所有文件
    for item in os.listdir(sft_adapter_path):
        src = os.path.join(sft_adapter_path, item)
        dst = os.path.join(dpo_adapter_path, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"✓ 复制文件: {item}")
    
    print(f"--- 模拟DPO训练完成，结果保存至 {dpo_adapter_path} ---")
    print("🎉 现在可以运行评估脚本测试完整流程！")
    print("运行: python scripts/04_evaluate_agent.py")

if __name__ == "__main__":
    main()
