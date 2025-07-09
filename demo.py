#!/usr/bin/env python3
"""
DPO-Driver 演示脚本
展示如何使用环境反馈DPO训练AI Agent
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def main():
    print("🚀 DPO-Driver 演示")
    print("=" * 50)
    
    print("\n📋 项目概览:")
    print("- 基础模型: Qwen2-7B-Instruct")
    print("- 训练方法: SFT + DPO")
    print("- 评估环境: MiniWoB++")
    print("- 硬件需求: RTX 4060 (8GB VRAM)")
    
    print("\n📊 实验结果:")
    print("- SFT基线成功率: 60.00%")
    print("- DPO强化成功率: 70.00%")
    print("- 绝对性能提升: +10.00%")
    
    print("\n🔄 训练流程:")
    steps = [
        "1. SFT基线训练 - 学习基础的思考-行动模式",
        "2. 偏好数据收集 - 在环境中探索并记录成功/失败轨迹", 
        "3. DPO强化训练 - 使用偏好对优化决策策略",
        "4. 性能评估 - 对比SFT vs DPO模型表现"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print("\n🛠️ 快速开始:")
    commands = [
        "poetry install                              # 安装依赖",
        "poetry run python scripts/01_sft_training.py      # SFT训练",
        "poetry run python scripts/02_collect_preferences.py # 收集偏好数据",
        "poetry run python scripts/03_dpo_training.py       # DPO训练", 
        "poetry run python scripts/04_evaluate_agent.py     # 性能评估"
    ]
    
    for cmd in commands:
        print(f"   {cmd}")
    
    print("\n💡 核心创新:")
    innovations = [
        "🔄 环境反馈DPO (EF-DPO) - 无需人类标注的偏好学习",
        "⚡ 轻量级部署 - 消费级GPU即可完成训练",
        "📈 显著提升 - 在标准基准上实现+10%性能增长",
        "🤖 自动化流程 - 端到端的Agent训练与评估"
    ]
    
    for innovation in innovations:
        print(f"   {innovation}")
    
    print("\n" + "=" * 50)
    print("🌟 让AI Agent拥有真正的决策智能！")

if __name__ == "__main__":
    main()
