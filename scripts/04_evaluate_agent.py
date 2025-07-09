# scripts/04_evaluate_agent.py

import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from tqdm import tqdm
from src.agent.model import AgentModel
from src.environment.interface import EnvironmentInterface

# --- 评估配置 ---
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
SFT_ADAPTER_PATH = "./models/sft_v1_adapter/checkpoint-100"
DPO_ADAPTER_PATH = "./models/dpo_v1_adapter"
EVAL_TASKS = ['click-button', 'enter-text']
NUM_EPISODES_PER_TASK = 10 # 每个任务评估10次以获得稳定的平均成功率

def evaluate_agent(agent: AgentModel, task_list: list, num_episodes: int) -> float:
    """
    评估一个Agent模型在指定任务列表上的平均成功率。
    """
    total_success = 0
    total_episodes = 0

    for task_id in task_list:
        env = EnvironmentInterface(task_id=task_id)
        task_success = 0
        for _ in tqdm(range(num_episodes), desc=f"Evaluating on {task_id}"):
            obs, _ = env.reset()
            prompt = obs['utterance']
            completion = agent.generate_completion(prompt)
            
            last_action = None
            if "action:" in completion:
                try:
                    last_action = completion.split("action:")[-1].strip()
                except: pass

            if last_action:
                _, reward, terminated, _, _ = env.step(last_action)
                if terminated and reward == 1.0:
                    task_success += 1
        
        env.close()
        total_success += task_success
        total_episodes += num_episodes
    
    return total_success / total_episodes if total_episodes > 0 else 0.0


def main():
    print("--- 开始最终模型评估 ---")

    # 1. 评估SFT基线模型
    print("\n--- 正在评估SFT基线模型 ---")
    sft_agent = AgentModel.from_sft_adapter(MODEL_NAME, SFT_ADAPTER_PATH)
    sft_success_rate = evaluate_agent(sft_agent, EVAL_TASKS, NUM_EPISODES_PER_TASK)
    del sft_agent # 释放显存

    # 2. 评估DPO强化模型
    print("\n--- 正在评估DPO强化模型 ---")
    dpo_agent = AgentModel.from_sft_adapter(MODEL_NAME, DPO_ADAPTER_PATH)
    dpo_success_rate = evaluate_agent(dpo_agent, EVAL_TASKS, NUM_EPISODES_PER_TASK)
    del dpo_agent # 释放显存

    # 3. 打印最终结果报告
    improvement = dpo_success_rate - sft_success_rate
    print("\n\n" + "="*50)
    print(" " * 15 + "最终评估报告")
    print("="*50)
    print(f"SFT 基线模型平均成功率: {sft_success_rate:.2%}")
    print(f"DPO 强化模型平均成功率: {dpo_success_rate:.2%}")
    print("-" * 50)
    print(f"绝对成功率提升: {improvement:+.2%}")
    print("="*50)
    
    # 根据PRD的目标进行判定
    if improvement >= 0.20:
        print("\n🎉 结论：成功！DPO显著提升了Agent性能，已达成项目核心目标！")
    else:
        print("\n⚠️ 结论：结果未达预期。DPO带来的提升有限，需要进一步分析。")


if __name__ == "__main__":
    main()
