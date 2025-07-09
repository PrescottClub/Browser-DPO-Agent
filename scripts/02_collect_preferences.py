# scripts/02_collect_preferences.py

import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from tqdm import tqdm
from src.agent.model import AgentModel
from src.environment.interface import EnvironmentInterface

# --- 配置 ---
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
ADAPTER_PATH = "./models/sft_v1_adapter/checkpoint-100" # 从阶段1训练好的SFT adapter加载
TASKS_TO_COLLECT = ['click-test', 'form-sequence']
EPISODES_PER_TASK = 3 # 每个任务尝试3次（用于快速测试）
OUTPUT_FILE = "data/preferences/dpo_v1_data.jsonl"

def main():
    print("--- 启动DPO偏好数据自动收流程 ---")

    # 1. 加载SFT Agent
    agent = AgentModel.from_sft_adapter(base_model_name=MODEL_NAME, adapter_path=ADAPTER_PATH)

    # 2. 收集原始交互数据
    interaction_log = []
    for task_id in TASKS_TO_COLLECT:
        print(f"\n--- 正在为任务 {task_id} 收集数据 ---")
        env = EnvironmentInterface(task_id=task_id)
        for i in tqdm(range(EPISODES_PER_TASK), desc=f"Processing {task_id}"):
            obs, info = env.reset()
            prompt = obs['utterance']
            
            # Agent生成完整的响应
            completion = agent.generate_completion(prompt)
            
            # 从响应中提取最后一个action
            # 这是一个简化的提取逻辑，未来可以改进
            last_action = None
            if "action:" in completion:
                try:
                    last_action = completion.split("action:")[-1].strip()
                except:
                    pass

            if last_action:
                _, reward, terminated, _, _ = env.step(last_action)
                if terminated:
                    interaction_log.append({
                        "prompt": prompt,
                        "completion": completion,
                        "reward": reward
                    })
            else: # 如果没有生成有效的action
                interaction_log.append({
                    "prompt": prompt,
                    "completion": completion,
                    "reward": -1.0 # 给予惩罚
                })

        env.close()

    # 3. 处理日志，构建偏好对
    print("\n--- 正在处理交互日志，构建偏好数据集 ---")
    preferences = []
    prompts = set([log['prompt'] for log in interaction_log])

    for prompt in prompts:
        prompt_logs = [log for log in interaction_log if log['prompt'] == prompt]
        
        success_completions = [log['completion'] for log in prompt_logs if log['reward'] == 1.0]
        fail_completions = [log['completion'] for log in prompt_logs if log['reward'] <= 0.0]

        # 如果同时有成功和失败的案例，则创建偏好对
        if success_completions and fail_completions:
            # 简单起见，我们随机选一个成功和失败的案例
            chosen = random.choice(success_completions)
            rejected = random.choice(fail_completions)
            preferences.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })

    # 4. 保存数据集
    print(f"成功构建了 {len(preferences)} 条偏好数据。")
    with open(OUTPUT_FILE, 'w') as f:
        for item in preferences:
            f.write(json.dumps(item) + "\n")
    print(f"偏好数据集已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main() 