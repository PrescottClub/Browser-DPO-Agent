# scripts/02_collect_preferences.py

import argparse
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random

from tqdm import tqdm

from src.agent.model import AgentModel
from src.environment.interface import EnvironmentInterface
from src.utils.config import load_config
from src.utils.reproducibility import set_seed, get_seed_from_config
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.mlflow_logger import MLflowLogger


def main():
    print("--- 启动DPO偏好数据自动收流程 ---")

    # 0. 解析命令行参数
    parser = argparse.ArgumentParser(description="偏好数据收集脚本")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="允许覆盖现有的偏好数据文件"
    )
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config_path)

    # 2. 设置随机种子确保可复现性
    seed = get_seed_from_config(config)
    set_seed(seed)

    # 3. 使用深度MLflow集成
    with MLflowLogger("DPO-Driver Preferences", args.config_path) as mlflow_logger:
        print(f"--- 启动偏好数据收集流程 (MLflow Run ID: {mlflow_logger.get_run_id()}) ---")
        
        # 记录配置参数
        mlflow_logger.log_config_params(config)

        # 4. 从配置中获取参数
        model_name = config.model.base_model_name
        
        # 使用checkpoint管理器获取SFT checkpoint路径
        checkpoint_manager = CheckpointManager("./models")
        try:
            adapter_path = checkpoint_manager.get_checkpoint_path("sft")
            print(f"✓ 从checkpoint管理器获取SFT路径: {adapter_path}")
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[警告] 无法从checkpoint管理器获取SFT路径: {e}")
            print("回退到配置文件中的路径...")
            # 回退到配置文件路径加checkpoint-100（保持向后兼容）
            adapter_path = config.paths.sft_adapter_path + "/checkpoint-100"
            print(f"使用回退路径: {adapter_path}")
        
        # 生成安全的输出文件路径
        preference_dir = os.path.dirname(config.paths.preference_data)
        os.makedirs(preference_dir, exist_ok=True)
        
        safe_output_file = f"{config.paths.preference_data}_{mlflow_logger.get_run_id()}"
        
        print(f"[安全] 安全偏好数据路径: {safe_output_file}")

        # 使用配置中的评估任务和集数
        tasks_to_collect = config.evaluation.tasks
        episodes_per_task = config.evaluation.num_episodes_per_task

        # 5. 加载SFT Agent
        agent = AgentModel.from_sft_adapter(
            base_model_name=model_name, adapter_path=adapter_path
        )

        # 6. 收集原始交互数据
        interaction_log = []
        for task_id in tasks_to_collect:
            print(f"\n--- 正在为任务 {task_id} 收集数据 ---")
            env = None
            try:
                env = EnvironmentInterface(task_id=task_id)
                for i in tqdm(range(episodes_per_task), desc=f"Processing {task_id}"):
                    try:
                        obs, info = env.reset()
                        prompt = obs["utterance"]

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
                                interaction_log.append(
                                    {"prompt": prompt, "completion": completion, "reward": reward}
                                )
                        else:  # 如果没有生成有效的action
                            interaction_log.append(
                                {
                                    "prompt": prompt,
                                    "completion": completion,
                                    "reward": -1.0,  # 给予惩罚
                                }
                            )
                    
                    except Exception as episode_error:
                        print(f"[警告] 处理回合 {i} 时发生错误: {episode_error}")
                        # 记录失败的回合但继续处理
                        interaction_log.append(
                            {
                                "prompt": f"Error in episode {i}",
                                "completion": f"Episode failed: {str(episode_error)}",
                                "reward": -1.0,
                            }
                        )
                        continue

            except Exception as task_error:
                print(f"[ERROR] 处理任务 {task_id} 时发生错误: {task_error}")
                continue
            
            finally:
                # 确保环境总是被关闭，即使发生异常
                if env is not None:
                    try:
                        env.close()
                        print(f"✓ 环境 {task_id} 已安全关闭")
                    except Exception as close_error:
                        print(f"[警告] 关闭环境时发生错误: {close_error}")
                        # 不抛出异常，继续处理下一个任务

        # 7. 处理日志，构建偏好对
        print("\n--- 正在处理交互日志，构建偏好数据集 ---")
        preferences = []
        prompts = set([log["prompt"] for log in interaction_log])

        for prompt in prompts:
            prompt_logs = [log for log in interaction_log if log["prompt"] == prompt]

            success_completions = [
                log["completion"] for log in prompt_logs if log["reward"] == 1.0
            ]
            fail_completions = [
                log["completion"] for log in prompt_logs if log["reward"] <= 0.0
            ]

            # 如果同时有成功和失败的案例，则创建偏好对
            if success_completions and fail_completions:
                # 简单起见，我们随机选一个成功和失败的案例
                chosen = random.choice(success_completions)
                rejected = random.choice(fail_completions)
                preferences.append(
                    {"prompt": prompt, "chosen": chosen, "rejected": rejected}
                )

        # 7. 保存数据集
        print(f"成功构建了 {len(preferences)} 条偏好数据。")
        with open(safe_output_file, "w") as f:
            for item in preferences:
                f.write(json.dumps(item) + "\n")
        print(f"偏好数据集已保存至 {safe_output_file}")
        
        # 8. 记录完成信息
        mlflow_logger.log_stage_completion(
            stage_name="preference_collection",
            preferences_count=len(preferences),
            output_file=safe_output_file,
            adapter_path=adapter_path
        )


if __name__ == "__main__":
    main()
