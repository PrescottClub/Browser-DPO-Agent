# scripts/04_evaluate_agent.py

import argparse
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import mlflow
from tqdm import tqdm

from src.agent.model import AgentModel
from src.environment.interface import EnvironmentInterface
from src.utils.config import load_config


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
            prompt = obs["utterance"]
            completion = agent.generate_completion(prompt)

            last_action = None
            if "action:" in completion:
                try:
                    last_action = completion.split("action:")[-1].strip()
                except:
                    pass

            if last_action:
                _, reward, terminated, _, _ = env.step(last_action)
                if terminated and reward == 1.0:
                    task_success += 1

        env.close()
        total_success += task_success
        total_episodes += num_episodes

    return total_success / total_episodes if total_episodes > 0 else 0.0


def main():
    """
    评估主流程，并使用MLflow进行追踪。
    """
    # 0. 解析命令行参数
    parser = argparse.ArgumentParser(description="模型评估脚本")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config_path)

    # 2. 启动一个新的MLflow Run用于记录本次评估
    mlflow.set_experiment("DPO-Driver Evaluation")
    with mlflow.start_run() as run:
        print(f"--- 开始最终模型评估 (MLflow Run ID: {run.info.run_id}) ---")

        # 3. 记录相关的配置
        print("记录评估配置到MLflow...")
        mlflow.log_param("sft_adapter_path", config.paths.sft_adapter_path)
        mlflow.log_param("dpo_adapter_path", config.paths.dpo_adapter_path)
        mlflow.log_params(config.evaluation.model_dump())

        # 4. 从配置中获取参数
        model_name = config.model.base_model_name
        sft_adapter_path = config.paths.sft_adapter_path + "/checkpoint-100"
        dpo_adapter_path = config.paths.dpo_adapter_path
        eval_tasks = config.evaluation.tasks
        num_episodes_per_task = config.evaluation.num_episodes_per_task

        # 5. 评估SFT基线模型
        print("\n--- 正在评估SFT基线模型 ---")
        sft_agent = AgentModel.from_sft_adapter(model_name, sft_adapter_path)
        sft_success_rate = evaluate_agent(sft_agent, eval_tasks, num_episodes_per_task)
        del sft_agent  # 释放显存

        # 6. 评估DPO强化模型
        print("\n--- 正在评估DPO强化模型 ---")
        dpo_agent = AgentModel.from_sft_adapter(model_name, dpo_adapter_path)
        dpo_success_rate = evaluate_agent(dpo_agent, eval_tasks, num_episodes_per_task)
        del dpo_agent  # 释放显存

        # 7. 计算改进幅度
        improvement = dpo_success_rate - sft_success_rate

        # 8. 记录核心性能指标
        print("记录核心性能指标到MLflow...")
        mlflow.log_metric("sft_success_rate", sft_success_rate)
        mlflow.log_metric("dpo_success_rate", dpo_success_rate)
        mlflow.log_metric("absolute_improvement", improvement)

        # 9. 打印最终结果报告
        report_text = f"""
==================================================
                     最终评估报告
==================================================
SFT 基线模型平均成功率: {sft_success_rate:.2%}
DPO 强化模型平均成功率: {dpo_success_rate:.2%}
--------------------------------------------------
绝对成功率提升: {improvement:+.2%}
==================================================
"""
        print(report_text)

        # 10. 将文本报告作为artifact保存
        with open("evaluation_summary.txt", "w") as f:
            f.write(report_text)
        mlflow.log_artifact("evaluation_summary.txt")

        # 根据PRD的目标进行判定
        if improvement >= 0.20:
            print("\n🎉 结论：成功！DPO显著提升了Agent性能，已达成项目核心目标！")
        else:
            print("\n⚠️ 结论：结果未达预期。DPO带来的提升有限，需要进一步分析。")


if __name__ == "__main__":
    main()
