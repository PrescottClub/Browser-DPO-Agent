# scripts/03_dpo_training.py

import argparse
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from datasets import load_dataset

from src.agent.model import AgentModel
from src.utils.config import load_config


def main():
    """
    DPO训练主流程，并使用MLflow进行追踪。
    """
    # 0. 解析命令行参数
    parser = argparse.ArgumentParser(description="DPO训练脚本")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config_path)

    # 2. 启动MLflow Run
    # 我们为DPO训练设置一个专门的实验名称
    mlflow.set_experiment("DPO-Driver DPO")
    with mlflow.start_run() as run:
        print(f"--- 启动DPO强化训练流程 (MLflow Run ID: {run.info.run_id}) ---")

        # 3. 记录所有相关参数
        print("记录实验参数到MLflow...")
        mlflow.log_params(config.model.model_dump())
        mlflow.log_params(config.training.dpo.model_dump())
        mlflow.log_param("preference_data_path", str(config.paths.preference_data))

        # 4. 从配置中获取参数
        model_name = config.model.base_model_name
        sft_adapter_path = (
            config.paths.sft_adapter_path + "/checkpoint-100"
        )  # 从SFT训练好的模型开始
        preference_data_path = config.paths.preference_data
        dpo_adapter_path = config.paths.dpo_adapter_path

        # 5. 加载SFT Agent作为DPO训练的起点
        # 这是关键一步，DPO是在SFT的基础上进行强化
        agent = AgentModel.from_sft_adapter(
            base_model_name=model_name, adapter_path=sft_adapter_path
        )

        # 6. 加载偏好数据集
        print(f"从 {preference_data_path} 加载偏好数据集...")
        preference_dataset = load_dataset(
            "json", data_files=preference_data_path, split="train"
        )
        print("数据集加载完毕。")

        # 7. 开始DPO训练 (将训练参数也从config传入)
        train_results = agent.train_dpo(
            dataset=preference_dataset,
            dpo_adapter_path=dpo_adapter_path,
            training_args_config=config.training.dpo,
        )

        # 8. 记录训练指标 (例如，最终的训练损失)
        if (
            train_results
            and hasattr(train_results, "metrics")
            and "train_loss" in train_results.metrics
        ):
            mlflow.log_metric("final_train_loss", train_results.metrics["train_loss"])

        # 9. 将训练好的adapter作为artifact记录
        print(f"记录模型adapter到MLflow...")
        mlflow.log_artifact(local_path=dpo_adapter_path, artifact_path="dpo_adapter")

        print("--- DPO训练流程全部结束 ---")


if __name__ == "__main__":
    main()
