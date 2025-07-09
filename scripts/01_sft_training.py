# scripts/01_sft_training.py

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from datasets import load_dataset

from src.agent.model import AgentModel
from src.utils.config import load_config


def main():
    """
    SFT训练主流程，并使用MLflow进行追踪。
    """
    # 0. 解析命令行参数
    parser = argparse.ArgumentParser(description="SFT训练脚本")
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
    # 我们为SFT训练设置一个专门的实验名称
    mlflow.set_experiment("DPO-Driver SFT")
    with mlflow.start_run() as run:
        print(f"--- 启动SFT训练流程 (MLflow Run ID: {run.info.run_id}) ---")

        # 3. 记录所有相关参数
        print("记录实验参数到MLflow...")
        mlflow.log_params(config.model.model_dump())
        mlflow.log_params(config.training.sft.model_dump())
        mlflow.log_param("sft_data_path", str(config.paths.sft_data))

        # 4. 从配置中获取参数
        model_name = config.model.base_model_name
        sft_data_path = config.paths.sft_data
        adapter_path = config.paths.sft_adapter_path

        # 5. 加载数据集
        print(f"从 {sft_data_path} 加载数据集...")
        sft_dataset = load_dataset("json", data_files=str(sft_data_path), split="train")

        # 格式化数据集 - 将prompt和completion组合成text字段
        def format_example(example):
            text = f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['completion']}"
            return {"text": text}

        sft_dataset = sft_dataset.map(format_example)
        print("数据集加载完毕。")

        # 6. 初始化Agent模型
        agent = AgentModel(model_name=model_name, quantization_config=None)

        # 7. 开始训练
        # 让train_sft方法返回训练过程中的指标，以便记录
        train_results = agent.train_sft(
            dataset=sft_dataset,
            adapter_path=adapter_path,
            training_args_config=config.training.sft,
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
        mlflow.log_artifact(local_path=adapter_path, artifact_path="sft_adapter")

        print("--- SFT训练流程全部结束 ---")


if __name__ == "__main__":
    main()
