# scripts/01_sft_training.py

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from src.agent.model import AgentModel
from src.utils.config import load_config
from src.utils.reproducibility import set_seed, get_seed_from_config
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.mlflow_logger import MLflowLogger


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
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="允许覆盖现有的输出目录"
    )
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config_path)

    # 2. 设置随机种子确保可复现性
    seed = get_seed_from_config(config)
    set_seed(seed)

    # 3. 使用深度MLflow集成
    with MLflowLogger("DPO-Driver SFT", args.config_path) as mlflow_logger:
        print(f"--- 启动SFT训练流程 (MLflow Run ID: {mlflow_logger.get_run_id()}) ---")

        # 4. 记录配置参数
        mlflow_logger.log_config_params(config)
        mlflow_logger.log_training_params(config.training.sft)

        # 5. 从配置中获取参数
        model_name = config.model.base_model_name
        sft_data_path = config.paths.sft_data
        
        # 6. 初始化checkpoint管理器并生成安全的输出路径
        checkpoint_manager = CheckpointManager("./models")
        safe_adapter_path = checkpoint_manager.get_safe_output_path(
            base_path=config.paths.sft_adapter_path,
            run_id=mlflow_logger.get_run_id(),
            allow_overwrite=args.overwrite
        )
        
        print(f"[安全] 安全适配器路径: {safe_adapter_path}")

        # 7. 加载数据集
        print(f"从 {sft_data_path} 加载数据集...")
        sft_dataset = load_dataset("json", data_files=str(sft_data_path), split="train")

        # 格式化数据集 - 将prompt和completion组合成text字段
        def format_example(example):
            text = f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['completion']}"
            return {"text": text}

        sft_dataset = sft_dataset.map(format_example)
        print("数据集加载完毕。")

        # 8. 初始化Agent模型
        agent = AgentModel(model_name=model_name, quantization_config=None)

        # 9. 开始训练
        # 让train_sft方法返回训练过程中的指标，以便记录
        train_results = agent.train_sft(
            dataset=sft_dataset,
            adapter_path=safe_adapter_path,
            config=config.training.sft,
        )

        # 10. 记录训练指标 (例如，最终的训练损失)
        if (
            train_results
            and hasattr(train_results, "metrics")
            and "train_loss" in train_results.metrics
        ):
            import mlflow
            mlflow.log_metric("final_train_loss", train_results.metrics["train_loss"])

        # 11. 将训练好的adapter作为artifact记录
        print(f"记录模型adapter到MLflow...")
        import mlflow
        mlflow.log_artifact(local_path=safe_adapter_path, artifact_path="sft_adapter")

        # 12. 保存最佳checkpoint路径信息
        best_checkpoint_path = checkpoint_manager.get_best_checkpoint_path(safe_adapter_path)
        checkpoint_manager.save_checkpoint_info(
            stage="sft",
            checkpoint_path=best_checkpoint_path,
            additional_info={
                "training_steps": config.training.sft.max_steps,
                "model_name": model_name,
                "mlflow_run_id": mlflow_logger.get_run_id(),
                "safe_adapter_path": safe_adapter_path
            }
        )
        
        # 13. 记录阶段完成信息
        mlflow_logger.log_stage_completion(
            stage_name="sft",
            checkpoint_path=best_checkpoint_path,
            adapter_path=safe_adapter_path
        )
        
        print(f"[完成] 最佳SFT checkpoint路径已保存: {best_checkpoint_path}")

        print("--- SFT训练流程全部结束 ---")


if __name__ == "__main__":
    main()
