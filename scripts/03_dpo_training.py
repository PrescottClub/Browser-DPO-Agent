# scripts/03_dpo_training.py

import argparse
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from src.agent.model import AgentModel
from src.utils.config import load_config
from src.utils.reproducibility import set_seed, get_seed_from_config
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.mlflow_logger import MLflowLogger


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
    with MLflowLogger("DPO-Driver DPO", args.config_path) as mlflow_logger:
        print(f"--- 启动DPO强化训练流程 (MLflow Run ID: {mlflow_logger.get_run_id()}) ---")

        # 4. 记录配置参数
        mlflow_logger.log_config_params(config)
        mlflow_logger.log_training_params(config.training.dpo)

        # 5. 从配置中获取参数
        model_name = config.model.base_model_name
        
        # 使用checkpoint管理器获取SFT checkpoint路径
        checkpoint_manager = CheckpointManager("./models")
        try:
            sft_adapter_path = checkpoint_manager.get_checkpoint_path("sft")
            print(f"✓ 从checkpoint管理器获取SFT路径: {sft_adapter_path}")
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[警告] 无法从checkpoint管理器获取SFT路径: {e}")
            print("回退到配置文件中的路径...")
            # 回退到配置文件路径加checkpoint-100（保持向后兼容）
            sft_adapter_path = config.paths.sft_adapter_path + "/checkpoint-100"
            print(f"使用回退路径: {sft_adapter_path}")
        
        preference_data_path = config.paths.preference_data
        
        # 生成安全的DPO输出路径
        safe_dpo_adapter_path = checkpoint_manager.get_safe_output_path(
            base_path=config.paths.dpo_adapter_path,
            run_id=mlflow_logger.get_run_id(),
            allow_overwrite=args.overwrite
        )
        
        print(f"[安全] 安全DPO适配器路径: {safe_dpo_adapter_path}")

        # 6. 加载SFT Agent作为DPO训练的起点
        # 这是关键一步，DPO是在SFT的基础上进行强化
        agent = AgentModel.from_sft_adapter(
            base_model_name=model_name, adapter_path=sft_adapter_path
        )

        # 7. 加载偏好数据集
        print(f"从 {preference_data_path} 加载偏好数据集...")
        preference_dataset = load_dataset(
            "json", data_files=preference_data_path, split="train"
        )
        print("数据集加载完毕。")

        # 8. 开始DPO训练 (将训练参数也从config传入)
        train_results = agent.train_dpo(
            dataset=preference_dataset,
            dpo_adapter_path=safe_dpo_adapter_path,
            config=config.training.dpo,
        )

        # 9. 记录训练指标 (例如，最终的训练损失)
        if (
            train_results
            and hasattr(train_results, "metrics")
            and "train_loss" in train_results.metrics
        ):
            import mlflow
            mlflow.log_metric("final_train_loss", train_results.metrics["train_loss"])

        # 10. 将训练好的adapter作为artifact记录
        print(f"记录模型adapter到MLflow...")
        import mlflow
        mlflow.log_artifact(local_path=safe_dpo_adapter_path, artifact_path="dpo_adapter")

        # 11. 保存最佳DPO checkpoint路径信息
        best_dpo_checkpoint_path = checkpoint_manager.get_best_checkpoint_path(safe_dpo_adapter_path)
        checkpoint_manager.save_checkpoint_info(
            stage="dpo",
            checkpoint_path=best_dpo_checkpoint_path,
            additional_info={
                "training_steps": config.training.dpo.max_steps,
                "model_name": model_name,
                "mlflow_run_id": mlflow_logger.get_run_id(),
                "based_on_sft": sft_adapter_path,
                "safe_dpo_adapter_path": safe_dpo_adapter_path
            }
        )
        
        # 12. 记录阶段完成信息
        mlflow_logger.log_stage_completion(
            stage_name="dpo",
            checkpoint_path=best_dpo_checkpoint_path,
            adapter_path=safe_dpo_adapter_path,
            based_on_sft=sft_adapter_path
        )
        
        print(f"✓ 最佳DPO checkpoint路径已保存: {best_dpo_checkpoint_path}")

        print("--- DPO训练流程全部结束 ---")


if __name__ == "__main__":
    main()
