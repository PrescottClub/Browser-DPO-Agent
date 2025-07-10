# src/utils/mlflow_logger.py

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import mlflow
import platform
import sys


class MLflowLogger:
    """
    深度集成MLflow的实验追踪器，自动捕获完整的实验环境信息。
    
    使用上下文管理器模式，确保不侵入主逻辑代码。
    """
    
    def __init__(self, experiment_name: str, config_path: str = "config.yaml"):
        """
        初始化MLflow logger。
        
        Args:
            experiment_name (str): MLflow实验名称
            config_path (str): 配置文件路径
        """
        self.experiment_name = experiment_name
        self.config_path = config_path
        self.run = None
        self.run_id = None
        
    def __enter__(self):
        """进入上下文管理器，启动MLflow run并记录环境信息。"""
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run()
        self.run_id = self.run.info.run_id
        
        print(f"[启动] 深度追踪的MLflow实验 (Run ID: {self.run_id})")
        
        # 自动记录所有环境信息
        self._log_git_info()
        self._log_system_info()
        self._log_config_file()
        self._log_dependencies()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器，结束MLflow run。"""
        if exc_type is not None:
            # 如果有异常，记录异常信息
            mlflow.log_param("execution_status", "failed")
            mlflow.log_param("error_type", str(exc_type.__name__))
            mlflow.log_param("error_message", str(exc_val))
            print(f"[失败] 实验执行失败: {exc_val}")
        else:
            mlflow.log_param("execution_status", "success")
            print("[成功] 实验执行成功")
        
        mlflow.end_run()
        print(f"[完成] MLflow实验追踪已完成 (Run ID: {self.run_id})")
    
    def _log_git_info(self):
        """捕获并记录Git信息。"""
        try:
            # 获取当前Git commit hash
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            
            # 获取当前分支
            branch_name = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            
            # 检查是否有未提交的更改
            try:
                subprocess.check_output(
                    ["git", "diff", "--quiet"],
                    stderr=subprocess.DEVNULL
                )
                has_uncommitted_changes = False
            except subprocess.CalledProcessError:
                has_uncommitted_changes = True
            
            # 记录Git信息
            mlflow.log_param("git_commit_hash", commit_hash)
            mlflow.log_param("git_branch", branch_name)
            mlflow.log_param("git_has_uncommitted_changes", has_uncommitted_changes)
            
            print(f"[Git] Git信息已记录: {branch_name}@{commit_hash[:8]}")
            
            if has_uncommitted_changes:
                print("[警告] 当前有未提交的更改，建议在运行实验前提交代码")
                
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # Git不可用或不在Git仓库中
            mlflow.log_param("git_status", "not_available")
            print(f"[警告] Git信息不可用: {e}")
    
    def _log_system_info(self):
        """记录系统环境信息。"""
        mlflow.log_param("python_version", sys.version.split()[0])
        mlflow.log_param("platform", platform.platform())
        mlflow.log_param("cpu_count", os.cpu_count())
        
        # 如果可用，记录GPU信息
        try:
            import torch
            if torch.cuda.is_available():
                mlflow.log_param("cuda_available", True)
                mlflow.log_param("cuda_device_count", torch.cuda.device_count())
                mlflow.log_param("cuda_device_name", torch.cuda.get_device_name(0))
            else:
                mlflow.log_param("cuda_available", False)
        except ImportError:
            mlflow.log_param("torch_available", False)
    
    def _log_config_file(self):
        """保存配置文件作为MLflow artifact。"""
        if Path(self.config_path).exists():
            mlflow.log_artifact(self.config_path, artifact_path="config")
            print(f"[配置] 配置文件已保存: {self.config_path}")
        else:
            print(f"[警告] 配置文件不存在: {self.config_path}")
    
    def _log_dependencies(self):
        """保存依赖锁定文件作为MLflow artifact。"""
        # 保存poetry.lock（如果存在）
        poetry_lock = Path("poetry.lock")
        if poetry_lock.exists():
            mlflow.log_artifact(str(poetry_lock), artifact_path="dependencies")
            print("[依赖] Poetry依赖锁定文件已保存")
        
        # 保存pyproject.toml（如果存在）
        pyproject = Path("pyproject.toml")
        if pyproject.exists():
            mlflow.log_artifact(str(pyproject), artifact_path="dependencies")
            print("[依赖] 项目配置文件已保存")
        
        # 生成并保存pip freeze输出
        try:
            pip_freeze = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"],
                text=True
            )
            
            # 创建临时文件保存pip freeze输出
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(pip_freeze)
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, artifact_path="dependencies/pip_freeze.txt")
            os.unlink(temp_path)
            print("[依赖] Pip依赖列表已保存")
            
        except subprocess.CalledProcessError as e:
            print(f"[警告] 无法获取pip依赖信息: {e}")
    
    def log_config_params(self, config):
        """记录配置参数到MLflow。"""
        if hasattr(config, 'model'):
            mlflow.log_params(config.model.model_dump())
        if hasattr(config, 'project'):
            mlflow.log_params(config.project.model_dump())
        
        print("[参数] 配置参数已记录到MLflow")
    
    def log_training_params(self, training_config):
        """记录训练参数到MLflow。"""
        mlflow.log_params(training_config.model_dump())
        print("[参数] 训练参数已记录到MLflow")
    
    def log_stage_completion(self, stage_name: str, **kwargs):
        """记录训练阶段完成信息。"""
        mlflow.log_param(f"{stage_name}_completed", True)
        for key, value in kwargs.items():
            mlflow.log_param(f"{stage_name}_{key}", value)
        
        print(f"[完成] {stage_name}阶段完成信息已记录")
    
    def get_run_id(self) -> Optional[str]:
        """获取当前MLflow run ID。"""
        return self.run_id
    
    def get_run_based_path(self, base_path: str) -> str:
        """
        基于MLflow run ID生成唯一的路径。
        
        Args:
            base_path (str): 基础路径
            
        Returns:
            str: 包含run ID的唯一路径
        """
        return f"{base_path}/{self.run_id}" 