# src/utils/mlflow_logger.py

import json
import os
import shutil
import subprocess
import tempfile
import time
import psutil
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
        self.start_time = None
        self.stage_start_times = {}
        
    def __enter__(self):
        """进入上下文管理器，启动MLflow run并记录环境信息。"""
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run()
        self.run_id = self.run.info.run_id
        self.start_time = time.time()
        
        print(f"[启动] 深度追踪的MLflow实验 (Run ID: {self.run_id})")
        
        # 自动记录所有环境信息
        self._log_git_info()
        self._log_system_info()
        self._log_config_file()
        self._log_dependencies()
        self._log_initial_performance()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器，结束MLflow run。"""
        # 记录总执行时间
        if self.start_time:
            total_time = time.time() - self.start_time
            mlflow.log_metric("total_execution_time_seconds", total_time)
            print(f"[时间] 总执行时间: {total_time:.2f}s")
        
        # 记录最终性能状态
        self._log_final_performance()
        
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
    
    def _log_initial_performance(self):
        """记录初始系统性能状态。"""
        try:
            # CPU信息
            mlflow.log_param("cpu_count", psutil.cpu_count())
            mlflow.log_param("cpu_count_logical", psutil.cpu_count(logical=True))
            
            # 内存信息
            memory = psutil.virtual_memory()
            mlflow.log_param("total_memory_gb", round(memory.total / (1024**3), 2))
            mlflow.log_metric("initial_memory_usage_percent", memory.percent)
            mlflow.log_metric("initial_memory_available_gb", round(memory.available / (1024**3), 2))
            
            # 磁盘信息
            disk = psutil.disk_usage('.')
            mlflow.log_param("disk_total_gb", round(disk.total / (1024**3), 2))
            mlflow.log_metric("initial_disk_usage_percent", round((disk.used / disk.total) * 100, 2))
            
            print("[性能] 初始系统性能状态已记录")
            
        except (ImportError, AttributeError) as e:
            print(f"[警告] 系统性能监控库不可用: {e}")
        except (OSError, PermissionError) as e:
            print(f"[警告] 无法访问系统性能信息: {e}")
        except Exception as e:
            print(f"[警告] 记录系统性能信息时发生未预期错误: {e}")
            print(f"[调试] 错误类型: {type(e).__name__}")
    
    def _log_final_performance(self):
        """记录最终系统性能状态。"""
        try:
            # 内存信息
            memory = psutil.virtual_memory()
            mlflow.log_metric("final_memory_usage_percent", memory.percent)
            mlflow.log_metric("final_memory_available_gb", round(memory.available / (1024**3), 2))
            
            # GPU信息（如果可用）
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        mlflow.log_metric(f"gpu_{i}_memory_allocated_gb", round(memory_allocated, 2))
                        mlflow.log_metric(f"gpu_{i}_memory_reserved_gb", round(memory_reserved, 2))
            except ImportError:
                pass
            
            print("[性能] 最终系统性能状态已记录")
            
        except (ImportError, AttributeError) as e:
            print(f"[警告] 性能监控库不可用: {e}")
        except (OSError, PermissionError) as e:
            print(f"[警告] 无法访问系统资源信息: {e}")
        except Exception as e:
            print(f"[警告] 记录最终性能信息时发生未预期错误: {e}")
            print(f"[调试] 错误类型: {type(e).__name__}")
    
    def start_stage_timer(self, stage_name: str):
        """开始记录某阶段的执行时间。"""
        self.stage_start_times[stage_name] = time.time()
        print(f"[计时] 开始记录 {stage_name} 阶段时间")
    
    def end_stage_timer(self, stage_name: str, log_metrics: Dict[str, Any] = None):
        """结束记录某阶段的执行时间并记录指标。"""
        if stage_name in self.stage_start_times:
            elapsed_time = time.time() - self.stage_start_times[stage_name]
            mlflow.log_metric(f"{stage_name}_duration_seconds", elapsed_time)
            print(f"[计时] {stage_name} 阶段完成，耗时: {elapsed_time:.2f}s")
            
            # 记录额外指标
            if log_metrics:
                for key, value in log_metrics.items():
                    mlflow.log_metric(f"{stage_name}_{key}", value)
            
            del self.stage_start_times[stage_name]
        else:
            print(f"[警告] 未找到 {stage_name} 阶段的开始时间")
    
    def log_training_progress(self, step: int, loss: float, learning_rate: float = None):
        """记录训练进度。"""
        mlflow.log_metric("loss", loss, step=step)
        if learning_rate:
            mlflow.log_metric("learning_rate", learning_rate, step=step)
        
        # 记录系统资源使用情况
        try:
            memory = psutil.virtual_memory()
            mlflow.log_metric("memory_usage_percent", memory.percent, step=step)
            
            # GPU内存使用（如果可用）
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                    mlflow.log_metric("gpu_memory_usage_gb", round(gpu_memory, 2), step=step)
            except ImportError:
                pass
                
        except Exception as e:
            print(f"[警告] 无法记录训练进度的系统信息: {e}")
    
    def log_evaluation_results(self, task_name: str, success_rate: float, avg_steps: float = None, 
                              avg_time: float = None):
        """记录评估结果。"""
        mlflow.log_metric(f"{task_name}_success_rate", success_rate)
        if avg_steps:
            mlflow.log_metric(f"{task_name}_avg_steps", avg_steps)
        if avg_time:
            mlflow.log_metric(f"{task_name}_avg_time_seconds", avg_time)
        
        print(f"[评估] {task_name} 结果已记录: 成功率={success_rate:.2%}")
    
    def log_model_size_info(self, model_path: str):
        """记录模型大小信息。"""
        try:
            if os.path.exists(model_path):
                # 计算目录总大小
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(model_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total_size += os.path.getsize(filepath)
                
                model_size_mb = total_size / (1024 * 1024)
                mlflow.log_metric("model_size_mb", round(model_size_mb, 2))
                print(f"[模型] 模型大小: {model_size_mb:.2f} MB")
                
        except Exception as e:
            print(f"[警告] 无法记录模型大小信息: {e}") 