# src/utils/checkpoint_manager.py

import json
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any


class CheckpointManager:
    """
    管理训练过程中的checkpoint路径，避免硬编码并提供覆盖保护。
    """
    
    def __init__(self, base_dir: str, enable_run_id_isolation: bool = True):
        """
        初始化checkpoint管理器。
        
        Args:
            base_dir (str): 基础目录路径
            enable_run_id_isolation (bool): 是否启用基于run_id的目录隔离
        """
        self.base_dir = Path(base_dir)
        self.metadata_file = self.base_dir / "checkpoint_metadata.json"
        self.enable_run_id_isolation = enable_run_id_isolation
        
    def get_safe_output_path(self, base_path: str, run_id: str, 
                           allow_overwrite: bool = False) -> str:
        """
        生成安全的输出路径，防止意外覆盖。
        
        Args:
            base_path (str): 基础路径
            run_id (str): MLflow run ID
            allow_overwrite (bool): 是否允许覆盖现有数据
            
        Returns:
            str: 安全的输出路径
        """
        if self.enable_run_id_isolation:
            safe_path = f"{base_path}/{run_id}"
        else:
            safe_path = base_path
        
        safe_path_obj = Path(safe_path)
        
        # 检查路径是否已存在
        if safe_path_obj.exists() and not allow_overwrite:
            # 如果不允许覆盖且路径已存在，添加后缀
            counter = 1
            while Path(f"{safe_path}_backup_{counter}").exists():
                counter += 1
            
            backup_path = f"{safe_path}_backup_{counter}"
            print(f"⚠️  输出路径已存在，使用备份路径: {backup_path}")
            return backup_path
        elif safe_path_obj.exists() and allow_overwrite:
            print(f"🔄 允许覆盖模式，将重用现有路径: {safe_path}")
            # 清空现有目录内容
            if safe_path_obj.is_dir():
                shutil.rmtree(safe_path_obj)
            safe_path_obj.mkdir(parents=True, exist_ok=True)
        
        return safe_path
        
    def save_checkpoint_info(self, stage: str, checkpoint_path: str, 
                           additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        保存checkpoint信息到metadata文件。
        
        Args:
            stage (str): 训练阶段名称 (如 'sft', 'dpo')
            checkpoint_path (str): checkpoint的完整路径
            additional_info (dict, optional): 额外信息
        """
        # 确保基础目录存在
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取现有的metadata
        metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # 更新metadata
        checkpoint_info = {
            "path": str(checkpoint_path),
            "absolute_path": str(Path(checkpoint_path).absolute()),
            "exists": Path(checkpoint_path).exists(),
            "creation_method": "run_id_isolated" if self.enable_run_id_isolation else "traditional"
        }
        
        if additional_info:
            checkpoint_info.update(additional_info)
            
        metadata[stage] = checkpoint_info
        
        # 保存metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ 已保存 {stage} 阶段的checkpoint信息到 {self.metadata_file}")
    
    def get_checkpoint_path(self, stage: str) -> str:
        """
        获取指定阶段的checkpoint路径。
        
        Args:
            stage (str): 训练阶段名称
            
        Returns:
            str: checkpoint路径
            
        Raises:
            FileNotFoundError: 如果metadata文件不存在
            KeyError: 如果指定的stage不存在
            ValueError: 如果checkpoint路径不存在
        """
        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Checkpoint metadata文件不存在: {self.metadata_file}. "
                f"请确保在此之前已经完成了相关的训练阶段。"
            )
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if stage not in metadata:
            available_stages = list(metadata.keys())
            raise KeyError(
                f"未找到 '{stage}' 阶段的checkpoint信息. "
                f"可用的阶段: {available_stages}"
            )
        
        checkpoint_info = metadata[stage]
        checkpoint_path = checkpoint_info["path"]
        
        # 验证checkpoint路径是否存在
        if not Path(checkpoint_path).exists():
            raise ValueError(
                f"Checkpoint路径不存在: {checkpoint_path}. "
                f"请检查训练是否成功完成。"
            )
        
        return checkpoint_path
    
    def get_best_checkpoint_path(self, base_adapter_path: str) -> str:
        """
        从训练输出目录中自动找到最佳的checkpoint路径。
        
        Args:
            base_adapter_path (str): 基础adapter路径
            
        Returns:
            str: 最佳checkpoint的完整路径
        """
        base_path = Path(base_adapter_path)
        
        # 查找所有checkpoint目录
        checkpoint_dirs = []
        if base_path.exists():
            for item in base_path.iterdir():
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    try:
                        step_num = int(item.name.split("-")[1])
                        checkpoint_dirs.append((step_num, item))
                    except (ValueError, IndexError):
                        continue
        
        if not checkpoint_dirs:
            # 如果没有找到checkpoint目录，返回基础路径
            return str(base_path)
        
        # 返回步数最大的checkpoint（通常是最后一个）
        checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
        best_checkpoint = checkpoint_dirs[0][1]
        
        return str(best_checkpoint)
    
    def cleanup_old_checkpoints(self, stage: str, keep_count: int = 3) -> None:
        """
        清理旧的checkpoint，只保留最新的几个。
        
        Args:
            stage (str): 训练阶段名称
            keep_count (int): 保留的checkpoint数量
        """
        try:
            checkpoint_path = self.get_checkpoint_path(stage)
            base_path = Path(checkpoint_path).parent
            
            # 查找所有相同类型的checkpoint目录
            checkpoint_dirs = []
            for item in base_path.iterdir():
                if item.is_dir() and f"_{stage}_" in item.name:
                    checkpoint_dirs.append(item)
            
            # 按修改时间排序，保留最新的
            checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 删除多余的checkpoint
            for old_checkpoint in checkpoint_dirs[keep_count:]:
                print(f"🗑️  清理旧checkpoint: {old_checkpoint}")
                shutil.rmtree(old_checkpoint)
                
        except (FileNotFoundError, KeyError, ValueError):
            print(f"⚠️  无法清理 {stage} 阶段的旧checkpoint，可能是第一次运行")
    
    def get_metadata_summary(self) -> Dict[str, Any]:
        """
        获取checkpoint元数据的摘要信息。
        
        Returns:
            dict: 元数据摘要
        """
        if not self.metadata_file.exists():
            return {"status": "no_metadata", "stages": []}
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        summary = {
            "status": "available",
            "stages": list(metadata.keys()),
            "total_checkpoints": len(metadata),
            "metadata_file": str(self.metadata_file)
        }
        
        for stage, info in metadata.items():
            summary[f"{stage}_exists"] = info.get("exists", False)
            summary[f"{stage}_method"] = info.get("creation_method", "unknown")
        
        return summary 