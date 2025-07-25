# src/utils/config.py

from typing import List

import yaml
from pydantic import BaseModel, DirectoryPath, Field, FilePath


# --- Define data models for validation ---
class SFTTrainingConfig(BaseModel):
    learning_rate: float = Field(..., gt=0)
    max_steps: int = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)
    grad_accumulation_steps: int = Field(..., gt=0)


class DPOTrainingConfig(BaseModel):
    learning_rate: float = Field(..., gt=0)
    max_steps: int = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)
    grad_accumulation_steps: int = Field(..., gt=0)
    beta: float = Field(..., gt=0)


class TrainingConfig(BaseModel):
    sft: SFTTrainingConfig
    dpo: DPOTrainingConfig


class PathsConfig(BaseModel):
    sft_data: str  # Changed from FilePath to str for flexibility
    preference_data: str  # Path might not exist yet
    sft_adapter_path: str
    dpo_adapter_path: str


class ModelConfig(BaseModel):
    base_model_name: str


class EvalConfig(BaseModel):
    num_episodes_per_task: int = Field(..., gt=0)
    tasks: List[str]


class ProjectConfig(BaseModel):
    seed: int = Field(default=42, ge=0)


class AppConfig(BaseModel):
    project: ProjectConfig
    model: ModelConfig
    paths: PathsConfig
    training: TrainingConfig
    evaluation: EvalConfig


# --- Global config loader ---
_config = None


def load_config(config_path: str = "config.yaml") -> AppConfig:
    global _config
    if _config is None:
        print(f"Loading configuration from {config_path}...")
        
        try:
            # Try UTF-8 encoding first
            with open(config_path, "r", encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except UnicodeDecodeError:
            # Fallback to system default encoding
            try:
                with open(config_path, "r", encoding='gbk') as f:
                    config_data = yaml.safe_load(f)
            except UnicodeDecodeError:
                # Last resort: ignore encoding errors
                with open(config_path, "r", encoding='utf-8', errors='ignore') as f:
                    config_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")

        try:
            _config = AppConfig(**config_data)
            print("Configuration loaded and validated successfully.")
        except Exception as e:
            print(f"Error validating configuration: {e}")
            raise ValueError(f"Configuration validation failed: {e}")
    return _config


# Provide a global accessor
def get_config() -> AppConfig:
    if _config is None:
        return load_config()
    return _config
