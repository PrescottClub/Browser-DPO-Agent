# src/utils/logger.py

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "dpo_driver",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    设置全局日志配置。
    
    Args:
        name (str): 日志器名称
        level (int): 日志级别
        log_file (str, optional): 日志文件路径
        format_string (str, optional): 日志格式字符串
        
    Returns:
        logging.Logger: 配置好的日志器
    """
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 防止重复添加handler
    if logger.handlers:
        return logger
    
    # 设置默认格式
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    formatter = logging.Formatter(format_string)
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出（如果指定了文件路径）
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "dpo_driver") -> logging.Logger:
    """
    获取已配置的日志器。
    
    Args:
        name (str): 日志器名称
        
    Returns:
        logging.Logger: 日志器实例
    """
    logger = logging.getLogger(name)
    
    # 如果没有配置过，使用默认配置
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


def log_error_with_context(
    logger: logging.Logger,
    error_message: str,
    context: dict = None,
    exception: Exception = None
) -> None:
    """
    记录包含上下文信息的错误。
    
    Args:
        logger: 日志器实例
        error_message: 错误信息
        context: 上下文信息字典
        exception: 异常对象
    """
    # 构建完整的错误信息
    full_message = error_message
    
    if context:
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        full_message += f" | Context: {context_str}"
    
    if exception:
        logger.error(full_message, exc_info=True)
    else:
        logger.error(full_message)


def log_action_parsing_error(
    logger: logging.Logger,
    action_str: str,
    error_reason: str,
    expected_format: str = None
) -> None:
    """
    专门记录动作解析错误的函数。
    
    Args:
        logger: 日志器实例
        action_str: 原始动作字符串
        error_reason: 错误原因
        expected_format: 期望的格式说明
    """
    context = {
        "action_str": repr(action_str),
        "error_reason": error_reason,
        "action_length": len(action_str) if action_str else 0
    }
    
    if expected_format:
        context["expected_format"] = expected_format
    
    log_error_with_context(
        logger=logger,
        error_message="Action parsing failed",
        context=context
    )


# 全局日志器实例
_global_logger = None


def init_global_logger(log_file: str = "logs/dpo_driver.log", level: int = logging.INFO) -> None:
    """
    初始化全局日志器。
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
    """
    global _global_logger
    _global_logger = setup_logger(
        name="dpo_driver",
        level=level,
        log_file=log_file
    )


def get_global_logger() -> logging.Logger:
    """
    获取全局日志器实例。
    
    Returns:
        logging.Logger: 全局日志器
    """
    global _global_logger
    if _global_logger is None:
        init_global_logger()
    return _global_logger 