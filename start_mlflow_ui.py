#!/usr/bin/env python3
"""
启动MLflow UI界面
"""

import os
import subprocess
import sys


def start_mlflow_ui():
    """启动MLflow UI"""
    print("🚀 启动MLflow UI...")
    print("📊 您可以在浏览器中查看实验结果")
    print("🌐 默认地址: http://localhost:5000")
    print("⏹️  按 Ctrl+C 停止服务")
    print("-" * 50)

    try:
        # 启动MLflow UI
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlflow",
                "ui",
                "--host",
                "0.0.0.0",
                "--port",
                "5000",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\n✅ MLflow UI 已停止")
    except Exception as e:
        print(f"❌ 启动MLflow UI失败: {e}")


if __name__ == "__main__":
    start_mlflow_ui()
