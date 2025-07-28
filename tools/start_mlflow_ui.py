#!/usr/bin/env python3
"""
启动MLflow UI界面
"""

import os
import subprocess
import sys


def start_mlflow_ui():
    """启动MLflow UI"""
    print("[STARTING] MLflow UI...")
    print("[INFO] You can view experiment results in browser")
    print("[URL] Default address: http://localhost:5000")
    print("[STOP] Press Ctrl+C to stop service")
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
        print("\n[SUCCESS] MLflow UI stopped")
    except Exception as e:
        print(f"[ERROR] Failed to start MLflow UI: {e}")


if __name__ == "__main__":
    start_mlflow_ui()
