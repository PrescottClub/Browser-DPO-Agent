#!/usr/bin/env python3
"""
DPO-Driver End-to-End Pipeline Runner

This script executes the complete DPO-Driver workflow:
1. Environment verification
2. SFT training  
3. Preference collection
4. DPO training
5. Performance evaluation

Usage: poetry run python scripts/run_pipeline.py
"""

import subprocess
import sys
import time
from pathlib import Path

def run_script(script_path: str, description: str) -> bool:
    """
    Run a single script and handle errors gracefully.
    
    Args:
        script_path: Path to the Python script
        description: Human-readable description of the script
        
    Returns:
        True if successful, False if failed
    """
    print(f"\n[RUNNING] {description}")
    print(f"   Running: {script_path}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ["poetry", "run", "python", script_path], 
            check=True,
            capture_output=False  # Show real-time output
        )
        
        elapsed = time.time() - start_time
        print(f"[SUCCESS] {description} completed successfully in {elapsed:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"[ERROR] {description} failed after {elapsed:.1f}s")
        print(f"   Exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n[WARNING] Pipeline interrupted by user")
        return False

def main():
    """Execute the complete DPO-Driver pipeline."""
    
    print("=" * 70)
    print("DPO-Driver: Complete Training Pipeline")
    print("   Environment Feedback Direct Preference Optimization")
    print("=" * 70)
    
    # Define the pipeline steps
    pipeline_steps = [
        ("scripts/00_verify_setup.py", "Environment Verification"),
        ("scripts/01_sft_training.py", "Supervised Fine-Tuning (SFT)"),
        ("scripts/02_collect_preferences.py", "Environment Feedback Collection"),
        ("scripts/03_dpo_training.py", "Direct Preference Optimization (DPO)"),
        ("scripts/04_evaluate_agent.py", "Performance Evaluation"),
    ]
    
    # Verify all scripts exist
    missing_scripts = []
    for script_path, _ in pipeline_steps:
        if not Path(script_path).exists():
            missing_scripts.append(script_path)
    
    if missing_scripts:
        print("[ERROR] Missing required scripts:")
        for script in missing_scripts:
            print(f"   - {script}")
        print("\nPlease ensure all pipeline scripts are present.")
        sys.exit(1)
    
    # Execute pipeline
    successful_steps = 0
    total_start_time = time.time()
    
    for i, (script_path, description) in enumerate(pipeline_steps, 1):
        print(f"\n[STEP] {i}/{len(pipeline_steps)}")
        
        if run_script(script_path, description):
            successful_steps += 1
        else:
            print(f"\n[FAILED] Pipeline failed at step {i}: {description}")
            print(f"   {successful_steps}/{len(pipeline_steps)} steps completed successfully")
            print("\n[TROUBLESHOOTING] Tips:")
            print("   - Check MLflow server is running: mlflow ui --host 127.0.0.1 --port 5000")
            print("   - Verify GPU availability: poetry run python -c 'import torch; print(torch.cuda.is_available())'")
            print("   - Review logs above for specific error messages")
            sys.exit(1)
    
    # Success!
    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 70)
    print("[SUCCESS] DPO-Driver Pipeline Completed Successfully!")
    print(f"   Total execution time: {total_elapsed/60:.1f} minutes")
    print(f"   All {len(pipeline_steps)} steps completed")
    print("=" * 70)
    print("\n[NEXT STEPS]:")
    print("   - Check MLflow UI for detailed experiment tracking")
    print("   - Review evaluation results in the final output")
    print("   - Explore model checkpoints in the models/ directory")
    print("\n[READY] Your DPO-enhanced agent is ready!")

if __name__ == "__main__":
    main() 