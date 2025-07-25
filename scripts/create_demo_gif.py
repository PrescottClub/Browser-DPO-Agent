#!/usr/bin/env python3
"""
DPO-Driver Demo GIF Creator

This script creates an animated GIF demonstration of the DPO-Driver evaluation pipeline.
It records a terminal session showing the agent in action and converts it to a polished GIF.

Prerequisites:
1. asciinema is installed (already included in dev dependencies)
2. agg (asciinema-agg) must be available:
   - For Linux/macOS: Install via `brew install agg` or your package manager
   - For Windows: Download from https://github.com/asciinema/agg/releases
     and place `agg.exe` in this `scripts/` directory

Usage: poetry run python scripts/create_demo_gif.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def check_dependencies():
    """Check if required tools are available."""
    
    # Check asciinema
    try:
        subprocess.run(["asciinema", "--version"], capture_output=True, check=True)
        print("[SUCCESS] asciinema is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] asciinema not found. Please run: poetry install")
        return False
    
    # Check agg
    agg_executable = "agg"
    if sys.platform == "win32":
        windows_agg_path = Path(__file__).parent / "agg.exe"
        if windows_agg_path.exists():
            agg_executable = str(windows_agg_path)
            print("[SUCCESS] agg.exe found in scripts/ directory")
        else:
            print("[ERROR] agg.exe not found in scripts/ directory")
            print("   Download from: https://github.com/asciinema/agg/releases")
            return False
    else:
        try:
            subprocess.run([agg_executable, "--version"], capture_output=True, check=True)
            print("[SUCCESS] agg is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("[ERROR] agg not found. Please install via your package manager")
            print("   macOS: brew install agg")
            print("   Linux: Check your distro's package manager")
            return False
    
    return True

def create_demo_script():
    """Create a simplified demo script for recording."""
    
    scripts_dir = Path(__file__).parent
    demo_script_path = scripts_dir / "demo_runner.py"
    
    demo_script_content = '''#!/usr/bin/env python3
"""
Demo script that simulates a quick DPO-Driver evaluation for recording purposes.
"""

import time
import sys

def simulate_evaluation():
    """Simulate a DPO-Driver evaluation with nice output."""
    
    print("[BOT] DPO-Driver: Environment Feedback Direct Preference Optimization")
    print("=" * 65)
    time.sleep(1)
    
    print("\\n[RUNNING] Loading DPO-enhanced agent...")
    print("   Model: Qwen2-7B-Instruct + DPO Adapter")
    print("   Environment: MiniWoB++ Web Automation")
    time.sleep(2)
    
    print("\\n[STEP] Starting evaluation on click-button-v1 task...")
    time.sleep(1)
    
    print("\\n[TARGET] Episode 1/3:")
    print("   [THOUGHT] Thought: I need to click the button labeled 'Submit'")
    print("   [ACTION] Action: click(element='submit-btn')")
    time.sleep(1.5)
    print("   [SUCCESS] Result: SUCCESS - Task completed!")
    
    print("\\n[TARGET] Episode 2/3:")
    print("   [THOUGHT] Thought: Looking for clickable button element")
    print("   [ACTION] Action: click(element='action-button')")
    time.sleep(1.5)
    print("   [SUCCESS] Result: SUCCESS - Task completed!")
    
    print("\\n[TARGET] Episode 3/3:")
    print("   [THOUGHT] Thought: Identifying the correct button to click")
    print("   [ACTION] Action: click(element='primary-btn')")
    time.sleep(1.5)
    print("   [SUCCESS] Result: SUCCESS - Task completed!")
    
    time.sleep(1)
    print("\\n[NEXT STEPS] Evaluation Results:")
    print("   Task: click-button-v1")
    print("   Success Rate: 100% (3/3)")
    print("   Avg. Time: 2.1s per episode")
    
    print("\\n[SUCCESS] DPO-Driver evaluation completed successfully!")
    print("   Environment feedback → Preference learning → Enhanced performance")
    time.sleep(2)

if __name__ == "__main__":
    simulate_evaluation()
'''
    
    with open(demo_script_path, 'w', encoding='utf-8') as f:
        f.write(demo_script_content)
    
    return demo_script_path

def create_demo_gif():
    """
    Creates an animated GIF demonstration of DPO-Driver.
    """
    
    print("[RUNNING] DPO-Driver Demo GIF Creator")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("\\n[ERROR] Missing dependencies. Please install required tools.")
        sys.exit(1)
    
    # Setup paths
    scripts_dir = Path(__file__).parent
    assets_dir = scripts_dir.parent / "assets"
    assets_dir.mkdir(exist_ok=True)
    
    demo_script_path = create_demo_script()
    output_cast_path = scripts_dir / "demo.cast"
    output_gif_path = assets_dir / "dpo_driver_demo.gif"
    
    # Determine agg executable
    agg_executable = "agg"
    if sys.platform == "win32":
        windows_agg_path = scripts_dir / "agg.exe"
        if windows_agg_path.exists():
            agg_executable = str(windows_agg_path)
    
    # Recording command
    record_command = [
        "asciinema", "rec",
        "--command", f"poetry run python {demo_script_path}",
        "--overwrite",
        "--title", "DPO-Driver Demo",
        str(output_cast_path)
    ]
    
    # Conversion command  
    convert_command = [
        agg_executable,
        str(output_cast_path),
        str(output_gif_path),
        "--speed", "1.5",
        "--theme", "monokai",
        "--font-size", "14",
        "--font-family", "JetBrains Mono,Consolas,monospace"
    ]
    
    try:
        print("\\n[RECORDING] Recording demo session...")
        print(f"   Command: {' '.join(record_command)}")
        subprocess.run(record_command, check=True)
        print(f"[SUCCESS] Recording saved to {output_cast_path}")
        
        print("\\n[CONVERTING] Converting to GIF...")
        print(f"   Output: {output_gif_path}")
        subprocess.run(convert_command, check=True)
        print(f"[SUCCESS] GIF created successfully!")
        
        # Cleanup
        demo_script_path.unlink()
        output_cast_path.unlink()
        
        print("\\n[SUCCESS] Demo GIF creation completed!")
        print(f"   [LOCATION] Location: {output_gif_path}")
        print(f"   [SIZE] File size: {output_gif_path.stat().st_size / 1024:.1f} KB")
        print("\\n   [INFO] The README.md already points to this location!")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] An error occurred: {e}")
        print("   Please check the error messages above and try again.")
        
        # Cleanup on error
        if demo_script_path.exists():
            demo_script_path.unlink()
            
    except FileNotFoundError as e:
        print(f"[ERROR] Command not found: {e}")
        print("   Please follow the installation instructions in the docstring.")

if __name__ == "__main__":
    create_demo_gif() 