# scripts/test_dpo_setup.py
# 测试DPO训练设置是否正确

import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试所有必要的导入是否正常"""
    print("测试导入...")
    
    try:
        from datasets import load_dataset
        print("✓ datasets 导入成功")
    except ImportError as e:
        print(f"✗ datasets 导入失败: {e}")
        return False
    
    try:
        from src.agent.model import AgentModel
        print("✓ AgentModel 导入成功")
    except ImportError as e:
        print(f"✗ AgentModel 导入失败: {e}")
        return False
    
    try:
        from src.environment.interface import EnvironmentInterface
        print("✓ EnvironmentInterface 导入成功")
    except ImportError as e:
        print(f"✗ EnvironmentInterface 导入失败: {e}")
        return False
    
    try:
        from trl import DPOTrainer
        print("✓ DPOTrainer 导入成功")
    except ImportError as e:
        print(f"✗ DPOTrainer 导入失败: {e}")
        return False
    
    return True

def test_data_files():
    """测试数据文件是否存在"""
    print("\n测试数据文件...")
    
    files_to_check = [
        "data/preferences/dpo_v1_data.jsonl",
        "models/sft_v1_adapter/checkpoint-100/adapter_config.json",
        "data/sft_golden_samples.jsonl"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {file_path} 存在")
        else:
            print(f"✗ {file_path} 不存在")
            all_exist = False
    
    return all_exist

def test_preference_data():
    """测试偏好数据格式是否正确"""
    print("\n测试偏好数据格式...")
    
    try:
        from datasets import load_dataset
        dataset = load_dataset("json", data_files="data/preferences/dpo_v1_data.jsonl", split="train")
        
        print(f"✓ 偏好数据集加载成功，包含 {len(dataset)} 条记录")
        
        # 检查必要的字段
        required_fields = ["prompt", "chosen", "rejected"]
        sample = dataset[0]
        
        for field in required_fields:
            if field in sample:
                print(f"✓ 字段 '{field}' 存在")
            else:
                print(f"✗ 字段 '{field}' 缺失")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 偏好数据测试失败: {e}")
        return False

def main():
    print("=== DPO训练设置测试 ===\n")
    
    tests = [
        ("导入测试", test_imports),
        ("数据文件测试", test_data_files),
        ("偏好数据格式测试", test_preference_data),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"\n✓ {test_name} 通过")
            else:
                print(f"\n✗ {test_name} 失败")
                all_passed = False
        except Exception as e:
            print(f"\n✗ {test_name} 出现异常: {e}")
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 所有测试通过！DPO训练设置正确。")
        print("现在可以运行:")
        print("  python scripts/03_dpo_training.py")
        print("  python scripts/04_evaluate_agent.py")
    else:
        print("⚠️ 部分测试失败，请检查上述错误。")
    print("="*50)

if __name__ == "__main__":
    main()
