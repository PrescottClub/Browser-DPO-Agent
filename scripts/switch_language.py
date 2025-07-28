#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Language Switcher for Browser-DPO-Agent README
语言切换器 - 用于Browser-DPO-Agent README文档

This script helps switch between Chinese and English versions of README.
此脚本帮助在中英文版本的README文档之间切换。

Usage / 使用方法:
    python scripts/switch_language.py --lang zh    # Switch to Chinese / 切换到中文
    python scripts/switch_language.py --lang en    # Switch to English / 切换到英文
    python scripts/switch_language.py --status     # Check current language / 检查当前语言
"""

import argparse
import os
import shutil
from pathlib import Path


class LanguageSwitcher:
    """README语言切换器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.readme_path = self.project_root / "README.md"
        self.readme_zh_path = self.project_root / "README_ZH.md"
        self.readme_en_path = self.project_root / "README_EN.md"
        
        # 语言标识符
        self.zh_identifier = "[🇨🇳 中文](README.md)"
        self.en_identifier = "[🇺🇸 English](README_EN.md)"
    
    def detect_current_language(self):
        """检测当前README的语言"""
        if not self.readme_path.exists():
            return "unknown"
        
        try:
            with open(self.readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查语言标识符
            if "🇨🇳 中文" in content and "Browser-DPO-Agent" in content:
                # 进一步检查是否包含中文内容
                if "生产级浏览器自动化智能体" in content:
                    return "zh"
                elif "Production-Grade Browser Automation Agent" in content:
                    return "en"
            
            return "unknown"
        except Exception as e:
            print(f"❌ 检测语言时出错: {e}")
            return "error"
    
    def backup_current_readme(self):
        """备份当前README为对应语言版本"""
        current_lang = self.detect_current_language()
        
        if current_lang == "zh":
            # 备份为中文版本
            if not self.readme_zh_path.exists():
                shutil.copy2(self.readme_path, self.readme_zh_path)
                print(f"✅ 已备份中文版本到: {self.readme_zh_path}")
        elif current_lang == "en":
            # 备份为英文版本
            if not self.readme_en_path.exists():
                shutil.copy2(self.readme_path, self.readme_en_path)
                print(f"✅ 已备份英文版本到: {self.readme_en_path}")
    
    def switch_to_chinese(self):
        """切换到中文版本"""
        # 首先备份当前版本
        self.backup_current_readme()
        
        # 检查中文版本是否存在
        if self.readme_zh_path.exists():
            # 使用已存在的中文版本
            shutil.copy2(self.readme_zh_path, self.readme_path)
            print("✅ 已切换到中文版本 (使用已存在的README_ZH.md)")
        else:
            # 使用当前README.md (假设它是中文版本)
            current_lang = self.detect_current_language()
            if current_lang == "zh":
                print("✅ 当前已经是中文版本")
            else:
                print("❌ 未找到中文版本文件 (README_ZH.md)")
                return False
        
        return True
    
    def switch_to_english(self):
        """切换到英文版本"""
        # 首先备份当前版本
        self.backup_current_readme()
        
        # 检查英文版本是否存在
        if self.readme_en_path.exists():
            # 使用英文版本
            shutil.copy2(self.readme_en_path, self.readme_path)
            print("✅ 已切换到英文版本 (使用README_EN.md)")
        else:
            print("❌ 未找到英文版本文件 (README_EN.md)")
            return False
        
        return True
    
    def show_status(self):
        """显示当前状态"""
        current_lang = self.detect_current_language()
        
        print("📋 Browser-DPO-Agent README 语言状态")
        print("=" * 50)
        
        # 当前语言
        if current_lang == "zh":
            print("🇨🇳 当前语言: 中文")
        elif current_lang == "en":
            print("🇺🇸 当前语言: English")
        else:
            print("❓ 当前语言: 未知")
        
        # 可用版本
        print("\n📁 可用版本:")
        print(f"   README.md: {'✅ 存在' if self.readme_path.exists() else '❌ 不存在'}")
        print(f"   README_ZH.md: {'✅ 存在' if self.readme_zh_path.exists() else '❌ 不存在'}")
        print(f"   README_EN.md: {'✅ 存在' if self.readme_en_path.exists() else '❌ 不存在'}")
        
        # 使用说明
        print("\n🔧 切换命令:")
        print("   切换到中文: python scripts/switch_language.py --lang zh")
        print("   切换到英文: python scripts/switch_language.py --lang en")
    
    def create_chinese_backup(self):
        """创建中文版本备份"""
        if self.readme_path.exists() and not self.readme_zh_path.exists():
            current_lang = self.detect_current_language()
            if current_lang == "zh":
                shutil.copy2(self.readme_path, self.readme_zh_path)
                print(f"✅ 已创建中文版本备份: {self.readme_zh_path}")
                return True
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Browser-DPO-Agent README Language Switcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/switch_language.py --lang zh     # Switch to Chinese
  python scripts/switch_language.py --lang en     # Switch to English  
  python scripts/switch_language.py --status      # Show current status
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--lang', 
        choices=['zh', 'en'], 
        help='Target language (zh=Chinese, en=English)'
    )
    group.add_argument(
        '--status', 
        action='store_true', 
        help='Show current language status'
    )
    
    args = parser.parse_args()
    
    switcher = LanguageSwitcher()
    
    if args.status:
        switcher.show_status()
    elif args.lang == 'zh':
        # 首先尝试创建中文备份
        switcher.create_chinese_backup()
        success = switcher.switch_to_chinese()
        if success:
            print("\n🎉 切换完成! README.md 现在是中文版本")
        else:
            print("\n❌ 切换失败，请检查文件是否存在")
    elif args.lang == 'en':
        # 首先尝试创建中文备份
        switcher.create_chinese_backup()
        success = switcher.switch_to_english()
        if success:
            print("\n🎉 Switch completed! README.md is now in English")
        else:
            print("\n❌ Switch failed, please check if files exist")


if __name__ == "__main__":
    main()
