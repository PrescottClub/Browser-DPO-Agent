# 🌐 Language Switching Guide | 语言切换指南

## 🇨🇳 中文说明

### 📖 关于双语支持
本项目支持中英文双语README，GitHub主页默认显示中文版本。您可以随时切换到英文版本。

### 🔄 如何切换语言

#### 方法1: 使用Python脚本（推荐）
```bash
# 克隆项目后，在项目根目录执行：

# 切换到英文版本
python scripts/switch_language.py --lang en

# 切换回中文版本
python scripts/switch_language.py --lang zh

# 查看当前语言状态
python scripts/switch_language.py --status
```

#### 方法2: 使用快捷脚本（Windows用户）
- 双击 `switch_to_english.bat` 切换到英文
- 双击 `switch_to_chinese.bat` 切换到中文

#### 方法3: 直接查看对应文件
- `README.md` - 当前显示的版本
- `README_ZH.md` - 中文版本
- `README_EN.md` - 英文版本

### 📁 文件说明
- 主页显示的是 `README.md`，默认为中文版本
- 切换语言实际上是替换 `README.md` 的内容
- 原始版本会自动备份，确保数据安全

---

## 🇺🇸 English Instructions

### 📖 About Bilingual Support
This project supports bilingual README in Chinese and English. The GitHub homepage displays the Chinese version by default. You can switch to English version at any time.

### 🔄 How to Switch Language

#### Method 1: Using Python Script (Recommended)
```bash
# After cloning the project, execute in the project root directory:

# Switch to English version
python scripts/switch_language.py --lang en

# Switch back to Chinese version
python scripts/switch_language.py --lang zh

# Check current language status
python scripts/switch_language.py --status
```

#### Method 2: Using Shortcut Scripts (Windows Users)
- Double-click `switch_to_english.bat` to switch to English
- Double-click `switch_to_chinese.bat` to switch to Chinese

#### Method 3: View Corresponding Files Directly
- `README.md` - Currently displayed version
- `README_ZH.md` - Chinese version
- `README_EN.md` - English version

### 📁 File Description
- The homepage displays `README.md`, which defaults to Chinese version
- Language switching actually replaces the content of `README.md`
- Original versions are automatically backed up to ensure data safety

---

## 🛠️ For Developers | 开发者说明

### 🔧 How the Language Switching Works
The language switching system works by:
1. Detecting the current language of `README.md`
2. Backing up the current version to the appropriate language file
3. Copying the target language version to `README.md`
4. Providing status feedback to the user

### 📝 Adding New Languages
To add support for additional languages:
1. Create a new README file (e.g., `README_FR.md` for French)
2. Update the `switch_language.py` script to include the new language
3. Add language detection logic for the new language

### 🔍 Troubleshooting
If language switching doesn't work:
1. Ensure you're in the project root directory
2. Check that Python is installed and accessible
3. Verify that all README files exist
4. Run `python scripts/switch_language.py --status` to diagnose issues

---

## 📞 Support | 技术支持

If you encounter any issues with language switching:
- 🐛 Report bugs in GitHub Issues
- 💬 Ask questions in Discussions
- 📧 Contact maintainers

如果您在语言切换过程中遇到问题：
- 🐛 在GitHub Issues中报告bug
- 💬 在Discussions中提问
- 📧 联系维护者
