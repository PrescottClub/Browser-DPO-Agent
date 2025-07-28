@echo off
echo 🔄 切换README到中文...
cd /d "%~dp0.."
python scripts/switch_language.py --lang zh
pause
