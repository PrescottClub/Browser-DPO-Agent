@echo off
echo 🔄 Switching README to English...
cd /d "%~dp0.."
python scripts/switch_language.py --lang en
pause
