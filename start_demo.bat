@echo off
setlocal

cd /d D:\ecommerce-multimodal-rag

set "PYTHONIOENCODING=utf-8"
set "TOKENIZERS_PARALLELISM=false"
set "TRANSFORMERS_VERBOSITY=error"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7860 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>nul
)

start "Ecommerce Multimodal RAG" /min cmd /c "D:\ecommerce-multimodal-rag\.venv\Scripts\python.exe D:\ecommerce-multimodal-rag\app\demo.py"

timeout /t 8 /nobreak >nul
start "" http://127.0.0.1:7860

endlocal
