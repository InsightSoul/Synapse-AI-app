@echo off
setlocal
cd /d "%~dp0"

echo Starting Synapse AI...

if not exist ".venv\Scripts\python.exe" (
    echo Creating a private Python environment...
    py -m venv .venv
    if errorlevel 1 (
        echo Could not create the environment. Install Python 3.10 or newer and try again.
        pause
        exit /b 1
    )
    echo Installing application packages. This may take a few minutes...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Package installation failed. Check your internet connection and try again.
        pause
        exit /b 1
    )
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo Created .env. Add your own OpenAI API key there to enable AI analysis and transcription.
)

start "" "http://127.0.0.1:8000"
".venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
