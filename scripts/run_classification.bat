@echo off
REM Batch script to run hyperspectral pavement classification
REM
REM This script:
REM 1. Activates the virtual environment
REM 2. Runs the main classification pipeline
REM 3. Opens the results folder when complete

echo =========================================
echo Hyperspectral Pavement Classification
echo =========================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo ERROR: Could not activate virtual environment
    echo Please ensure venv exists and is properly set up
    pause
    exit /b 1
)

echo Virtual environment activated
echo.

REM Run the main pipeline
echo Starting classification pipeline...
echo.
python main.py

REM Check if python script ran successfully
if errorlevel 1 (
    echo.
    echo ERROR: Classification pipeline failed
    echo Check the output above for error messages
    pause
    exit /b 1
)

echo.
echo =========================================
echo Classification Complete!
echo =========================================
echo.
echo Results saved to: outputs\
echo.
echo Opening results folder...
start outputs

pause

