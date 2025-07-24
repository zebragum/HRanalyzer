@echo off
title Heart Rate Analyzer - Web App
echo.
echo =====================================
echo    Heart Rate Analyzer Web App
echo =====================================
echo.
echo Starting the application...
echo.
echo Installing/updating dependencies...
pip install -r requirements.txt
echo.
echo Starting Flask server...
echo.
echo The application will be available at:
echo http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.
python app.py
pause 