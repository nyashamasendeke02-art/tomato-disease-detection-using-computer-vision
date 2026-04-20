@echo off
cd /d "c:\Users\golden_geek\Desktop\tomato-disease-detection-using-computer-vision"

:: Create new directories
mkdir scripts
mkdir docs
mkdir environments
mkdir pipeline

:: Create placeholder files
echo # Deployment scripts and data collection tools > scripts\README.md
echo # Project documentation and API references > docs\README.md
echo # Environment configuration files > environments\README.md
echo # CI/CD pipeline definitions > pipeline\README.md
