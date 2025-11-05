@echo off
REM Master script to build complete gold truth dataset (Windows version)
REM This orchestrates fetching NVD data, generating synthetics, and combining them

setlocal enabledelayedexpansion

echo =================================================
echo Building Gold Truth CVE Dataset
echo =================================================
echo.

REM Configuration
set DATA_DIR=..
set SCRIPTS_DIR=.
set OUTPUT_DIR=..\outputs

set CVE_LIST=%DATA_DIR%\cve_list_important.txt
set NVD_OUTPUT=%OUTPUT_DIR%\nvd_metadata.json
set SYNTHETIC_OUTPUT=%OUTPUT_DIR%\synthetic_cves.json
set GOLD_TRUTH_OUTPUT=%OUTPUT_DIR%\gold_truth_cves.json
set GOLD_TRUTH_CSV=%OUTPUT_DIR%\gold_truth_cves.csv

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Step 1: Fetch real CVE metadata from NVD
echo Step 1: Fetching real CVE metadata from NVD...
echo   Source: NIST National Vulnerability Database
echo   CVE list: %CVE_LIST%
echo.

if exist "%NVD_OUTPUT%" (
    echo   WARNING: NVD metadata already exists at %NVD_OUTPUT%
    set /p overwrite="  Overwrite? (y/n): "
    if /i not "!overwrite!"=="y" (
        echo   Using existing NVD metadata
        goto step1_complete
    )
)

python "%SCRIPTS_DIR%\fetch_nvd_metadata.py" --cve-list "%CVE_LIST%" --output "%NVD_OUTPUT%"
if errorlevel 1 (
    echo ERROR: Failed to fetch NVD metadata
    exit /b 1
)

:step1_complete
echo.
echo [32m^>^> Step 1 complete[0m
echo.

REM Step 2: Generate synthetic non-existent CVEs
echo Step 2: Generating synthetic non-existent CVEs...
echo   Purpose: Hallucination testing
echo   Count: 100 synthetic CVEs
echo.

python "%SCRIPTS_DIR%\generate_synthetic_cves.py" --output "%SYNTHETIC_OUTPUT%" --count 100 --seed 42
if errorlevel 1 (
    echo ERROR: Failed to generate synthetic CVEs
    exit /b 1
)

echo.
echo [32m^>^> Step 2 complete[0m
echo.

REM Step 3: Combine into gold truth dataset
echo Step 3: Creating unified gold truth dataset...
echo   Combining real and synthetic CVEs
echo.

python "%SCRIPTS_DIR%\create_gold_truth_dataset.py" --real "%NVD_OUTPUT%" --synthetic "%SYNTHETIC_OUTPUT%" --output "%GOLD_TRUTH_OUTPUT%" --csv "%GOLD_TRUTH_CSV%" --seed 42
if errorlevel 1 (
    echo ERROR: Failed to create gold truth dataset
    exit /b 1
)

echo.
echo [32m^>^> Step 3 complete[0m
echo.

REM Summary
echo =================================================
echo [32m^>^> Gold Truth Dataset Build Complete[0m
echo =================================================
echo.
echo Output files:
echo   - Real CVEs (NVD):     %NVD_OUTPUT%
echo   - Synthetic CVEs:      %SYNTHETIC_OUTPUT%
echo   - Gold Truth (JSON):   %GOLD_TRUTH_OUTPUT%
echo   - Gold Truth (CSV):    %GOLD_TRUTH_CSV%
echo.
echo Next steps:
echo   1. Review %GOLD_TRUTH_OUTPUT% for accuracy
echo   2. Use with LLM testing framework
echo   3. Update quarterly with: update_nvd_metadata.bat
echo.

endlocal
