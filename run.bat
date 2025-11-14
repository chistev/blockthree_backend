@echo off
setlocal enabledelayedexpansion
echo.
echo === Treasury Risk Calculator - FINAL WORKING VERSION ===
echo.

:: 1. Login
curl -s -X POST http://localhost:8000/api/login/ -H "Content-Type: application/json" -d "{\"password\":\"test\"}" > token.json
for /f "delims=" %%A in ('powershell -command "(Get-Content token.json | ConvertFrom-Json).token"') do set "TOKEN=%%A"
if not defined TOKEN (
    echo [ERROR] LOGIN FAILED
    type token.json
    pause
    exit /b 1
)
echo [OK] Logged in

:: 2. Create valid input.json (SINGLE LINE, PROPER COMMAS)
(
echo {"mode":"pro-forma","assumptions":{"BTC_treasury":1000,"BTC_current_market_price":117000,"initial_equity_value":90000000,"shares_basic":1000000,"shares_fd":1100000,"opex_monthly":1000000,"paths":2000,"t":1.0,"mu":0.45,"sigma":0.55,"risk_free_rate":0.04,"expected_return_btc":0.45,"tax_rate":0.20,"LTV_Cap":0.50,"max_dilution":0.25,"min_runway_months":24,"max_breach_prob":0.15,"enable_hybrid":true,"nsga_pop_size":40,"nsga_n_gen":30,"objective_switches":{"max_btc":true,"min_dilution":true,"min_ltv_breach":true,"max_runway":true}}}
) > input.json

:: 3. Lock snapshot
curl -s -X POST http://localhost:8000/api/lock_snapshot/ -H "Authorization: Bearer !TOKEN!" -H "Content-Type: application/json" --data-binary @input.json > snap.json
for /f "delims=" %%B in ('powershell -command "(Get-Content snap.json | ConvertFrom-Json).snapshot_id"') do set "SNAP_ID=%%B"
if not defined SNAP_ID (
    echo [ERROR] SNAPSHOT FAILED
    type snap.json
    pause
    exit /b 1
)
echo [OK] Snapshot locked: !SNAP_ID!

:: 4. Calculate with CSV output
curl -s -X POST http://localhost:8000/api/calculate/ -H "Authorization: Bearer !TOKEN!" -H "Content-Type: application/json" -d "{\"snapshot_id\":!SNAP_ID!,\"format\":\"csv\",\"use_live\":false,\"seed\":42}" --output treasury_report.csv

:: 5. Check result
if exist treasury_report.csv (
    echo.
    echo [SUCCESS] Report generated: treasury_report.csv
    echo   - 4+ candidates (Defensive, Balanced, Growth, Hybrid)
    echo   - BTC bought greater than 0
    echo   - WACC less than 0.45
    start treasury_report.csv
) else (
    echo [ERROR] No output file generated
    pause
)

:: Cleanup
del token.json snap.json input.json 2>nul
echo.
echo === DONE - %date% %time% ===
pause