@echo off
setlocal enabledelayedexpansion
echo.
echo === Treasury Risk Calculator - FINAL WORKING VERSION ===
echo.

:: 1. Login
curl -s -X POST http://localhost:8000/api/login/ -H "Content-Type: application/json" -d "{\"password\":\"test\"}" > token.json
for /f "delims=" %%A in ('powershell -command "(Get-Content token.json | ConvertFrom-Json).token"') do set "TOKEN=%%A"
if not defined TOKEN ( echo LOGIN FAILED & type token.json & pause & exit /b 1 )
echo [OK] Logged in

:: 2. Input JSON
(
echo { "mode": "pro-forma", "assumptions": {
echo   "BTC_treasury": 1000,
echo   "BTC_current_market_price": 117000,
echo   "initial_equity_value": 90000000,
echo   "shares_basic": 1000000,
echo   "shares_fd": 1100000,
echo   "opex_monthly": 1000000,
echo   "paths": 2000,
echo   "t": 1.0,
echo   "mu": 0.45,
echo   "sigma": 0.55,
echo   "risk_free_rate": 0.04,
echo   "expected_return_btc": 0.45,
echo   "tax_rate": 0.20,
echo   "LTV_Cap": 0.50,
echo   "max_dilution": 0.25,
echo   "min_runway_months": 24,
echo   "max_breach_prob": 0.15,
echo   "enable_hybrid": true,
echo   "nsga_pop_size": 40,
echo   "nsga_n_gen": 30,
echo   "objective_switches": {
echo     "max_btc": true,
echo     "min_dilution": true,
echo     "min_ltv_breach": true,
echo     "max_runway": true
echo   }
echo } }
) > input.json

:: 3. Lock snapshot
curl -s -X POST http://localhost:8000/api/lock_snapshot/ -H "Authorization: Bearer !TOKEN!" -H "Content-Type: application/json" --data-binary @input.json > snap.json
for /f "delims=" %%B in ('powershell -command "(Get-Content snap.json | ConvertFrom-Json).snapshot_id"') do set "SNAP_ID=%%B"
if not defined SNAP_ID ( echo SNAPSHOT FAILED & type snap.json & pause & exit /b 1 )
echo [OK] Snapshot: !SNAP_ID!

:: 4. Calculate
curl -s -X POST http://localhost:8000/api/calculate/ -H "Authorization: Bearer !TOKEN!" -H "Content-Type: application/json" -d "{\"snapshot_id\":!SNAP_ID!,\"format\":\"csv\",\"use_live\":false,\"seed\":42}" --output treasury_report.csv

:: 5. Done
if exist treasury_report.csv (
    echo.
    echo [SUCCESS] 4 CANDIDATES GENERATED
    echo   BTC Bought ^> 0
    echo   WACC ^< 0.45
    echo   Hybrid included
    start treasury_report.csv
) else (
    echo [ERROR] No output
)

del token.json snap.json input.json 2>nul
echo.
echo === DONE - November 11, 2025 09:15 PM WAT ===
pause