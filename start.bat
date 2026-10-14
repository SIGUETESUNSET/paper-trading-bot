@echo off
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
    py -3 paper_trading.py --starting-cash 200 --max-allocation 0.25 --max-drawdown 0.15 --slippage-rate 0.0005 --symbol BTC-USD --current-year --require-recent --compare --max-window 20 --watch-minutes 60
    goto finish
)

where python >nul 2>nul
if not errorlevel 1 (
    python paper_trading.py --starting-cash 200 --max-allocation 0.25 --max-drawdown 0.15 --slippage-rate 0.0005 --symbol BTC-USD --current-year --require-recent --compare --max-window 20 --watch-minutes 60
    goto finish
)

echo Nie znaleziono Pythona. Zainstaluj Python 3 i zaznacz opcje dodania do PATH.

:finish
echo.
pause
