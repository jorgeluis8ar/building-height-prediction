@echo off
setlocal

cd /d "S:\building-height-prediction"

for /L %%O in (0,25,1850) do (
    echo.
    echo Selecting city batch at offset %%O

    python data_source\source\planet_imagery\select_planet_global_city_scenes.py ^
        --city-offset %%O ^
        --city-limit 25 ^
        --asset-check-concurrency 4

    if errorlevel 1 (
        echo ERROR: Selection failed at offset %%O.
        echo Completed cities and asset checks remain resumable.
        exit /b 1
    )

    timeout /t 20 /nobreak >nul
)

echo SUCCESS: All selection batches completed.
exit /b 0