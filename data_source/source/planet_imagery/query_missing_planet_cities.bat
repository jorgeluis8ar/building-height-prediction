@echo off
setlocal

cd /d "S:\building-height-prediction"

for /F "usebackq delims=" %%C in ("data_source\source\planet_imagery\temp\missing_planet_city_slugs.txt") do (
    echo.
    echo ============================================================
    echo Querying missing city: %%C
    echo ============================================================

    python data_source\source\planet_imagery\search_planet_global_city_scenes.py ^
        --city-slug %%C ^
        --request-pause 1.0 ^
        --max-window-retries 8 ^
        --retry-base-delay 10

    if errorlevel 1 (
        echo.
        echo ERROR: Query failed for %%C.
        echo Completed windows remain checkpointed.
        exit /b 1
    )

    timeout /t 10 /nobreak >nul
)

echo.
echo SUCCESS: All missing cities were queried.
exit /b 0