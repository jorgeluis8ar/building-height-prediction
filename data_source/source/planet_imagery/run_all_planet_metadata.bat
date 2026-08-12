@echo off
setlocal

cd /d "S:\building-height-prediction"

set BATCH_SIZE=10
set FINAL_OFFSET=1860

for /L %%O in (0,%BATCH_SIZE%,%FINAL_OFFSET%) do (
    echo.
    echo ============================================================
    echo Starting city batch at offset %%O
    echo ============================================================

    python data_source\source\planet_imagery\search_planet_global_city_scenes.py ^
        --city-offset %%O ^
        --city-limit %BATCH_SIZE% ^
        --request-pause 1.0 ^
        --max-window-retries 8 ^
        --retry-base-delay 10

    if errorlevel 1 (
        echo.
        echo ERROR: Planet metadata search failed at city offset %%O.
        echo Completed city-year windows remain checkpointed.
        echo Rerun this batch file to resume.
        exit /b 1
    )

    echo Batch %%O completed. Waiting 30 seconds before the next batch.
    timeout /t 30 /nobreak >nul
)

echo.
echo SUCCESS: All requested city batches completed.
exit /b 0