@echo off
setlocal

rem Runs FastAPI backend (uvicorn) and opens the frontend in the default browser

set HOST=127.0.0.1
set PORT=8000
set URL=http://%HOST%:%PORT%/

rem Start backend in a new window so it keeps running.
start "VeriDex Backend" /min cmd /c "python -u VeriDex_WebApp\app.py"

rem Wait for server to come up
echo Waiting for backend on %URL% ...
set /a i=0
:waitloop
tasklist /fi "imagename eq python.exe" >nul 2>&1

rem Try curling the root. If curl isn't available, fallback to ping-based wait.
where curl >nul 2>&1
if %errorlevel%==0 (
  for /l %%k in (1,1,60) do (
    curl -s %URL% >nul && goto opened
    timeout /t 1 >nul
  )
) else (
  for /l %%k in (1,1,30) do (
    timeout /t 1 >nul
  )
)

goto opened

:opened
echo Opening frontend: %URL%
start "VeriDex Web" %URL%

endlocal

