@echo off
echo ============================================
echo  Simetrycal - Building Services
echo ============================================
echo.

if "%1"=="--no-cache" goto nocache
if "%1"=="/no-cache" goto nocache
if "%1"=="-nc" goto nocache
if "%1"=="--all" goto all
if "%1"=="/all" goto all
if "%1"=="-a" goto all
if "%1"=="--gui" goto gui
if "%1"=="/gui" goto gui
if "%1"=="-g" goto gui

echo Building Mail Receiver, Mail Preprocessor, Info Extractor and GUI...
docker-compose build mail-receiver mail-preprocessor info-extractor gui-backend gui-frontend

goto done

:nocache
echo Building Mail Receiver, Mail Preprocessor, Info Extractor and GUI (no cache)...
docker-compose build --no-cache mail-receiver mail-preprocessor info-extractor gui-backend gui-frontend
goto done

:gui
echo Building GUI only...
if "%2"=="--no-cache" (
    docker-compose build --no-cache gui-backend gui-frontend
) else (
    docker-compose build gui-backend gui-frontend
)
goto done

:all
echo Building ALL services...
if "%2"=="--no-cache" (
    docker-compose build --no-cache
) else (
    docker-compose build
)
goto done

:done
echo.
echo ============================================
echo  Build Complete!
echo ============================================
echo.
echo Usage:
echo   build.bat              Build mail-receiver, mail-preprocessor, info-extractor and gui
echo   build.bat --no-cache   Build without cache (clean build)
echo   build.bat --gui        Build GUI only (gui-backend + gui-frontend)
echo   build.bat --all        Build all services
echo   build.bat --all --no-cache  Build all services without cache
echo.
echo To start services: start.bat
echo.
