@echo off
echo ============================================
echo  Simetrycal - Stopping Services
echo ============================================
echo.

if "%1"=="--clean" goto clean
if "%1"=="/clean" goto clean
if "%1"=="-c" goto clean

echo Stopping GUI, Info Extractor, Mail Preprocessor, Mail Receiver, pgAdmin, RabbitMQ and PostgreSQL...
docker-compose stop gui-frontend gui-backend gui-db info-extractor mail-preprocessor mail-receiver pgadmin rabbitmq postgres

echo.
echo ============================================
echo  Services Stopped!
echo ============================================
echo.
echo Data preserved. To start again: start.bat
echo To remove everything: stop.bat --clean
echo.
goto end

:clean
echo Stopping and removing ALL containers and volumes...
echo This will DELETE all data (emails, queues, users, etc.)
echo.
set /p CONFIRM="Are you sure? (Y/N): "
if /i "%CONFIRM%"=="Y" goto doclean
goto cancelled

:doclean
docker-compose down -v --remove-orphans
echo.
echo ============================================
echo  Everything Removed!
echo ============================================
echo.
echo All containers and data deleted.
echo Run start.bat to begin fresh.
echo.
goto end

:cancelled
echo.
echo Cancelled. Nothing was removed.
echo.

:end
