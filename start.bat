@echo off
echo ============================================
echo  Simetrycal - Starting Services
echo ============================================
echo.

echo Starting PostgreSQL, RabbitMQ, pgAdmin, Mail Receiver, Mail Preprocessor, Info Extractor and GUI...
docker-compose up -d postgres rabbitmq pgadmin mail-receiver mail-preprocessor info-extractor gui-db gui-backend gui-frontend

echo.
echo Waiting for services to be ready...
timeout /t 5 /nobreak >nul

echo.
echo ============================================
echo  Services Started!
echo ============================================
echo.
echo  PostgreSQL:            localhost:14432
echo  pgAdmin:               http://localhost:14080
echo  RabbitMQ Management:   http://localhost:14673
echo  Mail Receiver API:     http://localhost:14001
echo  Mail Preprocessor API: http://localhost:14002
echo  Info Extractor API:    http://localhost:14003
echo  GUI:                   http://localhost:14090
echo  GUI Backend API:       http://localhost:14091
echo.
echo  pgAdmin credentials:
echo    Email:    admin@example.com
echo    Password: admin
echo.
echo  RabbitMQ credentials:
echo    User:     guest
echo    Password: guest
echo.
echo  GUI default users:
echo    root / root123       (Admin)
echo    manager / manager123 (Responsable)
echo    operador / operador123 (Operador)
echo.
echo ============================================
echo.
echo To view logs:  docker-compose logs -f mail-receiver mail-preprocessor info-extractor gui-backend
echo To stop:       stop.bat
echo.
