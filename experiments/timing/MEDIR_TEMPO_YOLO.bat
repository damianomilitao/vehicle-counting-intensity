@echo off
REM Instala o YOLO (ultralytics) e mede so o tempo do YOLO (cerca de 10-15 min).
cd /d "%~dp0"
python -m pip install ultralytics
echo.
python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
python medir_tempo.py --only yolo
echo.
echo Terminado. Pode fechar esta janela.
pause
