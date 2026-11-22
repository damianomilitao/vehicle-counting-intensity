@echo off
REM Mede o tempo de processamento e a memoria de cada metodo (cerca de 30-60 min).
REM Nao use o computador enquanto roda. O resultado fica em tempo_resultados.json.
cd /d "%~dp0"
echo Instalando pacotes (na primeira vez pode demorar alguns minutos)...
python -m pip install --quiet opencv-python numpy scipy psutil ultralytics
echo.
python medir_tempo.py
echo.
echo Terminado. Pode fechar esta janela.
pause
