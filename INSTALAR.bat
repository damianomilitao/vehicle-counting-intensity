@echo off
cd /d "%~dp0"
echo Instalando as bibliotecas (so precisa fazer uma vez)...
python -m pip install --upgrade opencv-python numpy scipy
echo.
python -c "import cv2, numpy, scipy; print('TUDO CERTO - OpenCV', cv2.__version__)"
pause
