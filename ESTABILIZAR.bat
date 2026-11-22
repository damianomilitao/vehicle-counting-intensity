@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
if not exist stab mkdir stab
echo ============ ESTABILIZAR (uma vez por video, DEPOIS de desenhar a ROI) ============
echo Arraste o ARQUIVO DE VIDEO para esta janela e tecle Enter:
set /p VID=
set VID=!VID:"=!
echo Arraste o arquivo da ROI (.json, pasta rois) e tecle Enter:
set /p ROI=
set ROI=!ROI:"=!
for %%F in ("!ROI!") do set RN=%%~nF
set RN=!RN:_roi=!
echo Calculando o movimento da camera (alguns minutos)...
python estabilizar.py --video "!VID!" --roi "!ROI!" --out "stab\!RN!_stab.npy"
echo.
echo Pronto: stab\!RN!_stab.npy  (o ANOTAR.bat usa este arquivo sozinho)
pause
