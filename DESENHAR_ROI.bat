@echo off
cd /d "%~dp0"
if not exist rois mkdir rois
echo ============ DESENHAR A ROI ============
echo Arraste o ARQUIVO DE VIDEO para esta janela e tecle Enter:
set /p VID=
set VID=%VID:"=%
for %%F in ("%VID%") do set NOME=%%~nF
echo Nome da ROI [Enter = %NOME%] (se a pista tiver 2 faixas use p.ex. %NOME%_faixa1):
set /p RNOME=
if "%RNOME%"=="" set RNOME=%NOME%
echo.
echo Na janela: clique ESQUERDO = ponto, DIREITO = desfaz, ENTER = salva, ESC = cancela
python annotate.py roi --video "%VID%" --out "rois\%RNOME%_roi.json" --name "%RNOME%"
echo.
echo Arquivos criados em: %~dp0rois\%RNOME%_roi.json e as imagens .png e _sobre_video.jpg
pause
