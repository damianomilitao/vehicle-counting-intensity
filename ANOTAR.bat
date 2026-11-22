@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
if not exist gt mkdir gt
echo ============ ANOTAR VEICULOS ============
if exist anotar_ultimo.bat (
  call anotar_ultimo.bat
  echo Ultima anotacao: !RN! - !ANOT!  trecho !INI!-!FIM! s
  set /p CONT=Continuar essa mesma anotacao? [S = sim, N = nova] 
  if /i "!CONT!"=="S" goto rodar
)
echo Arraste o ARQUIVO DE VIDEO para esta janela e tecle Enter:
set /p VID=
set VID=!VID:"=!
echo Arraste o arquivo da ROI (.json, pasta rois) e tecle Enter:
set /p ROI=
set ROI=!ROI:"=!
rem o nome do CSV e da estabilizacao vem do VIDEO (uma mesma ROI pode servir a varios videos)
for %%F in ("!VID!") do set RN=%%~nF
echo Seu primeiro nome, sem acento (ex.: felipe):
set /p ANOT=
set INI=
echo Inicio do trecho em segundos [Enter = 300 ; videos A e B: digite 0]:
set /p INI=
if "!INI!"=="" set INI=300
set FIM=
echo Fim do trecho em segundos [Enter = 600 ; videos A e B: digite 9999]:
set /p FIM=
if "!FIM!"=="" set FIM=600
(
 echo set "VID=!VID!"
 echo set "ROI=!ROI!"
 echo set "RN=!RN!"
 echo set "ANOT=!ANOT!"
 echo set "INI=!INI!"
 echo set "FIM=!FIM!"
) > anotar_ultimo.bat

:rodar
echo.
echo TECLAS: espaco=play/pausa  v=carro m=moto t=onibus/caminhao o=outro
echo         a/d=1 quadro  Shift+A/Shift+D=1 segundo  +/-=velocidade  u=desfaz  q=salva e sai
echo Salva a cada marca. Se a janela fechar sozinha, ela reabre e continua de onde parou.
set /a TENT=0
:loop
set STABARG=
if exist "stab\!RN!_stab.npy" set STABARG=--stab "stab\!RN!_stab.npy"
python annotate.py events --video "!VID!" --roi "!ROI!" --annotator !ANOT! --out "gt\!RN!_!ANOT!.csv" --start !INI! --end !FIM! !STABARG!
set ERR=!errorlevel!
if !ERR! EQU 0 goto fim
if !ERR! EQU 1 goto erro
if !ERR! EQU 3 goto erro
if !ERR! EQU 2 goto erro
set /a TENT+=1
echo.
echo *** A janela fechou sozinha (tentativa !TENT!). Reabrindo em 3 s... (Ctrl+C para parar) ***
if !TENT! GEQ 30 goto fim
timeout /t 3 >nul
goto loop
:erro
echo.
echo *** Erro acima (nao foi travamento). Feche esta janela, abra o ANOTAR.bat de novo e tecle N. ***
echo Arraste o VIDEO, tecle Enter; so depois arraste a ROI e tecle Enter.
if exist anotar_ultimo.bat del anotar_ultimo.bat
pause
exit /b
:fim
echo.
echo Arquivo salvo em: %~dp0gt\!RN!_!ANOT!.csv
if exist crash_log.txt echo (se houve fechamentos, mande tambem o arquivo crash_log.txt)
pause
