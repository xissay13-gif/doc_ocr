@echo off
chcp 65001 >nul
setlocal EnableExtensions
title Мониторинг папки - распознавание судебных приказов
pushd "%~dp0"

rem ==========================================================================
rem  ЗАПУСК НЕПРЕРЫВНОГО МОНИТОРИНГА ПАПКИ
rem
rem  Программа следит за папкой и распознаёт каждый документ сразу, как только
rem  он туда попал. Результат - CSV и распознанный PDF на каждый документ.
rem
rem  Остановка: Ctrl+C (на вопрос "Terminate batch job (Y/N)?" ответьте Y)
rem  либо просто закройте окно.
rem
rem  Путь к папке задаётся любым из трёх способов (сверху - главнее):
rem    1) перетащить папку мышью прямо на этот файл;
rem    2) прописать пути ниже, в блоке НАСТРОЙКИ;
rem    3) оставить пусто - возьмётся из court-ocr.json, а если его нет -
rem       папки input и output рядом с программой.
rem ==========================================================================

rem ============================== НАСТРОЙКИ =================================

rem Папка со сканами. Можно сетевую: \\server\scan\входящие
set "INPUT="

rem Папка результатов (CSV + PDF на каждый документ).
set "OUTPUT="

rem Если CSV и PDF нужно класть ВРОЗЬ - задайте папки отдельно.
rem Пусто - и то, и другое ляжет в OUTPUT.
set "CSV_DIR="
set "PDF_DIR="

rem Число потоков. Пусто - по числу ядер процессора.
set "THREADS="

rem Как часто проверять папку, секунд. Пусто - каждые 2 секунды.
set "INTERVAL="

rem ==========================================================================

rem Папку перетащили на этот файл - она и будет входной.
if not "%~1"=="" set "INPUT=%~1"

set "ARGS=--watch --no-menu"
if defined INPUT    set "ARGS=%ARGS% --input "%INPUT%""
if defined OUTPUT   set "ARGS=%ARGS% --output "%OUTPUT%""
if defined CSV_DIR  set "ARGS=%ARGS% --csv-dir "%CSV_DIR%""
if defined PDF_DIR  set "ARGS=%ARGS% --pdf-dir "%PDF_DIR%""
if defined THREADS  set "ARGS=%ARGS% --threads %THREADS%"
if defined INTERVAL set "ARGS=%ARGS% --interval %INTERVAL%"

rem Чем запускать: собранный exe рядом или исходники через Python.
set "RUN="
if exist "court-ocr.exe" set "RUN=court-ocr.exe"
if defined RUN goto :launch

if not exist "app.py" goto :no_program
py -3 --version >nul 2>&1 && set "RUN=py -3 app.py"
if defined RUN goto :launch
python --version >nul 2>&1 && set "RUN=python app.py"
if defined RUN goto :launch
goto :no_python

:launch
set /a TRIES=0

:again
echo.
%RUN% %ARGS%
set "CODE=%ERRORLEVEL%"

if "%CODE%"=="0" goto :stopped
if "%CODE%"=="130" goto :stopped
if "%CODE%"=="2" goto :setup_error

rem Неожиданная ошибка: мониторинг не должен молча умирать - перезапускаем.
set /a TRIES+=1
echo.
echo [!] Программа завершилась с ошибкой (код %CODE%).
if %TRIES% GEQ 5 goto :too_many
echo     Перезапуск через 10 секунд - попытка %TRIES% из 5. Ctrl+C - отмена.
timeout /t 10 >nul
goto :again

:stopped
echo.
echo Мониторинг остановлен.
goto :end

:setup_error
echo.
echo Не найден движок Tesseract или языковая модель - подробности выше.
echo Проверьте, что рядом с программой лежит папка tessdata,
echo а в ней файлы rus.traineddata и osd.traineddata.
goto :end

:too_many
echo.
echo Пять неудачных запусков подряд - остановка.
echo Посмотрите текст ошибки выше.
goto :end

:no_program
echo.
echo Рядом с этим файлом нет ни court-ocr.exe, ни app.py.
echo Положите "Мониторинг.bat" в папку с программой.
goto :end

:no_python
echo.
echo Не найден Python. Либо используйте собранный court-ocr.exe,
echo либо установите Python с python.org, отметив при установке
echo галочку "Add Python to PATH".
goto :end

:end
popd
echo.
pause
endlocal
