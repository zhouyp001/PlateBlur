@echo off
set PYTHONUTF8=1
chcp 65001 >nul
cd /d "%~dp0.."
echo ========================================
echo   车牌马赛克工具 - Nuitka 打包脚本
echo ========================================
echo.

python -m nuitka --standalone ^
       --jobs=8 ^
       --windows-console-mode=disable ^
       --show-progress ^
       --show-scons ^
       --module-parameter=torch-disable-jit=yes ^
       --enable-plugin=pyside6 ^
       --include-qt-plugins=multimedia ^
       --include-package=ultralytics ^
       --include-package=utils ^
       --include-package=gui ^
       --include-package=torch.cuda ^
       --include-data-files=weights/best.pt=./weights/best.pt ^
       --include-data-files=weights/best_imgsz_1280.onnx=./weights/best_imgsz_1280.onnx ^
       --include-data-files=config/config.yml=./config/config.yml ^
       --include-data-files=openh264-1.8.0-win64.dll=./openh264-1.8.0-win64.dll ^
       --output-dir=dist ^
       main.py

echo.
if %ERRORLEVEL% EQU 0 (
    echo ========================================
    echo   打包成功！
    echo   产物目录: dist\main.dist\
    echo   可执行文件: dist\main.dist\main.exe
    echo ========================================
) else (
    echo ========================================
    echo   打包失败，请检查错误信息
    echo ========================================
)
pause
