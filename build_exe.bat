@echo off
REM === SyLC 3D Player - standalone no-console build (Nuitka) ===
REM Prereqs: a Python 3.14 venv with: requirements.txt + nuitka (run from the activated venv)
REM          + MSVC 2022 build tools. Run from this folder.
cd /d "%~dp0"

python -m nuitka SyLC_3D_Player.py ^
  --standalone ^
  --assume-yes-for-downloads ^
  --msvc=latest ^
  --windows-console-mode=disable ^
  --enable-plugin=pyside6 ^
  --include-module=mvc_demuxer_cpp ^
  --include-module=bluray_disc ^
  --include-data-files=edge264.dll=edge264.dll ^
  --include-data-files=libwinpthread-1.dll=libwinpthread-1.dll ^
  --include-data-files=mpv-2.dll=mpv-2.dll ^
  --include-data-files=ffprobe.exe=ffprobe.exe ^
  --include-data-files=avcodec-62.dll=avcodec-62.dll ^
  --include-data-files=avdevice-62.dll=avdevice-62.dll ^
  --include-data-files=avfilter-11.dll=avfilter-11.dll ^
  --include-data-files=avformat-62.dll=avformat-62.dll ^
  --include-data-files=avutil-60.dll=avutil-60.dll ^
  --include-data-files=swresample-6.dll=swresample-6.dll ^
  --include-data-files=swscale-9.dll=swscale-9.dll ^
  --windows-icon-from-ico=icon.ico ^
  --include-data-files=icon.png=icon.png ^
  --output-dir=build_nuitka ^
  --output-filename=SyLC_3D_Player.exe ^
  --company-name=SyLC --product-name="SyLC 3D Player" --file-version=4.0.0 --product-version=4.0.0

echo.
echo Build done. Result: build_nuitka\SyLC_3D_Player.dist\SyLC_3D_Player.exe
