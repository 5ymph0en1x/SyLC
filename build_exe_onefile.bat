@echo off
REM === SyLC 3D Player - SINGLE-FILE no-console build (Nuitka --onefile) ===
REM Single .exe. First launch extracts the payload to a persistent cache dir
REM ({CACHE_DIR}\SyLC_3D_Player_v3) so subsequent launches are fast.
REM Prereqs: Python 3.13 venv with requirements + nuitka + pybind11; MSVC 2022.
cd /d "%~dp0"

python -m nuitka SyLC_3D_Player.py ^
  --onefile ^
  --assume-yes-for-downloads ^
  --msvc=latest ^
  --windows-console-mode=disable ^
  --enable-plugin=pyside6 ^
  --include-module=mvc_demuxer_cpp ^
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
  --onefile-tempdir-spec="{CACHE_DIR}/SyLC_3D_Player_v3" ^
  --output-dir=build_onefile ^
  --output-filename=SyLC_3D_Player_v3.0_win-x64.exe ^
  --company-name=SyLC --product-name="SyLC 3D Player" --file-version=3.0.0 --product-version=3.0.0

echo.
echo Single-file build done: build_onefile\SyLC_3D_Player_v3.0_win-x64.exe
pause
