@echo off
REM === SyLC 3D Player v5.0.0 - SINGLE-FILE no-console build (Nuitka --onefile) ===
REM Single .exe. First launch extracts the payload to a persistent cache dir
REM ({CACHE_DIR}\SyLC_3D_Player_v5_0_0) so subsequent launches are fast.
REM Prereqs: Python 3.14 venv with requirements + nuitka; MSVC 2022 (run from the activated venv).
set "PROJECT_ROOT=%~dp0..\.."
cd /d "%PROJECT_ROOT%"
set "PYTHONPATH=%PROJECT_ROOT%\src;%PROJECT_ROOT%\runtime;%PYTHONPATH%"

python -m nuitka SyLC_3D_Player.py ^
  --onefile ^
  --assume-yes-for-downloads ^
  --msvc=latest ^
  --windows-console-mode=disable ^
  --enable-plugin=pyside6 ^
  --include-module=mvc_demuxer_cpp ^
  --include-module=sylc.bluray_disc ^
  --include-module=sylc.lavf_h264_demuxer ^
  --include-module=sylc.thumbnail_service ^
  --include-package=sylc.native_renderer ^
  --include-data-files=runtime/edge264.dll=runtime/edge264.dll ^
  --include-data-files=runtime/mpv-2.dll=runtime/mpv-2.dll ^
  --include-data-files=runtime/ffprobe.exe=runtime/ffprobe.exe ^
  --include-data-files=runtime/avcodec-62.dll=runtime/avcodec-62.dll ^
  --include-data-files=runtime/avdevice-62.dll=runtime/avdevice-62.dll ^
  --include-data-files=runtime/avfilter-11.dll=runtime/avfilter-11.dll ^
  --include-data-files=runtime/avformat-62.dll=runtime/avformat-62.dll ^
  --include-data-files=runtime/avutil-60.dll=runtime/avutil-60.dll ^
  --include-data-files=runtime/swresample-6.dll=runtime/swresample-6.dll ^
  --include-data-files=runtime/swscale-9.dll=runtime/swscale-9.dll ^
  --windows-icon-from-ico=assets/icon.ico ^
  --include-data-files=assets/icon.png=assets/icon.png ^
  --onefile-tempdir-spec="{CACHE_DIR}/SyLC_3D_Player_v5_0_0" ^
  --output-dir=build_onefile ^
  --output-filename=SyLC_3D_Player_v5.0.0_win-x64.exe ^
  --company-name=SyLC --product-name="SyLC 3D Player" --file-version=5.0.0 --product-version=5.0.0

echo.
echo Single-file build done: build_onefile\SyLC_3D_Player_v5.0.0_win-x64.exe
