@echo off
REM === SyLC 3D Player v5.0.3 - standalone no-console build (Nuitka, one-folder) ===
REM Prereqs: Python 3.14 venv with requirements.txt + nuitka (run from the activated venv)
REM          + MSVC 2022 build tools. Run from this folder (has current .pyd/.dll/tools).
REM v5.0.3 additions over v5.0.1: cast_sender package (SyLC Cast sender) and
REM bundled tools/ (x265 + mp4box for the MV-HEVC export).
REM DO NOT list matroska.dll / ebml.dll / libwinpthread-1.dll as data files:
REM Nuitka already ships them as mvc_demuxer_cpp.pyd dependencies, and the
REM duplicate makes it abort with a FATAL data-file/dll conflict.
cd /d "%~dp0"

uv run python -m nuitka SyLC_3D_Player.py ^
  --standalone ^
  --assume-yes-for-downloads ^
  --msvc=latest ^
  --windows-console-mode=disable ^
  --enable-plugin=pyside6 ^
  --include-module=mvc_demuxer_cpp ^
  --include-module=bluray_disc ^
  --include-module=lavf_h264_demuxer ^
  --include-module=lavf_hevc_source ^
  --include-module=hevc_decode_thread ^
  --include-module=hevc_stereo_detect ^
  --include-module=mvhevc_exporter ^
  --include-module=thumbnail_service ^
  --include-package=native_renderer ^
  --include-package=cast_sender ^
  --include-data-files=edge264.dll=edge264.dll ^
  --include-data-files=mpv-2.dll=mpv-2.dll ^
  --include-data-files=ffprobe.exe=ffprobe.exe ^
  --include-data-files=avcodec-62.dll=avcodec-62.dll ^
  --include-data-files=avdevice-62.dll=avdevice-62.dll ^
  --include-data-files=avfilter-11.dll=avfilter-11.dll ^
  --include-data-files=avformat-62.dll=avformat-62.dll ^
  --include-data-files=avutil-60.dll=avutil-60.dll ^
  --include-data-files=swresample-6.dll=swresample-6.dll ^
  --include-data-files=swscale-9.dll=swscale-9.dll ^
  --include-data-dir=tools=tools ^
  --windows-icon-from-ico=icon.ico ^
  --include-data-files=icon.png=icon.png ^
  --include-data-files=splash.png=splash.png ^
  --output-dir=build_release_v503 ^
  --output-filename=SyLC_3D_Player.exe ^
  --company-name=SyLC --product-name="SyLC 3D Player" --file-version=5.0.3 --product-version=5.0.3

REM Nuitka strips .exe/.dll files out of --include-data-dir trees (security
REM policy) -- x265/mp4box must be copied in AFTER the build.
xcopy /E /I /Y tools build_release_v503\SyLC_3D_Player.dist\tools >nul

echo.
echo Build done. Result in build_release_v503\SyLC_3D_Player.dist
