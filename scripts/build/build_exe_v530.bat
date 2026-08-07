@echo off
REM === SyLC 3D Player v5.3.0 - standalone no-console build (Nuitka, one-folder) ===
REM Prereqs: Python 3.14 venv with requirements.txt + nuitka (run from the activated venv)
REM          + MSVC 2022 build tools. The script may be run from any directory.
REM The portable build ships no depth model weights. The twenty ONNX graphs
REM are fetched at runtime by model_fetcher.py from the manifest. The local
REM GitHub_v530 COMPLETE snapshot may contain those graphs and optional
REM MatAnyone2/TensorRT runtimes, but they are deliberately not swept into the
REM redistributable portable by this exact-file whitelist.
REM
REM Deferred/load-by-path modules are explicit because a frozen build can
REM otherwise compile and launch while silently losing the corresponding
REM feature. synth3d_matanyone2_worker.py remains an external data file for the
REM isolated PyTorch interpreter.
REM DO NOT list matroska.dll / ebml.dll / libwinpthread-1.dll as data files:
REM Nuitka already ships them as mvc_demuxer_cpp.pyd dependencies.
set "PROJECT_ROOT=%~dp0..\.."
cd /d "%PROJECT_ROOT%"
set "PYTHONPATH=%PROJECT_ROOT%\src;%PROJECT_ROOT%\runtime;%PYTHONPATH%"

uv run python -m nuitka SyLC_3D_Player.py ^
  --standalone ^
  --assume-yes-for-downloads ^
  --msvc=latest ^
  --windows-console-mode=disable ^
  --enable-plugin=pyside6 ^
  --include-module=mvc_demuxer_cpp ^
  --include-module=sylc.bluray_disc ^
  --include-module=sylc.lavf_h264_demuxer ^
  --include-module=sylc.lavf_hevc_source ^
  --include-module=sylc.hevc_decode_thread ^
  --include-module=sylc.hevc_stereo_detect ^
  --include-module=sylc.mvhevc_exporter ^
  --include-module=sylc.thumbnail_service ^
  --include-module=sylc.model_fetcher ^
  --include-module=sylc.model_download_dialog ^
  --include-module=sylc.trt_runtime ^
  --include-module=sylc.trt_fetcher ^
  --include-module=sylc.trt_engines ^
  --include-module=sylc.lookahead_scout ^
  --include-module=sylc.playback_memory ^
  --include-module=sylc.synth3d_matting_service ^
  --include-module=sylc.synth3d_stereo_comfort ^
  --include-package=sylc.native_renderer ^
  --include-package=sylc.cast_sender ^
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
  --include-data-files=runtime/onnxruntime.dll=runtime/onnxruntime.dll ^
  --include-data-files=runtime/DirectML.dll=runtime/DirectML.dll ^
  --include-data-files=NOTICE-THIRD-PARTY.md=NOTICE-THIRD-PARTY.md ^
  --include-data-files=models/MANIFEST.json=models/MANIFEST.json ^
  --include-data-files=src/sylc/synth3d_matanyone2_worker.py=synth3d_matanyone2_worker.py ^
  --include-data-dir=tools=tools ^
  --windows-icon-from-ico=assets/icon.ico ^
  --include-data-files=assets/icon.png=assets/icon.png ^
  --include-data-files=assets/splash.png=assets/splash.png ^
  --output-dir=build_release_v530 ^
  --output-filename=SyLC_3D_Player.exe ^
  --company-name=SyLC --product-name="SyLC 3D Player" --file-version=5.3.0 --product-version=5.3.0

if errorlevel 1 (
  echo.
  echo BUILD FAILED - nuitka returned %errorlevel%. Not copying tools\.
  exit /b 1
)

REM Nuitka strips .exe/.dll files out of --include-data-dir trees. Copy the
REM external encoding/muxing tools after a successful build.
xcopy /E /I /Y tools build_release_v530\SyLC_3D_Player.dist\tools >nul

echo.
echo Build done. Result in build_release_v530\SyLC_3D_Player.dist
