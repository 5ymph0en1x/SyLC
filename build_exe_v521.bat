@echo off
REM === SyLC 3D Player v5.2.1 - standalone no-console build (Nuitka, one-folder) ===
REM Prereqs: Python 3.14 venv with requirements.txt + nuitka (run from the activated venv)
REM          + MSVC 2022 build tools. Run from this folder (has current .pyd/.dll/tools).
REM v5.2.1 ships NO depth models. The twenty ONNX graphs (4.61 GB) live in the
REM HuggingFace repo Symphoenix/sylc_TRT and are fetched at runtime by
REM model_fetcher.py, driven by models/MANIFEST.json and the "Depth models..."
REM entry in the AI menu. That is why onnxruntime.dll and DirectML.dll are still
REM bundled but no models/*.onnx line appears below.
REM The two lines that MUST NOT be dropped when removing the model whitelist are
REM --include-module=model_fetcher and the MANIFEST.json data file: without them
REM the build compiles, launches, and simply has no downloader at all.
REM --include-module=trt_engines is load-bearing in a way the others are not:
REM the exe RE-INVOKES ITSELF with --sylc-trt-engine-probe to build each
REM TensorRT engine out of process (see trt_engines.py's header), so without
REM that module every one of the 21 probes fails with an ImportError and no
REM .trt_verified is ever written -- the exact v5.2.0 symptom this fixes.
REM
REM NEW IN v5.2.1 - three modules whose ONLY import is deferred inside a
REM function, so Nuitka cannot discover them by following the import graph:
REM   lookahead_scout          (mvc_decoder.set_lookahead_enabled)
REM   playback_memory          (PlayerWindow._playback_memory_store)
REM   synth3d_matting_service  (PlayerWindow, inside a try/except)
REM Omitting any of them still COMPILES and LAUNCHES; the feature simply never
REM works in the frozen build -- silent per-file memory loss, and a look-ahead
REM scout that never arms, which re-opens the one-frame tear at every cut that
REM SYLC_LOOKAHEAD_DECAY exists to close. Same failure class as trt_engines.
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
  --include-module=model_fetcher ^
  --include-module=model_download_dialog ^
  --include-module=trt_runtime ^
  --include-module=trt_fetcher ^
  --include-module=trt_engines ^
  --include-module=lookahead_scout ^
  --include-module=playback_memory ^
  --include-module=synth3d_matting_service ^
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
  --include-data-files=onnxruntime.dll=onnxruntime.dll ^
  --include-data-files=DirectML.dll=DirectML.dll ^
  --include-data-files=NOTICE-THIRD-PARTY.md=NOTICE-THIRD-PARTY.md ^
  --include-data-files=models/MANIFEST.json=models/MANIFEST.json ^
  --include-data-dir=tools=tools ^
  --windows-icon-from-ico=icon.ico ^
  --include-data-files=icon.png=icon.png ^
  --include-data-files=splash.png=splash.png ^
  --output-dir=build_release_v521 ^
  --output-filename=SyLC_3D_Player.exe ^
  --company-name=SyLC --product-name="SyLC 3D Player" --file-version=5.2.1 --product-version=5.2.1

REM Stop here on a failed build. Without this the xcopy below still runs and
REM leaves a .dist directory holding nothing but tools\ -- which looks enough
REM like a real build to be mistaken for one.
if errorlevel 1 (
  echo.
  echo BUILD FAILED - nuitka returned %errorlevel%. Not copying tools\.
  exit /b 1
)

REM Nuitka strips .exe/.dll files out of --include-data-dir trees (security
REM policy) -- x265/mp4box must be copied in AFTER the build.
xcopy /E /I /Y tools build_release_v521\SyLC_3D_Player.dist\tools >nul

echo.
echo Build done. Result in build_release_v521\SyLC_3D_Player.dist
