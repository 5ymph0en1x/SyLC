@echo off
setlocal
set "MINGW_BIN=C:\msys64\mingw64\bin"
set "GCC=%MINGW_BIN%\gcc.exe"
if not exist "%GCC%" (
  echo ERROR: MSYS2 MinGW64 GCC was not found at %GCC%
  exit /b 1
)
set "PATH=%MINGW_BIN%;%PATH%"
"%GCC%" @build_base.rsp || exit /b 1
"%GCC%" @build_v2.rsp || exit /b 1
"%GCC%" @build_v3.rsp || exit /b 1
"%GCC%" @build_link.rsp || exit /b 1
echo Built edge264_candidate.dll
