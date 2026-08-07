cmake_minimum_required(VERSION 3.15)
project(mvc_realtime_demuxer VERSION 1.0.0 LANGUAGES CXX)

# The application packages its runtime DLLs explicitly. Disable vcpkg's
# PowerShell applocal post-build step: on locked-down Windows/Store PowerShell
# installations that helper can fail after a completely successful link and
# incorrectly mark the native build as failed.
if(DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_APPLOCAL_DEPS OFF CACHE BOOL
        "Runtime DLLs are packaged by the SyLC release scripts" FORCE)
endif()

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(-D_FILE_OFFSET_BITS=64)
add_definitions(-D_LARGEFILE_SOURCE)

# Compiler options with aggressive optimizations for maximum performance
if(MSVC)
    add_compile_options(/W4)
    # MSVC optimization flags for maximum speed
    add_compile_options(/O2 /Oi /Ot /GL /arch:AVX2)
    # Enable link-time code generation
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /LTCG")
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    # Keep release artifacts portable: edge264 performs its own CPU dispatch and
    # the demuxer must not inherit the build host's instruction set.
    add_compile_options(-O3 -ffast-math -funroll-loops)
    # Enable LTO (Link Time Optimization) for inter-procedural optimizations
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Find pybind11
# Force pybind11's modern FindPython integration. The legacy FindPythonLibs path
# parses "3.14" as a split library name (python3 + newline + 14.lib) and makes a
# clean configure fail with current CMake/Python.
set(PYBIND11_FINDPYTHON ON)
if(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python_EXECUTABLE)
    # Preserve the interpreter explicitly selected by existing build scripts /
    # caches instead of silently picking vcpkg's helper Python.
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH
        "Python interpreter used for the extension" FORCE)
endif()
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# First try CONFIG mode (vcpkg provides this)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    # Try to find Python and use pip-installed pybind11
    find_package(Python3 COMPONENTS Interpreter Development QUIET)

    if(Python3_FOUND)
        message(STATUS "Python found: ${Python3_EXECUTABLE}")
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE pybind11_result
        )

        if(pybind11_result EQUAL 0 AND EXISTS "${pybind11_DIR}")
            message(STATUS "pybind11 found via Python: ${pybind11_DIR}")
            find_package(pybind11 CONFIG REQUIRED)
        else()
            message(FATAL_ERROR
                "pybind11 not found!\n"
                "Please install pybind11 using one of these methods:\n"
                "  1. vcpkg (recommended for Windows):\n"
                "     vcpkg install pybind11:x64-windows\n"
                "  2. Python pip:\n"
                "     python -m pip install pybind11\n"
                "  3. Set pybind11_DIR to the pybind11 cmake directory"
            )
        endif()
    else()
        message(FATAL_ERROR
            "pybind11 not found and Python not available!\n"
            "Please install pybind11 using one of these methods:\n"
            "  1. vcpkg (recommended for Windows):\n"
            "     vcpkg install pybind11:x64-windows\n"
            "  2. Install Python and then:\n"
            "     python -m pip install pybind11\n"
            "  3. Set pybind11_DIR to the pybind11 cmake directory"
        )
    endif()
endif()

# Try to find libmatroska and libebml (optional but recommended for full MVC support)
# vcpkg provides these as lowercase config packages
set(HAVE_LIBMATROSKA OFF)
set(MATROSKA_TARGETS "")

# Try vcpkg-style config packages first (lowercase)
find_package(matroska CONFIG QUIET)
find_package(ebml CONFIG QUIET)

# Debug output
message(STATUS "matroska_FOUND: ${matroska_FOUND}")
message(STATUS "ebml_FOUND: ${ebml_FOUND}")

# Check if vcpkg is being used
# Try multiple detection methods
set(VCPKG_DETECTED FALSE)

if(DEFINED VCPKG_TARGET_TRIPLET)
    message(STATUS "vcpkg detected via VCPKG_TARGET_TRIPLET: ${VCPKG_TARGET_TRIPLET}")
    set(VCPKG_DETECTED TRUE)
endif()

if(DEFINED CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg")
    message(STATUS "vcpkg detected via CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
    set(VCPKG_DETECTED TRUE)
endif()

# Also check if _VCPKG_INSTALLED_DIR is set (vcpkg sets this internally)
if(DEFINED _VCPKG_INSTALLED_DIR)
    message(STATUS "vcpkg detected via _VCPKG_INSTALLED_DIR: ${_VCPKG_INSTALLED_DIR}")
    set(VCPKG_DETECTED TRUE)
endif()

if(VCPKG_DETECTED)
    message(STATUS "vcpkg detected - attempting manual library detection")

    # Try to extract vcpkg root from various sources
    set(VCPKG_ROOT "")

    if(DEFINED _VCPKG_INSTALLED_DIR)
        # This is the most reliable - vcpkg sets this to {root}/installed (without triplet)
        string(REGEX REPLACE "/installed$" "" VCPKG_ROOT "${_VCPKG_INSTALLED_DIR}")
        string(REGEX REPLACE "\\\\installed$" "" VCPKG_ROOT "${VCPKG_ROOT}")
        string(REPLACE "\\" "/" VCPKG_ROOT "${VCPKG_ROOT}")
        message(STATUS "VCPKG_ROOT extracted from _VCPKG_INSTALLED_DIR: ${VCPKG_ROOT}")
    elseif(CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg")
        string(REGEX REPLACE "/scripts/buildsystems/vcpkg.cmake$" "" VCPKG_ROOT "${CMAKE_TOOLCHAIN_FILE}")
        string(REGEX REPLACE "\\\\scripts\\\\buildsystems\\\\vcpkg.cmake$" "" VCPKG_ROOT "${CMAKE_TOOLCHAIN_FILE}")
        string(REPLACE "\\" "/" VCPKG_ROOT "${VCPKG_ROOT}")
        message(STATUS "VCPKG_ROOT extracted from CMAKE_TOOLCHAIN_FILE: ${VCPKG_ROOT}")

    endif()

    # Determine the triplet
    set(VCPKG_TRIPLET "")
    if(DEFINED VCPKG_TARGET_TRIPLET)
        set(VCPKG_TRIPLET ${VCPKG_TARGET_TRIPLET})
    elseif(DEFINED _VCPKG_INSTALLED_DIR)
        # Extract from _VCPKG_INSTALLED_DIR path (e.g., .../installed/x64-windows)
        string(REGEX MATCH "/([^/]+)$" VCPKG_TRIPLET "${_VCPKG_INSTALLED_DIR}")
        string(REPLACE "/" "" VCPKG_TRIPLET "${VCPKG_TRIPLET}")
    elseif(WIN32)
        set(VCPKG_TRIPLET "x64-windows")
    else()
        set(VCPKG_TRIPLET "x64-linux")
    endif()
    message(STATUS "VCPKG_TRIPLET: ${VCPKG_TRIPLET}")

    # Determine installed directory (must include triplet subdirectory)
    if(DEFINED _VCPKG_INSTALLED_DIR AND VCPKG_TRIPLET)
        # _VCPKG_INSTALLED_DIR is the base (e.g., C:/vcpkg/installed)
        # We need to append the triplet to get the actual library location
        set(VCPKG_INSTALLED "${_VCPKG_INSTALLED_DIR}/${VCPKG_TRIPLET}")
    elseif(VCPKG_ROOT AND VCPKG_TRIPLET)
        set(VCPKG_INSTALLED "${VCPKG_ROOT}/installed/${VCPKG_TRIPLET}")
    endif()

    if(VCPKG_INSTALLED)
        message(STATUS "VCPKG_INSTALLED: ${VCPKG_INSTALLED}")

        # Manually find libraries in vcpkg
        message(STATUS "Searching for libraries in: ${VCPKG_INSTALLED}/lib")

        find_library(EBML_LIBRARY_RELEASE
            NAMES ebml
            PATHS "${VCPKG_INSTALLED}/lib"
            NO_DEFAULT_PATH
        )
        message(STATUS "EBML_LIBRARY_RELEASE: ${EBML_LIBRARY_RELEASE}")

        find_library(EBML_LIBRARY_DEBUG
            NAMES ebml
            PATHS "${VCPKG_INSTALLED}/debug/lib"
            NO_DEFAULT_PATH
        )
        message(STATUS "EBML_LIBRARY_DEBUG: ${EBML_LIBRARY_DEBUG}")

        find_library(MATROSKA_LIBRARY_RELEASE
            NAMES matroska
            PATHS "${VCPKG_INSTALLED}/lib"
            NO_DEFAULT_PATH
        )
        message(STATUS "MATROSKA_LIBRARY_RELEASE: ${MATROSKA_LIBRARY_RELEASE}")

        find_library(MATROSKA_LIBRARY_DEBUG
            NAMES matroska
            PATHS "${VCPKG_INSTALLED}/debug/lib"
            NO_DEFAULT_PATH
        )
        message(STATUS "MATROSKA_LIBRARY_DEBUG: ${MATROSKA_LIBRARY_DEBUG}")

        find_path(EBML_INCLUDE_DIR
            NAMES ebml/EbmlHead.h
            PATHS "${VCPKG_INSTALLED}/include"
            NO_DEFAULT_PATH
        )
        message(STATUS "EBML_INCLUDE_DIR: ${EBML_INCLUDE_DIR}")

        find_path(MATROSKA_INCLUDE_DIR
            NAMES matroska/KaxSegment.h
            PATHS "${VCPKG_INSTALLED}/include"
            NO_DEFAULT_PATH
        )
        message(STATUS "MATROSKA_INCLUDE_DIR: ${MATROSKA_INCLUDE_DIR}")

        if(EBML_LIBRARY_RELEASE AND MATROSKA_LIBRARY_RELEASE)
            message(STATUS "Found vcpkg libraries manually:")
            message(STATUS "  EBML: ${EBML_LIBRARY_RELEASE}")
            message(STATUS "  Matroska: ${MATROSKA_LIBRARY_RELEASE}")
            message(STATUS "  Include: ${EBML_INCLUDE_DIR}")

            set(HAVE_LIBMATROSKA ON)

            # Use optimized/debug library selection
            set(MATROSKA_LIBRARIES
                optimized ${MATROSKA_LIBRARY_RELEASE}
                optimized ${EBML_LIBRARY_RELEASE}
            )

            if(EBML_LIBRARY_DEBUG AND MATROSKA_LIBRARY_DEBUG)
                list(APPEND MATROSKA_LIBRARIES
                    debug ${MATROSKA_LIBRARY_DEBUG}
                    debug ${EBML_LIBRARY_DEBUG}
                )
            endif()

            set(MATROSKA_INCLUDE_DIRS ${MATROSKA_INCLUDE_DIR} ${EBML_INCLUDE_DIR})
        endif()
    endif()
endif()

# If still not found, try CONFIG packages
if(NOT HAVE_LIBMATROSKA AND (matroska_FOUND OR ebml_FOUND OR TARGET matroska OR TARGET ebml))
    message(STATUS "Matroska packages found via CONFIG mode")

    # vcpkg provides targets - check various possible names
    # Common patterns: matroska, matroska::matroska, ebml, ebml::ebml
    set(MATROSKA_TARGETS "")
    foreach(target_name IN ITEMS matroska ebml matroska::matroska ebml::ebml Matroska::Matroska EBML::EBML)
        if(TARGET ${target_name})
            list(APPEND MATROSKA_TARGETS ${target_name})
            message(STATUS "  Found target: ${target_name}")
        endif()
    endforeach()

    # If no targets found but package was found, try various variable naming conventions
    if(NOT MATROSKA_TARGETS)
        message(STATUS "  No modern targets found, checking for library variables...")

        # Try lowercase variables (some vcpkg ports use this)
        if(DEFINED matroska_LIBRARIES)
            list(APPEND MATROSKA_LIBRARIES ${matroska_LIBRARIES})
            message(STATUS "  Using matroska_LIBRARIES: ${matroska_LIBRARIES}")
        endif()
        if(DEFINED ebml_LIBRARIES)
            list(APPEND MATROSKA_LIBRARIES ${ebml_LIBRARIES})
            message(STATUS "  Using ebml_LIBRARIES: ${ebml_LIBRARIES}")
        endif()

        # Try uppercase variables (traditional CMake style)
        if(DEFINED MATROSKA_LIBRARIES)
            list(APPEND MATROSKA_LIBRARIES ${MATROSKA_LIBRARIES})
            message(STATUS "  Using MATROSKA_LIBRARIES: ${MATROSKA_LIBRARIES}")
        endif()
        if(DEFINED EBML_LIBRARIES)
            list(APPEND MATROSKA_LIBRARIES ${EBML_LIBRARIES})
            message(STATUS "  Using EBML_LIBRARIES: ${EBML_LIBRARIES}")
        endif()

        # Try include directories
        foreach(var_name IN ITEMS matroska_INCLUDE_DIRS MATROSKA_INCLUDE_DIRS matroska_INCLUDE_DIR MATROSKA_INCLUDE_DIR)
            if(DEFINED ${var_name})
                list(APPEND MATROSKA_INCLUDE_DIRS ${${var_name}})
                message(STATUS "  Using ${var_name}: ${${var_name}}")
            endif()
        endforeach()

        foreach(var_name IN ITEMS ebml_INCLUDE_DIRS EBML_INCLUDE_DIRS ebml_INCLUDE_DIR EBML_INCLUDE_DIR)
            if(DEFINED ${var_name})
                list(APPEND MATROSKA_INCLUDE_DIRS ${${var_name}})
                message(STATUS "  Using ${var_name}: ${${var_name}}")
            endif()
        endforeach()
    endif()

    # Only enable if we actually found usable libraries or targets
    if(MATROSKA_TARGETS OR MATROSKA_LIBRARIES)
        message(STATUS "libmatroska found via CONFIG: Full MVC support enabled")
        set(HAVE_LIBMATROSKA ON)
    else()
        message(WARNING "Matroska packages found but no usable targets or libraries detected")
        message(WARNING "Falling back to manual detection...")
    endif()
endif()

# Only try traditional detection if still not found
if(NOT HAVE_LIBMATROSKA)
    # Try traditional find_package (uppercase)
    find_package(Matroska QUIET)
    find_package(EBML QUIET)

    if(Matroska_FOUND AND EBML_FOUND)
        set(MATROSKA_LIBRARIES ${Matroska_LIBRARIES} ${EBML_LIBRARIES})
        set(MATROSKA_INCLUDE_DIRS ${Matroska_INCLUDE_DIRS} ${EBML_INCLUDE_DIRS})

        # Verify we actually have usable variables
        if(MATROSKA_LIBRARIES OR MATROSKA_INCLUDE_DIRS)
            message(STATUS "libmatroska found via find_package: Full MVC support enabled")
            set(HAVE_LIBMATROSKA ON)
        else()
            message(STATUS "find_package found Matroska but no usable libraries/includes, trying manual detection...")
            # V7b+++++++ FIX: Fall through to manual detection even when find_package "found" packages
            set(Matroska_FOUND FALSE)
        endif()
    endif()

    # V7b+++++++ FIX: Try manual detection if still not configured
    if(NOT HAVE_LIBMATROSKA)
        # Try manual detection
        find_path(MATROSKA_INCLUDE_DIR
            NAMES matroska/KaxSegment.h
            PATHS
                /usr/include
                /usr/local/include
                C:/Program Files/Matroska/include
                C:/msys64/mingw64/include
                ${CMAKE_PREFIX_PATH}/include
        )

        find_path(EBML_INCLUDE_DIR
            NAMES ebml/EbmlHead.h
            PATHS
                /usr/include
                /usr/local/include
                C:/Program Files/EBML/include
                C:/msys64/mingw64/include
                ${CMAKE_PREFIX_PATH}/include
        )

        find_library(MATROSKA_LIBRARY
            NAMES matroska libmatroska
            PATHS
                /usr/lib
                /usr/local/lib
                C:/Program Files/Matroska/lib
                C:/msys64/mingw64/lib
                ${CMAKE_PREFIX_PATH}/lib
        )

        find_library(EBML_LIBRARY
            NAMES ebml libebml
            PATHS
                /usr/lib
                /usr/local/lib
                C:/Program Files/EBML/lib
                C:/msys64/mingw64/lib
                ${CMAKE_PREFIX_PATH}/lib
        )

        if(MATROSKA_INCLUDE_DIR AND EBML_INCLUDE_DIR AND MATROSKA_LIBRARY AND EBML_LIBRARY)
            message(STATUS "libmatroska found (manual): Full MVC support enabled")
            set(HAVE_LIBMATROSKA ON)
            set(MATROSKA_LIBRARIES ${MATROSKA_LIBRARY} ${EBML_LIBRARY})
            set(MATROSKA_INCLUDE_DIRS ${MATROSKA_INCLUDE_DIR} ${EBML_INCLUDE_DIR})
        else()
            message(WARNING "libmatroska not found: Building with limited MVC support")
            message(WARNING "  To enable full MVC support, install libmatroska and libebml:")
            message(WARNING "    Ubuntu/Debian: sudo apt-get install libmatroska-dev libebml-dev")
            message(WARNING "    Windows (vcpkg): vcpkg install ebml:x64-windows matroska:x64-windows")
            message(WARNING "    Or build from source and set CMAKE_PREFIX_PATH")
        endif()
    endif()
endif()

# Final safety check: verify we have usable matroska libraries/includes
if(HAVE_LIBMATROSKA)
    # For targets, CMake handles includes automatically, so we're OK
    # For libraries, we MUST have include directories
    if(NOT MATROSKA_TARGETS AND NOT MATROSKA_INCLUDE_DIRS)
        message(WARNING "HAVE_LIBMATROSKA was enabled but no include directories found!")
        message(WARNING "Disabling matroska support to prevent build errors")
        set(HAVE_LIBMATROSKA OFF)
        set(MATROSKA_LIBRARIES "")
    endif()
endif()

# edge264 decoder headers. The decoder DLL is built independently with
# ../edge264/build_sylc_edge264.bat and resolved at runtime by mvc_decoder.cpp.
# Never link a MinGW import/static archive into the MSVC extension.
set(EDGE264_DIR "${CMAKE_SOURCE_DIR}/../edge264")
set(EDGE264_INCLUDE_DIR "${EDGE264_DIR}")

# edge264.dll is loaded with LoadLibrary/GetProcAddress by mvc_decoder.cpp.
if(WIN32)
    set(HAVE_EDGE264 ON)
    set(EDGE264_DYNAMIC_LOAD ON)
    message(STATUS "edge264 MVC decoder: dynamic runtime loading ENABLED")
else()
    set(HAVE_EDGE264 OFF)
    set(EDGE264_DYNAMIC_LOAD OFF)
endif()

# Source files
set(SOURCES
    src/file_byte_source.cpp
    src/frame_ring_buffer.cpp
    src/h264_nal_parser.cpp
    src/mvc_demuxer.cpp
    src/matroska_reader.cpp
    src/mvc_matroska_demuxer.cpp
    src/m2ts_reader.cpp
    src/mvc_m2ts_demuxer.cpp
    src/ssif_parser.cpp
    src/mvc_ssif_demuxer.cpp
    src/mvc_decoder.cpp
)

set(HEADERS
    include/file_byte_source.h
    include/frame_ring_buffer.h
    include/h264_nal_parser.h
    include/mvc_demuxer.h
    include/matroska_reader.h
    include/mvc_matroska_demuxer.h
    include/m2ts_reader.h
    include/mvc_m2ts_demuxer.h
    include/ssif_parser.h
    include/mvc_ssif_demuxer.h
    include/mvc_decoder.h
)

# Include directories
include_directories(include)
include_directories(${EDGE264_INCLUDE_DIR})

if(HAVE_LIBMATROSKA)
    # Only add include directories if using old-style variables
    # Modern targets handle this automatically via target_link_libraries
    if(NOT MATROSKA_TARGETS AND MATROSKA_INCLUDE_DIRS)
        include_directories(${MATROSKA_INCLUDE_DIRS})
    endif()
    add_definitions(-DHAVE_LIBMATROSKA)
endif()

# Python module
pybind11_add_module(mvc_demuxer_cpp
    ${SOURCES}
    python/python_bindings.cpp
)

# Link with edge264 if available
if(HAVE_EDGE264)
    target_compile_definitions(mvc_demuxer_cpp PRIVATE
        EDGE264_AVAILABLE EDGE264_DYNAMIC_LOAD)
    message(STATUS "MVCDecoder exposed; edge264.dll will be resolved at runtime")
else()
    message(STATUS "edge264 library not found - MVCDecoder will NOT be available (demuxer still works)")
endif()

if(HAVE_LIBMATROSKA)
    if(MATROSKA_TARGETS)
        # Use modern CMake targets (vcpkg)
        message(STATUS "Linking mvc_demuxer_cpp with targets: ${MATROSKA_TARGETS}")
        target_link_libraries(mvc_demuxer_cpp PRIVATE ${MATROSKA_TARGETS})
    elseif(MATROSKA_LIBRARIES)
        # Use old-style variables
        message(STATUS "Linking mvc_demuxer_cpp with libraries: ${MATROSKA_LIBRARIES}")
        target_link_libraries(mvc_demuxer_cpp PRIVATE ${MATROSKA_LIBRARIES})
    else()
        message(WARNING "HAVE_LIBMATROSKA is ON but no targets or libraries found!")
    endif()
endif()

# Set output directory
set_target_properties(mvc_demuxer_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)

# --- Native D3D11 renderer (Tokyo #3) ------------------------------------------
# Isolated, opt-in, Windows-only. Adds ONLY to the python module target; does not
# touch the HAVE_EDGE264 / anti-virus-safe gate above. The renderer source and the
# pybind exposure are both guarded by SYLC_NATIVE_RENDERER, so when this is OFF the
# module builds exactly as before.
if(WIN32)
    option(BUILD_NATIVE_RENDERER "Build the native D3D11 renderer (Tokyo #3)" ON)
else()
    set(BUILD_NATIVE_RENDERER OFF)
endif()

if(BUILD_NATIVE_RENDERER)
    # Compile every HLSL entry point as a dependency-aware build artifact and
    # embed the resulting bytecode in the PYD. Release machines no longer need
    # D3DCompile, source strings, or loose shader files next to the application.
    enable_language(RC)

    get_filename_component(_sylc_rc_dir "${CMAKE_RC_COMPILER}" DIRECTORY)
    set(_sylc_fxc_hints "${_sylc_rc_dir}")
    if(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION)
        list(APPEND _sylc_fxc_hints
            "C:/Program Files (x86)/Windows Kits/10/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
            "C:/Program Files/Windows Kits/10/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots;KitsRoot10]/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64")
    endif()
    if(DEFINED ENV{WindowsSdkVerBinPath})
        file(TO_CMAKE_PATH "$ENV{WindowsSdkVerBinPath}" _sylc_sdk_bin)
        list(APPEND _sylc_fxc_hints "${_sylc_sdk_bin}/x64")
    endif()
    find_program(SYLC_FXC_EXECUTABLE NAMES fxc.exe fxc
        HINTS ${_sylc_fxc_hints})
    if(NOT SYLC_FXC_EXECUTABLE)
        get_filename_component(_sylc_kits_root
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots;KitsRoot10]"
            ABSOLUTE)
        file(GLOB _sylc_fxc_candidates LIST_DIRECTORIES FALSE
            "${_sylc_kits_root}/bin/*/x64/fxc.exe"
            "${_sylc_kits_root}/bin/x64/fxc.exe")
        list(SORT _sylc_fxc_candidates)
        list(REVERSE _sylc_fxc_candidates)
        if(_sylc_fxc_candidates)
            list(GET _sylc_fxc_candidates 0 SYLC_FXC_EXECUTABLE)
            set(SYLC_FXC_EXECUTABLE "${SYLC_FXC_EXECUTABLE}" CACHE FILEPATH
                "DirectX shader compiler used for SyLC D3D11 resources" FORCE)
        endif()
    endif()
    if(NOT SYLC_FXC_EXECUTABLE)
        message(FATAL_ERROR
            "BUILD_NATIVE_RENDERER requires fxc.exe from the Windows SDK")
    endif()

    set(SYLC_SHADER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_shaders")
    file(MAKE_DIRECTORY "${SYLC_SHADER_BINARY_DIR}")
    function(sylc_compile_hlsl output_var output_name source entry profile)
        set(_output "${SYLC_SHADER_BINARY_DIR}/${output_name}.cso")
        add_custom_command(
            OUTPUT "${_output}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${SYLC_SHADER_BINARY_DIR}"
            COMMAND "${SYLC_FXC_EXECUTABLE}"
                /nologo /Ges /O3 /Qstrip_debug /Qstrip_reflect
                /E "${entry}" /T "${profile}"
                /Fo "${_output}" "${source}"
            DEPENDS "${source}"
            COMMENT "FXC ${output_name} (${entry}/${profile})"
            VERBATIM)
        set(${output_var} "${_output}" PARENT_SCOPE)
    endfunction()

    set(_native_vert "${CMAKE_CURRENT_SOURCE_DIR}/../src/sylc/native_renderer/shaders/yuv_framepack.vert.hlsl")
    set(_native_frag "${CMAKE_CURRENT_SOURCE_DIR}/../src/sylc/native_renderer/shaders/yuv_framepack.frag.hlsl")
    set(_sbs_hlsl "${CMAKE_CURRENT_SOURCE_DIR}/shaders/yuv_to_nv12_sbs.hlsl")
    set(_synth_hlsl "${CMAKE_CURRENT_SOURCE_DIR}/shaders/synth3d.hlsl")
    set(_synth_owner_hlsl "${CMAKE_CURRENT_SOURCE_DIR}/shaders/synth3d_ownership.hlsl")
    set(_stereo_lab_hlsl "${CMAKE_CURRENT_SOURCE_DIR}/shaders/stereo_lab.hlsl")

    sylc_compile_hlsl(SYLC_NATIVE_VS_CSO native_vs "${_native_vert}" main vs_5_0)
    sylc_compile_hlsl(SYLC_NATIVE_PS_CSO native_ps "${_native_frag}" main ps_5_0)
    sylc_compile_hlsl(SYLC_SBS_VS_CSO sbs_vs "${_sbs_hlsl}" VS vs_5_0)
    sylc_compile_hlsl(SYLC_SBS_PS_LUMA_CSO sbs_ps_luma "${_sbs_hlsl}" PS_Luma ps_5_0)
    sylc_compile_hlsl(SYLC_SBS_PS_CHROMA_CSO sbs_ps_chroma "${_sbs_hlsl}" PS_Chroma ps_5_0)
    sylc_compile_hlsl(SYLC_SYNTH3D_VS_FULL_CSO synth3d_vs_full "${_synth_hlsl}" VS_Full vs_5_0)
    sylc_compile_hlsl(SYLC_SYNTH3D_PS_DEPTH_PREP_CSO synth3d_ps_depth_prep "${_synth_hlsl}" PS_DepthPrep ps_5_0)
    sylc_compile_hlsl(SYLC_SYNTH3D_PS_WARP_LUMA_CSO synth3d_ps_warp_luma "${_synth_hlsl}" PS_WarpLuma ps_5_0)
    sylc_compile_hlsl(SYLC_SYNTH3D_PS_WARP_CHROMA_CSO synth3d_ps_warp_chroma "${_synth_hlsl}" PS_WarpChroma ps_5_0)
    sylc_compile_hlsl(SYLC_SYNTH3D_PS_DEPTH_VIEW_LUMA_CSO synth3d_ps_depth_view_luma "${_synth_hlsl}" PS_DepthViewLuma ps_5_0)
    sylc_compile_hlsl(SYLC_SYNTH3D_PS_DEPTH_VIEW_CHROMA_CSO synth3d_ps_depth_view_chroma "${_synth_hlsl}" PS_DepthViewChroma ps_5_0)
    sylc_compile_hlsl(SYLC_SYNTH3D_PS_PLATE_ACCUM_CSO synth3d_ps_plate_accum "${_synth_hlsl}" PS_PlateAccum ps_5_0)
    sylc_compile_hlsl(SYLC_SYNTH3D_PS_PROVENANCE_CSO synth3d_ps_provenance "${_synth_hlsl}" PS_WarpProvenance ps_5_0)
    sylc_compile_hlsl(SYLC_SYNTH3D_CS_OWNER_UNCERTAINTY_CSO synth3d_cs_owner_uncertainty "${_synth_owner_hlsl}" CS_OwnerUncertainty cs_5_0)
    sylc_compile_hlsl(SYLC_SYNTH3D_CS_OWNER_DILATE_CSO synth3d_cs_owner_dilate "${_synth_owner_hlsl}" CS_OwnerDilate cs_5_0)
    sylc_compile_hlsl(SYLC_SYNTH3D_CS_OWNER_LOCAL_CSO synth3d_cs_owner_local "${_synth_owner_hlsl}" CS_OwnerLocal cs_5_0)
    sylc_compile_hlsl(SYLC_SYNTH3D_CS_OWNER_PROPAGATE_CSO synth3d_cs_owner_propagate "${_synth_owner_hlsl}" CS_OwnerPropagate cs_5_0)
    sylc_compile_hlsl(SYLC_SYNTH3D_CS_OWNER_COMPOSE_CSO synth3d_cs_owner_compose "${_synth_owner_hlsl}" CS_OwnerCompose cs_5_0)
    sylc_compile_hlsl(SYLC_STEREO_LAB_PS_LUMA_CSO stereo_lab_ps_luma "${_stereo_lab_hlsl}" PS_LabLuma ps_5_0)
    sylc_compile_hlsl(SYLC_STEREO_LAB_PS_CHROMA_CSO stereo_lab_ps_chroma "${_stereo_lab_hlsl}" PS_LabChroma ps_5_0)
    sylc_compile_hlsl(SYLC_STEREO_LAB_PS_METRICS_CSO stereo_lab_ps_metrics "${_stereo_lab_hlsl}" PS_LabMetrics ps_5_0)
    sylc_compile_hlsl(SYLC_STEREO_LAB_PS_PAIR_FIELD_CSO stereo_lab_ps_pair_field "${_stereo_lab_hlsl}" PS_PairField ps_5_0)

    set(SYLC_SHADER_OUTPUTS
        "${SYLC_NATIVE_VS_CSO}" "${SYLC_NATIVE_PS_CSO}"
        "${SYLC_SBS_VS_CSO}" "${SYLC_SBS_PS_LUMA_CSO}" "${SYLC_SBS_PS_CHROMA_CSO}"
        "${SYLC_SYNTH3D_VS_FULL_CSO}" "${SYLC_SYNTH3D_PS_DEPTH_PREP_CSO}"
        "${SYLC_SYNTH3D_PS_WARP_LUMA_CSO}" "${SYLC_SYNTH3D_PS_WARP_CHROMA_CSO}"
        "${SYLC_SYNTH3D_PS_DEPTH_VIEW_LUMA_CSO}"
        "${SYLC_SYNTH3D_PS_DEPTH_VIEW_CHROMA_CSO}"
        "${SYLC_SYNTH3D_PS_PLATE_ACCUM_CSO}"
        "${SYLC_SYNTH3D_PS_PROVENANCE_CSO}"
        "${SYLC_SYNTH3D_CS_OWNER_UNCERTAINTY_CSO}"
        "${SYLC_SYNTH3D_CS_OWNER_DILATE_CSO}"
        "${SYLC_SYNTH3D_CS_OWNER_LOCAL_CSO}"
        "${SYLC_SYNTH3D_CS_OWNER_PROPAGATE_CSO}"
        "${SYLC_SYNTH3D_CS_OWNER_COMPOSE_CSO}"
        "${SYLC_STEREO_LAB_PS_LUMA_CSO}"
        "${SYLC_STEREO_LAB_PS_CHROMA_CSO}"
        "${SYLC_STEREO_LAB_PS_METRICS_CSO}"
        "${SYLC_STEREO_LAB_PS_PAIR_FIELD_CSO}")
    add_custom_target(sylc_shader_bytecode DEPENDS ${SYLC_SHADER_OUTPUTS})

    set(_sylc_shader_rc "${CMAKE_CURRENT_BINARY_DIR}/generated_shaders/shader_resources.rc")
    configure_file(src/shader_resources.rc.in "${_sylc_shader_rc}" @ONLY)
    set_source_files_properties("${_sylc_shader_rc}" PROPERTIES GENERATED TRUE)

    target_sources(mvc_demuxer_cpp PRIVATE
        src/native_renderer.cpp
        src/shader_resources.cpp
        "${_sylc_shader_rc}")
    add_dependencies(mvc_demuxer_cpp sylc_shader_bytecode)
    # SyLC Cast: NVENC HEVC encoder wrapper (NV12 D3D11 texture -> HEVC NAL bytes).
    # Windows-only body; nvEncodeAPI64.dll is dynamic-loaded at runtime (no NVENC .lib linked).
    target_sources(mvc_demuxer_cpp PRIVATE src/nvenc_encoder.cpp)
    # SyLC Cast: YUV -> NV12 side-by-side packer. Renders the renderer's 6 YUV plane SRVs
    # into one NV12 3840x1080 SBS texture (the NVENC input).
    target_sources(mvc_demuxer_cpp PRIVATE src/sbs_nv12_packer.cpp)
    # synth3d (2D->3D): ONNX Runtime depth inference wrapper. ORT is dynamic-loaded
    # at runtime (LoadLibraryExW + GetProcAddress) -- no onnxruntime.lib is linked,
    # only the vendored C API headers are compiled in (see depth_engine.cpp).
    target_sources(mvc_demuxer_cpp PRIVATE src/depth_engine.cpp)
    # synth3d: temporal depth stabilizer (fit scale/shift + EMA + scene-cut snap
    # + percentile normalization). Pure C++ math, no ORT/D3D dependency.
    target_sources(mvc_demuxer_cpp PRIVATE src/depth_stabilizer.cpp)
    # One process-wide inference/stabilization service shared by every renderer
    # surface. GPU warp resources remain device-local; the immutable depth map
    # and ORT session do not.
    target_sources(mvc_demuxer_cpp PRIVATE src/shared_depth_service.cpp)
    # synth3d: the GPU half -- side^2 depth-prep pass (side = the depth preset's
    # inference grid), non-blocking staging readback ring, inference worker thread
    # (DepthEngine + DepthStabilizer), depth upload and
    # the DIBR warp / depth-view passes. Shader bytecode is embedded above.
    target_sources(mvc_demuxer_cpp PRIVATE src/synth3d.cpp)
    # Stereo Lab: additive post-warp coherence guard. It consumes Synth3D's
    # immutable raw planes/provenance and owns separate corrected outputs.
    target_sources(mvc_demuxer_cpp PRIVATE src/stereo_lab.cpp)
    target_include_directories(mvc_demuxer_cpp PRIVATE third_party/onnxruntime/include)
    target_compile_definitions(mvc_demuxer_cpp PRIVATE SYLC_NATIVE_RENDERER)
    # Runtime needs only D3D11/DXGI; shader compilation is a build dependency.
    target_link_libraries(mvc_demuxer_cpp PRIVATE d3d11 dxgi dxguid)
    # SyLC Cast (NVENC): vendored MIT nv-codec-headers only (nvEncodeAPI.h) — the
    # encoder DLL itself is dynamic-loaded from the user's NVIDIA driver at runtime,
    # nothing NVIDIA-licensed is compiled/linked in. See third_party/nv-codec-headers/LICENSE.
    target_include_directories(mvc_demuxer_cpp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nv-codec-headers)

    # Optional NVIDIA Optical Flow Accelerator probe/backend.  It uses the
    # driver API only (no cudart) and dynamically loads nvofapi64.dll, so a
    # build/run without CUDA or NVIDIA keeps the existing CPU flow unchanged.
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_sources(mvc_demuxer_cpp PRIVATE src/nvof_flow.cpp)
        target_include_directories(mvc_demuxer_cpp PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/NVIDIAOpticalFlowSDK)
        target_link_libraries(mvc_demuxer_cpp PRIVATE CUDA::cuda_driver)
        target_compile_definitions(mvc_demuxer_cpp PRIVATE SYLC_NVOF_CUDA)
        message(STATUS "NVIDIA Optical Flow CUDA backend: ENABLED")
    else()
        message(STATUS "NVIDIA Optical Flow CUDA backend: disabled (CUDA toolkit not found)")
    endif()
    message(STATUS "Native D3D11 renderer: ENABLED (FXC=${SYLC_FXC_EXECUTABLE})")
else()
    message(STATUS "Native D3D11 renderer: disabled")
endif()

# Installation
install(TARGETS mvc_demuxer_cpp
    LIBRARY DESTINATION .
)

# Optional: Build static library for C++ users
add_library(mvc_demuxer_static STATIC ${SOURCES})
target_include_directories(mvc_demuxer_static PUBLIC include)
target_include_directories(mvc_demuxer_static PUBLIC ${EDGE264_INCLUDE_DIR})

# Link static library with edge264 if available
if(HAVE_EDGE264)
    target_compile_definitions(mvc_demuxer_static PUBLIC
        EDGE264_AVAILABLE EDGE264_DYNAMIC_LOAD)
endif()

if(HAVE_LIBMATROSKA)
    if(MATROSKA_TARGETS)
        # Use modern CMake targets (vcpkg)
        target_link_libraries(mvc_demuxer_static PUBLIC ${MATROSKA_TARGETS})
    elseif(MATROSKA_LIBRARIES)
        # Use old-style variables
        target_link_libraries(mvc_demuxer_static PUBLIC ${MATROSKA_LIBRARIES})
        if(MATROSKA_INCLUDE_DIRS)
            target_include_directories(mvc_demuxer_static PUBLIC ${MATROSKA_INCLUDE_DIRS})
        endif()
    endif()
    target_compile_definitions(mvc_demuxer_static PUBLIC HAVE_LIBMATROSKA)
endif()

# Testing (optional)
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    # Add test subdirectory if you create tests
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "========================================")
message(STATUS "MVC Demuxer Configuration")
message(STATUS "========================================")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ Standard: C++${CMAKE_CXX_STANDARD}")
if(HAVE_LIBMATROSKA)
    message(STATUS "  libmatroska:  FOUND - Full MVC support")
    if(MATROSKA_TARGETS)
        message(STATUS "    Targets:    ${MATROSKA_TARGETS}")
    elseif(MATROSKA_LIBRARIES)
        message(STATUS "    Libraries:  ${MATROSKA_LIBRARIES}")
    else()
        message(STATUS "    Libraries:  (empty - linking may fail)")
    endif()
else()
    message(STATUS "  libmatroska:  NOT FOUND - Limited support")
endif()
message(STATUS "========================================")
message(STATUS "")
