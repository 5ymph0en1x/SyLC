cmake_minimum_required(VERSION 3.15)
project(mvc_realtime_demuxer VERSION 1.0.0 LANGUAGES CXX)

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
    # GCC/Clang optimization flags for maximum speed
    # -O3: Aggressive optimizations
    # -march=native: Use all CPU features available on build machine
    # -mtune=native: Tune for build machine's CPU
    # -ffast-math: Fast floating point (safe for video processing)
    # -funroll-loops: Unroll loops for better performance
    # -flto: Link-time optimization
    add_compile_options(-O3 -march=native -mtune=native -ffast-math -funroll-loops)
    # Enable LTO (Link Time Optimization) for inter-procedural optimizations
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Find pybind11
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

# edge264 decoder library (optional for MVC decoding)
# Use precompiled edge264 library (MinGW-compiled)
set(EDGE264_DIR "${CMAKE_SOURCE_DIR}/../edge264")

# Don't compile edge264 with MSVC - use precompiled MinGW library
# add_subdirectory(${EDGE264_DIR} ${CMAKE_BINARY_DIR}/edge264_build)

# Set include directory
set(EDGE264_INCLUDE_DIR "${EDGE264_DIR}")

# Find precompiled library
set(EDGE264_LIBRARY "${EDGE264_DIR}/libedge264.a")

# AV SAFETY & 3D FIX: Use Dynamic Loading (LoadLibrary) instead of Static Linking
# This prevents embedding the MinGW static lib into the MSVC binary (AV false positive)
# while keeping the C++ logic intact (Full 3D support).
set(HAVE_EDGE264 OFF)
add_definitions(-DEDGE264_DYNAMIC_LOAD)
message(STATUS "Enabling Dynamic Loading of edge264.dll (Anti-Virus Safe Mode)")

# When using MinGW-compiled edge264 with MSVC, we need MinGW runtime libraries
set(MINGW_LIBRARIES "")
if(WIN32 AND MSVC AND EDGE264_LIBRARY AND HAVE_EDGE264)
    # Check if edge264 was built with MinGW (look for .a extension or check for MinGW symbols)
    get_filename_component(EDGE264_EXT "${EDGE264_LIBRARY}" EXT)
    if(EDGE264_EXT STREQUAL ".a")
        message(STATUS "Detected MinGW-compiled edge264 library")
        message(STATUS "Searching for MinGW runtime libraries...")

        # Try to find MSYS2/MinGW installation
        set(MINGW_SEARCH_PATHS
            "C:/msys64/mingw64/lib"
            "C:/msys64/mingw64/x86_64-w64-mingw32/lib"
            "C:/msys2/mingw64/lib"
            "C:/msys2/mingw64/x86_64-w64-mingw32/lib"
            "$ENV{MSYSTEM_PREFIX}/lib"
            "$ENV{MINGW_PREFIX}/lib"
        )

        # Add GCC version-specific directories (for libgcc.a)
        # These are typically in lib/gcc/x86_64-w64-mingw32/<version>/
        foreach(base_path "C:/msys64/mingw64" "C:/msys2/mingw64")
            file(GLOB gcc_version_dirs "${base_path}/lib/gcc/x86_64-w64-mingw32/*")
            foreach(gcc_dir ${gcc_version_dirs})
                if(IS_DIRECTORY "${gcc_dir}")
                    list(APPEND MINGW_SEARCH_PATHS "${gcc_dir}")
                endif()
            endforeach()
        endforeach()

        # Find required MinGW libraries
        # CMake's find_library on Windows MSVC doesn't recognize .a files, so check directly
        set(MINGW_PTHREAD_LIB "")
        set(MINGW_GCC_LIB "")
        set(MINGW_MINGWEX_LIB "")

        # Check for libwinpthread.a
        foreach(search_path ${MINGW_SEARCH_PATHS})
            if(EXISTS "${search_path}/libwinpthread.a")
                set(MINGW_PTHREAD_LIB "${search_path}/libwinpthread.a")
                break()
            endif()
        endforeach()

        # Check for libgcc.a in GCC version-specific directories
        foreach(search_path ${MINGW_SEARCH_PATHS})
            if(EXISTS "${search_path}/libgcc.a")
                set(MINGW_GCC_LIB "${search_path}/libgcc.a")
                break()
            endif()
        endforeach()

        # Check for libmingwex.a
        foreach(search_path ${MINGW_SEARCH_PATHS})
            if(EXISTS "${search_path}/libmingwex.a")
                set(MINGW_MINGWEX_LIB "${search_path}/libmingwex.a")
                break()
            endif()
        endforeach()

        if(MINGW_PTHREAD_LIB)
            list(APPEND MINGW_LIBRARIES ${MINGW_PTHREAD_LIB})
            message(STATUS "  Found pthread: ${MINGW_PTHREAD_LIB}")
        else()
            message(WARNING "  pthread library not found - linking may fail!")
        endif()

        if(MINGW_GCC_LIB)
            list(APPEND MINGW_LIBRARIES ${MINGW_GCC_LIB})
            message(STATUS "  Found gcc: ${MINGW_GCC_LIB}")
        else()
            message(WARNING "  gcc library not found - linking may fail!")
        endif()

        if(MINGW_MINGWEX_LIB)
            list(APPEND MINGW_LIBRARIES ${MINGW_MINGWEX_LIB})
            message(STATUS "  Found mingwex: ${MINGW_MINGWEX_LIB}")
        else()
            message(WARNING "  mingwex library not found - linking may fail!")
        endif()

        if(NOT MINGW_PTHREAD_LIB OR NOT MINGW_GCC_LIB OR NOT MINGW_MINGWEX_LIB)
            message(WARNING "")
            message(WARNING "Some MinGW runtime libraries not found!")
            message(WARNING "  This may cause linker errors when using MinGW-compiled edge264 with MSVC")
            message(WARNING "  Searched in:")
            foreach(path ${MINGW_SEARCH_PATHS})
                message(WARNING "    - ${path}")
            endforeach()
            message(WARNING "")
            message(WARNING "  Install MSYS2 from https://www.msys2.org/ if not already installed")
            message(WARNING "")
        endif()
    endif()
endif()

# Source files
set(SOURCES
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

# Add MinGW compatibility layer when cross-compiling with MSVC
if(WIN32 AND MSVC AND MINGW_LIBRARIES)
    message(STATUS "Adding MinGW-MSVC compatibility layer (mingw_compat.cpp)")
    list(APPEND SOURCES src/mingw_compat.cpp)
endif()

set(HEADERS
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
    target_link_libraries(mvc_demuxer_cpp PRIVATE "${EDGE264_LIBRARY}")
    target_compile_definitions(mvc_demuxer_cpp PRIVATE EDGE264_AVAILABLE)
    message(STATUS "edge264 available - MVCDecoder will be exposed in Python")
else()
    message(STATUS "edge264 library not found - MVCDecoder will NOT be available (demuxer still works)")
endif()

# Link MinGW runtime libraries if needed (cross-compilation MSVC+MinGW)
if(MINGW_LIBRARIES)
    message(STATUS "Linking MinGW runtime libraries: ${MINGW_LIBRARIES}")
    target_link_libraries(mvc_demuxer_cpp PRIVATE ${MINGW_LIBRARIES})

    # MinGW libraries depend on C runtime functions with different naming conventions
    # Add MSVC legacy libraries to provide missing CRT symbols
    if(MSVC)
        target_link_libraries(mvc_demuxer_cpp PRIVATE
            legacy_stdio_definitions.lib  # Provides fprintf, _vsnprintf, _vscprintf
        )
    endif()
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
    target_sources(mvc_demuxer_cpp PRIVATE src/native_renderer.cpp)
    target_compile_definitions(mvc_demuxer_cpp PRIVATE SYLC_NATIVE_RENDERER)
    # Windows SDK (MSVC) and MinGW both resolve these to the system D3D libs.
    # d3dcompiler is used to compile the embedded HLSL at runtime (S2+).
    target_link_libraries(mvc_demuxer_cpp PRIVATE d3d11 dxgi dxguid d3dcompiler)
    message(STATUS "Native D3D11 renderer: ENABLED (NativeRenderer exposed)")
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
    target_link_libraries(mvc_demuxer_static PUBLIC "${EDGE264_LIBRARY}")
    target_compile_definitions(mvc_demuxer_static PUBLIC EDGE264_AVAILABLE)
endif()

# Link MinGW runtime libraries if needed
if(MINGW_LIBRARIES)
    target_link_libraries(mvc_demuxer_static PUBLIC ${MINGW_LIBRARIES})

    # MinGW libraries depend on C runtime functions with different naming conventions
    # Add MSVC legacy libraries to provide missing CRT symbols
    if(MSVC)
        target_link_libraries(mvc_demuxer_static PUBLIC
            legacy_stdio_definitions.lib  # Provides fprintf, _vsnprintf, _vscprintf
        )
    endif()
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
