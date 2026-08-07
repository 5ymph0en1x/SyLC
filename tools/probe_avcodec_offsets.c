/* G:\SyLC-main\tools\probe_avcodec_offsets.c  (Task 9, Step 1)
   Prints the byte offsets of the two AVCodecContext fields written by offset in
   the HEVC HW-init path (get_format, hw_device_ctx), plus LIBAVCODEC_VERSION_MAJOR
   (must be 62 to match the bundled avcodec-62.dll ABI).

   These offsets MUST come from this compiled probe, never hand-counted.

   Build (headers only, no libs needed for offsetof):
     gcc -I<ffmpeg-8.0-src-root> -o probe_offsets.exe probe_avcodec_offsets.c
   Toolchain used: C:\msys64\mingw64\bin\gcc.exe (the edge264-build MinGW).
   ffmpeg source: ffmpeg-8.0 (https://ffmpeg.org/releases/ffmpeg-8.0.tar.xz).
   Generated headers libavutil/avconfig.h + libavutil/ffversion.h were absent
   from the source tree (they are configure-generated); minimal stubs were placed
   in the -I include path. Struct offsets on x64 are unaffected by their content. */
#include <stddef.h>
#include <stdio.h>
#include "libavcodec/avcodec.h"
int main(void) {
    printf("OFF_GET_FORMAT    %zu\n", offsetof(AVCodecContext, get_format));
    printf("OFF_HW_DEVICE_CTX %zu\n", offsetof(AVCodecContext, hw_device_ctx));
    /* Color-metadata fields of AVCodecParameters (HDR10 PQ/BT.2020 plumbing).
       Read by offset in lavf_hevc_source.py and cross-checked at runtime by name
       (av_color_space_name / av_color_transfer_name). LIBAVCODEC major 62 only. */
    printf("OFF_COLOR_SPACE     %zu\n", offsetof(AVCodecParameters, color_space));
    printf("OFF_COLOR_TRC       %zu\n", offsetof(AVCodecParameters, color_trc));
    printf("OFF_COLOR_PRIMARIES %zu\n", offsetof(AVCodecParameters, color_primaries));
    printf("OFF_COLOR_RANGE     %zu\n", offsetof(AVCodecParameters, color_range));
    printf("AVCODEC_MAJOR     %d\n", LIBAVCODEC_VERSION_MAJOR);   /* must be 62 */
    return 0;
}
