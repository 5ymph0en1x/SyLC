#include "mvc_decoder.h"
#include "h264_nal_parser.h"
#include <cerrno>
#include <cstring>
#include <iostream>
#include <cstdio>
#include <string> // Ensure std::string is available

#ifdef EDGE264_AVAILABLE
#include "edge264.h"

namespace mvc_demux {

// Edge264 logging callback
static void edge264_log_callback(const char* message, void* arg) {
    (void)arg;  // Unused
    // Print directly to stdout to ensure it's visible
    std::cout << message << std::flush;
}

} // namespace mvc_demux
#endif

namespace mvc_demux {

MVCDecoder::MVCDecoder()
    : decoder_(nullptr)
    , last_error_("")
{
}

MVCDecoder::~MVCDecoder() {
#ifdef EDGE264_AVAILABLE
    if (decoder_) {
        edge264_free(&decoder_);
    }
#endif
}

bool MVCDecoder::init(int n_threads) {
#ifdef EDGE264_AVAILABLE
    if (decoder_) {
        last_error_ = "Decoder already initialized";
        return false;
    }

    // Allocate decoder
    // Parameters: threads, log_cb, log_arg, log_mbs, alloc_cb, free_cb, alloc_arg
    decoder_ = edge264_alloc(
        n_threads,              // Auto-detect threads if -1
        edge264_log_callback,   // Enable logging
        nullptr,                // No log arg
        0,                      // Don't log macroblocks
        nullptr,                // Use default malloc
        nullptr,                // Use default free
        nullptr                 // No alloc arg
    );

    if (!decoder_) {
        last_error_ = "Failed to allocate edge264 decoder";
        return false;
    }

    return true;
#else
    last_error_ = "Edge264 support disabled at compile time";
    return false;
#endif
}

int MVCDecoder::decodeNAL(const uint8_t* nal_data, size_t nal_size) {
#ifdef EDGE264_AVAILABLE
    if (!decoder_) {
        last_error_ = "Decoder not initialized";
        return EINVAL;
    }

    if (!nal_data || nal_size == 0) {
        last_error_ = "Invalid NAL unit data";
        return EINVAL;
    }

    const uint8_t* end = nal_data + nal_size;
    const uint8_t* next_nal = nullptr;

    // Decode NAL unit
    // Parameters: decoder, buf, end, non_blocking, unref_cb, unref_arg, next_NAL
    int result = edge264_decode_NAL(
        decoder_,
        nal_data,
        end,
        1,              // NON-blocking mode (was causing deadlock!)
        nullptr,        // No unref callback
        nullptr,        // No unref arg
        &next_nal       // Get pointer to next NAL if in Annex B stream
    );

    // Handle result codes
    switch (result) {
        case 0:
            // Success
            last_error_ = "";
            break;
        case ENOTSUP:
            last_error_ = "Unsupported stream feature";
            break;
        case EBADMSG:
            last_error_ = "Invalid stream (corrupted data)";
            break;
        case ENOMEM:
            last_error_ = "Out of memory";
            break;
        case ENOBUFS:
            last_error_ = "Buffer full - call getFrame() to release frames";
            break;
        case EWOULDBLOCK:
            last_error_ = "Would block (shouldn't happen in blocking mode)";
            break;
        default:
            last_error_ = "Unknown error code: " + std::to_string(result);
            break;
    }

    return result;
#else
    return -1;
#endif
}

int MVCDecoder::decodeAnnexBStream(const uint8_t* data, size_t size) {
#ifdef EDGE264_AVAILABLE
    if (!decoder_) {
        last_error_ = "Decoder not initialized";
        return EINVAL;
    }
    if (!data || size == 0) {
        return 0;
    }

    size_t pos = 0;
    while (pos + 3 < size) {
        int prefixLen = H264NALParser::findStartCodePrefixLen(data + pos, size - pos);
        if (prefixLen == 0) {
            ++pos;
            continue;
        }

        size_t nalStart = pos + static_cast<size_t>(prefixLen);
        pos = nalStart;

        // Locate next start code
        size_t nextStart = size;
        size_t searchPos = nalStart + 1; // Skip current start code to avoid zero-length NALs
        while (searchPos + 3 < size) {
            if (H264NALParser::findStartCodePrefixLen(data + searchPos, size - searchPos) > 0) {
                nextStart = searchPos;
                break;
            }
            ++searchPos;
        }

        size_t nalSize = (nextStart < size) ? (nextStart - nalStart) : (size - nalStart);
        if (nalSize == 0) {
            pos = (nextStart < size) ? nextStart : size;
            continue;
        }

        const uint8_t* nalEnd = data + nalStart + nalSize;
        int result = edge264_decode_NAL(
            decoder_,
            data + nalStart,
            nalEnd,
            1,              // Non-blocking to avoid UI stalls
            nullptr,
            nullptr,
            nullptr
        );

        if (result != 0 && result != EWOULDBLOCK) {
            last_error_ = "decodeAnnexB failed on NAL: " + std::to_string(result);
            return result;
        }

        pos = (nextStart < size) ? nextStart : size;
    }

    return 0;
#else
    return -1;
#endif
}

bool MVCDecoder::getFrame(DecodedMVCFrame& out_frame) {
#ifdef EDGE264_AVAILABLE
    if (!decoder_) {
        last_error_ = "Decoder not initialized";
        return false;
    }

    Edge264Frame frame;
    std::memset(&frame, 0, sizeof(frame));

    // Try to get a frame
    // Parameters: decoder, out, borrow (0 = don't borrow, 1 = borrow)
    int result = edge264_get_frame(decoder_, &frame, 0);

    if (result == ENOMSG) {
        // No frame available yet
        return false;
    }

    if (result != 0) {
        last_error_ = "Failed to get frame: " + std::to_string(result);
        return false;
    }

    // Convert to our frame structure
    convertFrame(frame, out_frame);
    return true;
#else
    return false;
#endif
}

void MVCDecoder::flush() {
#ifdef EDGE264_AVAILABLE
    if (decoder_) {
        edge264_flush(decoder_);
    }
#endif
}

#ifdef EDGE264_AVAILABLE
void MVCDecoder::convertFrame(const Edge264Frame& src, DecodedMVCFrame& dst) {
    // Base view (always present)
    dst.base_view.y_plane = src.samples[0];
    dst.base_view.cb_plane = src.samples[1];
    dst.base_view.cr_plane = src.samples[2];
    dst.base_view.width = src.width_Y;
    dst.base_view.height = src.height_Y;
    dst.base_view.stride_y = src.stride_Y;
    dst.base_view.stride_c = src.stride_C;

    // Dependent view (MVC second view)
    dst.has_mvc = (src.samples_mvc[0] != nullptr);
    if (dst.has_mvc) {
        dst.dependent_view.y_plane = src.samples_mvc[0];
        dst.dependent_view.cb_plane = src.samples_mvc[1];
        dst.dependent_view.cr_plane = src.samples_mvc[2];
        dst.dependent_view.width = src.width_Y;  // Same as base
        dst.dependent_view.height = src.height_Y;
        dst.dependent_view.stride_y = src.stride_Y;
        dst.dependent_view.stride_c = src.stride_C;
    }

    // Frame IDs
    dst.frame_id = src.FrameId;
    dst.frame_id_mvc = src.FrameId_mvc;

    // Calculate display dimensions (after cropping)
    // frame_crop_offsets = {top, right, bottom, left}
    int crop_left = src.frame_crop_offsets[3];
    int crop_right = src.frame_crop_offsets[1];
    int crop_top = src.frame_crop_offsets[0];
    int crop_bottom = src.frame_crop_offsets[2];

    dst.display_width = src.width_Y - crop_left - crop_right;
    dst.display_height = src.height_Y - crop_top - crop_bottom;
}
#endif

} // namespace mvc_demux
