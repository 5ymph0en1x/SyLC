#include "mvc_decoder.h"
#include "h264_nal_parser.h"
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <thread>

#ifdef EDGE264_AVAILABLE
#include "edge264.h"
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace mvc_demux {

namespace {

struct Edge264Api {
#ifdef _WIN32
    HMODULE module = nullptr;
#endif
    decltype(&edge264_alloc) alloc = nullptr;
    decltype(&edge264_flush) flush = nullptr;
    decltype(&edge264_free) free_decoder = nullptr;
    decltype(&edge264_decode_NAL) decode_nal = nullptr;
    decltype(&edge264_get_frame) get_frame = nullptr;
    decltype(&edge264_return_frame) return_frame = nullptr;
    decltype(&edge264_bump_frames) bump_frames = nullptr;
    decltype(&edge264_get_busy_tasks) get_busy_tasks = nullptr;
    std::string diagnostic;
};

Edge264Api g_edge264;
std::mutex g_edge264_load_mutex;
int g_edge264_module_anchor = 0;

#ifdef _WIN32
std::wstring directoryOf(const std::wstring& path) {
    const auto pos = path.find_last_of(L"\\/");
    return pos == std::wstring::npos ? std::wstring() : path.substr(0, pos);
}

std::wstring modulePath(HMODULE module) {
    std::wstring result(32768, L'\0');
    const DWORD size = GetModuleFileNameW(module, result.data(),
                                          static_cast<DWORD>(result.size()));
    if (size == 0 || size >= result.size()) {
        return {};
    }
    result.resize(size);
    return result;
}

std::string win32Error(DWORD code) {
    char* text = nullptr;
    const DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER |
                        FORMAT_MESSAGE_FROM_SYSTEM |
                        FORMAT_MESSAGE_IGNORE_INSERTS;
    const DWORD length = FormatMessageA(flags, nullptr, code, 0,
                                        reinterpret_cast<char*>(&text), 0, nullptr);
    std::string result = length && text ? std::string(text, length)
                                        : ("Win32 error " + std::to_string(code));
    if (text) {
        LocalFree(text);
    }
    while (!result.empty() &&
           (result.back() == '\r' || result.back() == '\n' || result.back() == ' ')) {
        result.pop_back();
    }
    return result;
}

std::string utf8(const std::wstring& value) {
    if (value.empty()) return {};
    const int required = WideCharToMultiByte(
        CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
        nullptr, 0, nullptr, nullptr);
    if (required <= 0) return {};
    std::string result(static_cast<size_t>(required), '\0');
    WideCharToMultiByte(
        CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
        result.data(), required, nullptr, nullptr);
    return result;
}

template <typename T>
bool loadProc(HMODULE module, const char* name, T& destination,
              std::string& diagnostic) {
    destination = reinterpret_cast<T>(GetProcAddress(module, name));
    if (!destination) {
        diagnostic = std::string("edge264.dll: export manquant: ") + name;
        return false;
    }
    return true;
}
#endif

bool loadEdge264(std::string* diagnostic) {
    std::lock_guard<std::mutex> guard(g_edge264_load_mutex);
    if (g_edge264.alloc) {
        if (diagnostic) *diagnostic = g_edge264.diagnostic;
        return true;
    }

#ifdef _WIN32
    std::vector<std::wstring> candidates;

    wchar_t env_path[32768] = {};
    const DWORD env_size = GetEnvironmentVariableW(
        L"SYLC_EDGE264_DLL", env_path,
        static_cast<DWORD>(sizeof(env_path) / sizeof(env_path[0])));
    if (env_size > 0 && env_size < (sizeof(env_path) / sizeof(env_path[0]))) {
        candidates.emplace_back(env_path, env_size);
    }

    HMODULE self_module = nullptr;
    if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&g_edge264_module_anchor), &self_module)) {
        const auto self_dir = directoryOf(modulePath(self_module));
        if (!self_dir.empty()) candidates.push_back(self_dir + L"\\edge264.dll");
    }

    const auto exe_dir = directoryOf(modulePath(nullptr));
    if (!exe_dir.empty()) candidates.push_back(exe_dir + L"\\edge264.dll");
    candidates.emplace_back(L"edge264.dll");

    DWORD last_error = ERROR_MOD_NOT_FOUND;
    std::wstring loaded_path;
    for (const auto& candidate : candidates) {
        HMODULE module = nullptr;
        if (candidate.find_first_of(L"\\/") != std::wstring::npos) {
            module = LoadLibraryExW(candidate.c_str(), nullptr,
                                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        } else {
            module = LoadLibraryW(candidate.c_str());
        }
        if (module) {
            g_edge264.module = module;
            loaded_path = candidate;
            break;
        }
        last_error = GetLastError();
    }

    if (!g_edge264.module) {
        g_edge264.diagnostic =
            "Impossible de charger edge264.dll (" + win32Error(last_error) +
            "). Placez la DLL edge264 autonome a cote de "
            "mvc_demuxer_cpp.pyd, ou definissez SYLC_EDGE264_DLL.";
        if (diagnostic) *diagnostic = g_edge264.diagnostic;
        return false;
    }

    bool ok =
        loadProc(g_edge264.module, "edge264_alloc", g_edge264.alloc,
                 g_edge264.diagnostic) &&
        loadProc(g_edge264.module, "edge264_flush", g_edge264.flush,
                 g_edge264.diagnostic) &&
        loadProc(g_edge264.module, "edge264_free", g_edge264.free_decoder,
                 g_edge264.diagnostic) &&
        loadProc(g_edge264.module, "edge264_decode_NAL", g_edge264.decode_nal,
                 g_edge264.diagnostic) &&
        loadProc(g_edge264.module, "edge264_get_frame", g_edge264.get_frame,
                 g_edge264.diagnostic) &&
        loadProc(g_edge264.module, "edge264_return_frame",
                 g_edge264.return_frame, g_edge264.diagnostic) &&
        loadProc(g_edge264.module, "edge264_bump_frames",
                 g_edge264.bump_frames, g_edge264.diagnostic) &&
        loadProc(g_edge264.module, "edge264_get_busy_tasks",
                 g_edge264.get_busy_tasks, g_edge264.diagnostic);
    if (!ok) {
        const std::string missing_export = g_edge264.diagnostic;
        FreeLibrary(g_edge264.module);
        g_edge264 = {};
        g_edge264.diagnostic = missing_export;
        if (diagnostic) *diagnostic = missing_export;
        return false;
    }

    // Deliberately keep the DLL loaded for the process lifetime. Unloading code
    // while another decoder worker is active is a classic shutdown crash.
    std::ostringstream status;
    status << "edge264 charge dynamiquement";
    const auto resolved_path = modulePath(g_edge264.module);
    if (!resolved_path.empty()) {
        status << " depuis " << utf8(resolved_path);
    }
    g_edge264.diagnostic = status.str();
    if (diagnostic) *diagnostic = g_edge264.diagnostic;
    return true;
#else
    g_edge264.diagnostic =
        "Le chargement dynamique edge264 n'est implemente que sous Windows.";
    if (diagnostic) *diagnostic = g_edge264.diagnostic;
    return false;
#endif
}

void copyPlane(const uint8_t* source, int source_stride,
               int width, int height, std::vector<uint8_t>& destination) {
    destination.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int row = 0; row < height; ++row) {
        std::memcpy(destination.data() + static_cast<size_t>(row) * width,
                    source + static_cast<size_t>(row) * source_stride,
                    static_cast<size_t>(width));
    }
}

void edge264_unref_owned_nal(int, void* argument) {
    delete static_cast<std::vector<uint8_t>*>(argument);
}

} // namespace

} // namespace mvc_demux
#endif

namespace mvc_demux {

MVCDecoder::MVCDecoder()
    : decoder_(nullptr)
    , worker_threads_(0)
    , last_error_("")
{
}

MVCDecoder::~MVCDecoder() {
    close();
}

bool MVCDecoder::init(int n_threads) {
#ifdef EDGE264_AVAILABLE
    if (decoder_) {
        last_error_ = "Decoder already initialized";
        return false;
    }

    if (!loadEdge264(&last_error_)) {
        return false;
    }

    decoder_ = g_edge264.alloc(
        n_threads,              // Auto-detect threads if -1
        nullptr,                // Release DLL is built without verbose logging
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

    worker_threads_ = n_threads;
    abort_requested_.store(false, std::memory_order_release);
    last_error_.clear();
    return true;
#else
    last_error_ = "Edge264 support disabled at compile time";
    return false;
#endif
}

int MVCDecoder::decodeNAL(const uint8_t* nal_data, size_t nal_size) {
#ifdef EDGE264_AVAILABLE
    if (abort_requested_.load(std::memory_order_acquire)) {
        last_error_ = "Decode aborted";
        return ECANCELED;
    }
    if (!decoder_) {
        last_error_ = "Decoder not initialized";
        return EINVAL;
    }

    if (!nal_data || nal_size == 0) {
        last_error_ = "Invalid NAL unit data";
        return EINVAL;
    }

    // edge264 worker threads may retain slice bytes after decode_NAL returns.
    // Own each compressed NAL in C++ and release it from edge264's completion
    // callback; this removes the old Python deque lifetime workaround.
    // edge264's SIMD bitstream refill operates on whole machine/vector words.
    // Keep the same guard contract as the validated feeder: two bytes before
    // the NAL and 64 zero bytes after its logical end. Passing an exact-size
    // vector is not safe even though `end` is correct: a refill can legally
    // over-read within this caller-owned padding and otherwise consume heap
    // metadata as trailing bits, eventually corrupting the MVC DPB.
    constexpr size_t kPrefixGuard = 2;
    constexpr size_t kSuffixGuard = 64;
    auto* owned_nal = new (std::nothrow) std::vector<uint8_t>(
        kPrefixGuard + nal_size + kSuffixGuard, 0);
    if (!owned_nal) {
        last_error_ = "Out of memory while retaining compressed NAL";
        return ENOMEM;
    }
    (*owned_nal)[0] = 0xff;
    (*owned_nal)[1] = 0xff;
    std::memcpy(owned_nal->data() + kPrefixGuard, nal_data, nal_size);
    const uint8_t* begin = owned_nal->data() + kPrefixGuard;
    const uint8_t* end = begin + nal_size;
    // Current edge264 ABI: decoder, begin, end, unref callback, callback arg.
    // In synchronous mode the task is complete when decode_NAL returns. Keep
    // ownership local there: several patched edge264 builds have historically
    // treated a non-null unref callback differently on their inline worker path.
    // Worker mode still uses the official completion callback.
    const auto unref_cb =
        worker_threads_ == 0 ? nullptr : edge264_unref_owned_nal;
    void* unref_arg = worker_threads_ == 0 ? nullptr : owned_nal;
    int result =
        g_edge264.decode_nal(decoder_, begin, end, unref_cb, unref_arg);
    for (int retry = 0; (result == 119 || result == ENOBUFS) && retry < 4;
         ++retry) {
        if (abort_requested_.load(std::memory_order_acquire)) {
            result = ECANCELED;
            break;
        }
        g_edge264.bump_frames(decoder_);
        int drained = 0;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(40);
        do {
            DecodedMVCFrame rescued;
            if (fetchNativeFrame(rescued)) {
                ready_frames_.push_back(std::move(rescued));
                ++drained;
                continue;
            }
            if (std::chrono::steady_clock::now() >= deadline) break;
            if (abort_requested_.load(std::memory_order_acquire)) {
                result = ECANCELED;
                break;
            }
            // Workers finish slice/deblock tasks asynchronously. A short native
            // wait here replaces Python-side retry sleeps and releases the GIL.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (g_edge264.get_busy_tasks(decoder_) != 0 || drained == 0);

        if (drained == 0 && retry == 3) {
            break;
        }
        result =
            g_edge264.decode_nal(decoder_, begin, end, unref_cb, unref_arg);
    }
    if (worker_threads_ == 0 || result != 0) {
        delete owned_nal;
    }

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
    if (abort_requested_.load(std::memory_order_acquire)) {
        last_error_ = "Decode aborted";
        return ECANCELED;
    }
    if (!decoder_) {
        last_error_ = "Decoder not initialized";
        return EINVAL;
    }
    if (!data || size == 0) {
        return 0;
    }

    size_t pos = 0;
    while (pos + 3 < size) {
        if (abort_requested_.load(std::memory_order_acquire)) {
            last_error_ = "Decode aborted";
            return ECANCELED;
        }
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

        const int result = decodeNAL(data + nalStart, nalSize);

        // MinGW errno values used by the shipped DLL:
        // 129=ENOTSUP (optional/unknown NAL: skip), 104=EBADMSG (a dependent
        // slice can arrive before its reference and is safely skipped by the
        // validated legacy feeder), 11/122/140=temporary would-block variants.
        const bool soft_error =
            result == 11 || result == 104 || result == 122 ||
            result == 129 || result == 140;
        if (result != 0 && !soft_error) {
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

int MVCDecoder::decodeAccessUnitPair(
    const uint8_t* base_data, size_t base_size,
    const uint8_t* dependent_data, size_t dependent_size) {
    if (!decoder_) {
        last_error_ = "Decoder not initialized";
        return EINVAL;
    }
    if (abort_requested_.load(std::memory_order_acquire)) {
        last_error_ = "Decode aborted";
        return ECANCELED;
    }

    // First release the pair completed by the previous access unit. Keeping
    // this pre-drain as well as the post-drain is important on long B-pyramid
    // GOPs: edge264 may only make a delayed picture bumpable at the next call.
    g_edge264.bump_frames(decoder_);
    cacheAvailableFrames();

    // Feed the complete base/dependent pair without draining between its NALs.
    // edge264's MVC state machine pairs both views at this boundary.
    const int base_result = decodeAnnexBStream(base_data, base_size);
    if (base_result != 0) return base_result;
    const int dependent_result =
        decodeAnnexBStream(dependent_data, dependent_size);
    if (dependent_result != 0) return dependent_result;

    // Match the validated feeder: synchronous decoding needs one bump; worker
    // mode gets a few additional scheduling opportunities before draining.
    const int bump_count = worker_threads_ == 0 ? 1 : 5;
    for (int i = 0; i < bump_count; ++i) {
        if (abort_requested_.load(std::memory_order_acquire)) return ECANCELED;
        g_edge264.bump_frames(decoder_);
        cacheAvailableFrames();
    }
    for (int retry = 0;
         retry < 3 && g_edge264.get_busy_tasks(decoder_) != 0;
         ++retry) {
        if (abort_requested_.load(std::memory_order_acquire)) return ECANCELED;
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        cacheAvailableFrames();
    }
    return 0;
}

bool MVCDecoder::getFrame(DecodedMVCFrame& out_frame) {
#ifdef EDGE264_AVAILABLE
    if (!decoder_) {
        last_error_ = "Decoder not initialized";
        return false;
    }

    if (!ready_frames_.empty()) {
        out_frame = std::move(ready_frames_.front());
        ready_frames_.pop_front();
        return true;
    }
    return fetchNativeFrame(out_frame);
#else
    return false;
#endif
}

bool MVCDecoder::fetchNativeFrame(DecodedMVCFrame& out_frame) {
#ifdef EDGE264_AVAILABLE
    Edge264Frame frame{};
    const int result = g_edge264.get_frame(decoder_, &frame, 1);
    // MSVC and MinGW use different errno numbers for ENOMSG.
    if (result == ENOMSG || result == 42 || result == 122) return false;
    if (result != 0) {
        last_error_ = "Failed to get frame: " + std::to_string(result);
        return false;
    }
    const bool converted = convertFrame(frame, out_frame);
    g_edge264.return_frame(decoder_, frame.return_arg);
    return converted;
#else
    return false;
#endif
}

void MVCDecoder::cacheAvailableFrames() {
#ifdef EDGE264_AVAILABLE
    while (ready_frames_.size() < 64) {
        DecodedMVCFrame frame;
        if (!fetchNativeFrame(frame)) break;
        ready_frames_.push_back(std::move(frame));
    }
#endif
}

void MVCDecoder::flush() {
#ifdef EDGE264_AVAILABLE
    if (decoder_) {
        g_edge264.flush(decoder_);
    }
#endif
    ready_frames_.clear();
}

void MVCDecoder::bumpFrames() {
#ifdef EDGE264_AVAILABLE
    if (decoder_) g_edge264.bump_frames(decoder_);
#endif
}

void MVCDecoder::close() {
#ifdef EDGE264_AVAILABLE
    abort_requested_.store(true, std::memory_order_release);
    if (decoder_ && g_edge264.free_decoder) {
        g_edge264.free_decoder(&decoder_);
    }
#endif
    decoder_ = nullptr;
    worker_threads_ = 0;
    ready_frames_.clear();
}

void MVCDecoder::requestAbort() {
    abort_requested_.store(true, std::memory_order_release);
}

void MVCDecoder::clearAbort() {
    abort_requested_.store(false, std::memory_order_release);
}

bool MVCDecoder::runtimeAvailable(std::string* diagnostic) {
#ifdef EDGE264_AVAILABLE
    return loadEdge264(diagnostic);
#else
    if (diagnostic) *diagnostic = "Edge264 support disabled at compile time";
    return false;
#endif
}

#ifdef EDGE264_AVAILABLE
bool MVCDecoder::convertFrame(const Edge264Frame& src, DecodedMVCFrame& dst) {
    if (src.bit_depth_Y != 8 || src.bit_depth_C != 8) {
        last_error_ = "Seuls les flux MVC AVC 8 bits sont pris en charge";
        return false;
    }
    if (!src.samples[0] || !src.samples[1] || !src.samples[2] ||
        src.width_Y <= 0 || src.height_Y <= 0 ||
        src.width_C <= 0 || src.height_C <= 0) {
        last_error_ = "edge264 a retourne une image incomplete";
        return false;
    }

    dst.base_view.width = src.width_Y;
    dst.base_view.height = src.height_Y;
    dst.base_view.chroma_width = src.width_C;
    dst.base_view.chroma_height = src.height_C;
    dst.base_view.stride_y = src.width_Y;
    dst.base_view.stride_c = src.width_C;
    copyPlane(src.samples[0], src.stride_Y, src.width_Y, src.height_Y,
              dst.base_view.y_plane);
    copyPlane(src.samples[1], src.stride_C, src.width_C, src.height_C,
              dst.base_view.cb_plane);
    copyPlane(src.samples[2], src.stride_C, src.width_C, src.height_C,
              dst.base_view.cr_plane);

    dst.has_mvc = (src.samples_mvc[0] != nullptr);
    if (dst.has_mvc) {
        if (!src.samples_mvc[1] || !src.samples_mvc[2]) {
            last_error_ = "edge264 a retourne une vue MVC incomplete";
            return false;
        }
        dst.dependent_view.width = src.width_Y;
        dst.dependent_view.height = src.height_Y;
        dst.dependent_view.chroma_width = src.width_C;
        dst.dependent_view.chroma_height = src.height_C;
        dst.dependent_view.stride_y = src.width_Y;
        dst.dependent_view.stride_c = src.width_C;
        copyPlane(src.samples_mvc[0], src.stride_Y, src.width_Y, src.height_Y,
                  dst.dependent_view.y_plane);
        copyPlane(src.samples_mvc[1], src.stride_C, src.width_C, src.height_C,
                  dst.dependent_view.cb_plane);
        copyPlane(src.samples_mvc[2], src.stride_C, src.width_C, src.height_C,
                  dst.dependent_view.cr_plane);
    } else {
        dst.dependent_view = {};
    }

    dst.frame_id = src.FrameId;
    dst.frame_id_mvc = src.FrameId_mvc;
    dst.picture_order_cnt = src.PictureOrderCnt;
    dst.picture_order_cnt_mvc = src.PictureOrderCnt_mvc;

    // edge264 width_Y/height_Y are already crop-adjusted.
    dst.display_width = src.width_Y;
    dst.display_height = src.height_Y;
    last_error_.clear();
    return true;
}
#endif

} // namespace mvc_demux
