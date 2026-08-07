#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>
#include "mvc_demuxer.h"
#include "mvc_matroska_demuxer.h"
#include "mvc_m2ts_demuxer.h"
#include "mvc_ssif_demuxer.h"
#include "ssif_parser.h"
#include "matroska_reader.h"
#include "mvc_decoder.h"
#include "frame_ring_buffer.h"
#include "native_renderer.h"   // gated internally on SYLC_NATIVE_RENDERER
#include "depth_engine.h"      // DepthEngine itself is unguarded, but its .cpp is only
                                // compiled under BUILD_NATIVE_RENDERER (see CMakeLists.txt);
                                // usage below stays inside #ifdef SYLC_NATIVE_RENDERER
#include "depth_stabilizer.h"  // same gating as depth_engine.h above
#include "cut_gate.h"          // header-only; same gating as depth_engine.h above
#include "shared_depth_service.h"  // synth3d_flow::estimate_flow test surface
#ifdef SYLC_NATIVE_RENDERER
#include "stereo_lab.h"
#ifdef SYLC_NVOF_CUDA
#include "nvof_flow.h"
#endif
#endif

namespace py = pybind11;
using namespace mvc_demux;

// Python-friendly frame data with numpy arrays
struct PyFrameData {
    py::array_t<uint8_t> data;
    uint64_t timestamp;
    bool isKeyframe;

    static PyFrameData fromFrameData(const FrameData& frame) {
        PyFrameData pyFrame;
        py::array_t<uint8_t> arr(frame.data.size());
        if (!frame.data.empty()) {
            std::memcpy(arr.mutable_data(), frame.data.data(), frame.data.size());
        }
        pyFrame.data = std::move(arr);
        pyFrame.timestamp = frame.timestamp;
        pyFrame.isKeyframe = frame.isKeyframe;
        return pyFrame;
    }
};

PYBIND11_MODULE(mvc_demuxer_cpp, m) {
    m.doc() = "MVC Real-time Demuxer - Extracts H.264 base and MVC dependent views from MKV files";

    // Version and build info
    m.def("get_build_info", []() -> py::dict {
        py::dict info;
        info["version"] = "1.0.0";
#ifdef HAVE_LIBMATROSKA
        info["has_libmatroska"] = true;
        info["matroska_support"] = "full";
#else
        info["has_libmatroska"] = false;
        info["matroska_support"] = "fallback";
#endif
        return info;
    }, "Get build information and feature support");

    // Expose StreamType enum
    py::enum_<StreamType>(m, "StreamType")
        .value("Unknown", StreamType::Unknown)
        .value("BaseAVC", StreamType::BaseAVC)
        .value("MVCDependent", StreamType::MVCDependent)
        .export_values();

    // Expose NALUnitType enum
    py::enum_<NALUnitType>(m, "NALUnitType")
        .value("Unspecified", NALUnitType::Unspecified)
        .value("CodedSliceNonIDR", NALUnitType::CodedSliceNonIDR)
        .value("CodedSliceIDR", NALUnitType::CodedSliceIDR)
        .value("SEI", NALUnitType::SEI)
        .value("SPS", NALUnitType::SPS)
        .value("PPS", NALUnitType::PPS)
        .value("SubsetSPS", NALUnitType::SubsetSPS)
        .value("SliceExtension", NALUnitType::SliceExtension)
        .export_values();

    // Expose FrameData
    py::class_<PyFrameData>(m, "FrameData")
        .def(py::init<>())
        .def_readwrite("data", &PyFrameData::data)
        .def_readwrite("timestamp", &PyFrameData::timestamp)
        .def_readwrite("isKeyframe", &PyFrameData::isKeyframe);

    // Zero-copy ring buffer for frame pairs
    py::class_<FrameRingBuffer, std::shared_ptr<FrameRingBuffer>>(m, "FrameRingBuffer")
        .def(py::init<size_t, size_t>(),
             py::arg("capacity") = 120,
             py::arg("max_frame_bytes") = 4 * 1024 * 1024)
        .def_property_readonly("capacity", &FrameRingBuffer::capacity)
        .def_property_readonly("size", &FrameRingBuffer::size)
        .def_property_readonly("dropped", &FrameRingBuffer::dropped)
        .def_property_readonly("max_frame_bytes", &FrameRingBuffer::maxFrameBytes)
        .def("clear", &FrameRingBuffer::clear,
             "Drop every queued access unit (borrowed arrays remain valid)")
        .def("pop",
             [](std::shared_ptr<FrameRingBuffer> self) -> py::tuple {  // explicit return type: clang needs it (MSVC tolerant)
                 FrameBufferView view;
                 size_t slotIndex = 0;
                 if (!self->pop(view, slotIndex)) {
                     return py::make_tuple(false, py::none(), py::none(), 0, false, 0);
                 }

                 auto guard = new FrameRingBuffer::SlotGuard(self, slotIndex);
                 py::capsule capsule(static_cast<void*>(guard), [](void* p) {
                     delete reinterpret_cast<FrameRingBuffer::SlotGuard*>(p);
                 });

                 py::object baseArray;
                 py::object depArray;

                 if (view.basePtr && view.baseSize > 0) {
                     baseArray = py::array_t<uint8_t>(
                         {static_cast<py::ssize_t>(view.baseSize)},
                         {static_cast<py::ssize_t>(1)},
                         view.basePtr,
                         capsule
                     );
                 } else {
                     baseArray = py::array_t<uint8_t>();
                 }

                 if (view.depPtr && view.depSize > 0) {
                     depArray = py::array_t<uint8_t>(
                         {static_cast<py::ssize_t>(view.depSize)},
                         {static_cast<py::ssize_t>(1)},
                         view.depPtr,
                         capsule
                     );
                 } else {
                     depArray = py::array_t<uint8_t>();
                 }

                 return py::make_tuple(true, baseArray, depArray, view.timestamp, view.isKeyframe, view.sequence);
             },
             "Pop the oldest frame without copying. Returns (ok, base, dep, timestamp_ms, is_keyframe, sequence)");

    // Expose VideoInfo
    py::class_<MVCDemuxer::VideoInfo>(m, "VideoInfo")
        .def(py::init<>())
        .def_readwrite("width", &MVCDemuxer::VideoInfo::width)
        .def_readwrite("height", &MVCDemuxer::VideoInfo::height)
        .def_readwrite("fps", &MVCDemuxer::VideoInfo::fps)
        .def_readwrite("hasMVC", &MVCDemuxer::VideoInfo::hasMVC)
        .def_readwrite("trackCount", &MVCDemuxer::VideoInfo::trackCount);

    // Expose MVCDemuxer
    py::class_<MVCDemuxer>(m, "MVCDemuxer")
        .def(py::init<>())
        .def("open", &MVCDemuxer::open,
             "Open an MKV file for demuxing",
             py::arg("file_path"))
        .def("close", &MVCDemuxer::close,
             "Close the file")
        .def("is_open", &MVCDemuxer::isOpen,
             "Check if file is open")
        .def("get_video_info", &MVCDemuxer::getVideoInfo,
             "Get video metadata")
        .def("read_next_frame_pair",
             [](MVCDemuxer& self) -> py::tuple {
                 FrameData baseView, dependentView;
                 bool success = false;
                 {
                     py::gil_scoped_release release;
                     success = self.readNextFramePair(baseView, dependentView);
                 }

                 if (!success) {
                     return py::make_tuple(false, py::none(), py::none());
                 }

                 // Return dict instead of PyFrameData for consistency and PTS access
                 py::dict base;
                 base["data"] = py::array_t<uint8_t>(baseView.data.size(), baseView.data.data());
                 base["timestamp"] = baseView.timestamp;
                 base["isKeyframe"] = baseView.isKeyframe;

                 py::dict dep;
                 dep["data"] = py::array_t<uint8_t>(dependentView.data.size(), dependentView.data.data());
                 dep["timestamp"] = dependentView.timestamp;
                 dep["isKeyframe"] = dependentView.isKeyframe;

                 return py::make_tuple(true, base, dep);
             },
             "Read next frame pair (base + dependent views). Returns (success, base_dict, dependent_dict)")
        .def("read_next_into_ring",
             [](MVCDemuxer& self, FrameRingBuffer& ring) {
                 py::gil_scoped_release release;
                 return self.readNextFramePairIntoRing(ring);
             },
             py::arg("ring_buffer"),
             "Demux next frame pair directly into a C++ ring buffer (zero-copy to Python)")
        .def("set_frame_callback",
             [](MVCDemuxer& self, py::function callback) {
                 self.setFrameCallback([callback](const FrameData& base, const FrameData& dep) {
                     // Convert to Python-friendly format
                     PyFrameData pyBase = PyFrameData::fromFrameData(base);
                     PyFrameData pyDep = PyFrameData::fromFrameData(dep);

                     // Call Python callback
                     py::gil_scoped_acquire acquire;
                     callback(pyBase, pyDep);
                 });
             },
             "Set callback for streaming mode",
             py::arg("callback"))
        .def("process_file",
             [](MVCDemuxer& self) {
                 py::gil_scoped_release release;
                 return self.processFile();
             },
             "Process entire file with callback (streaming mode)")
        .def("seek",
             [](MVCDemuxer& self, uint64_t timestamp_ms) {
                 py::gil_scoped_release release;
                 return self.seek(timestamp_ms);
             },
             "Seek to timestamp in milliseconds",
             py::arg("timestamp_ms"));

    // Expose H264NALParser
    py::class_<H264NALParser>(m, "H264NALParser")
        .def(py::init<>())
        .def("parse_buffer",
             [](H264NALParser& self, py::array_t<uint8_t> buffer) {
                 auto buf_info = buffer.request();
                std::vector<NALUnit> nalUnits;
                {
                    py::gil_scoped_release release;
                    nalUnits = self.parseBuffer(
                        static_cast<const uint8_t*>(buf_info.ptr),
                        buf_info.size
                    );
                }

                 // Convert to Python list of dictionaries
                 py::list result;
                 for (const auto& nal : nalUnits) {
                     py::dict d;
                     d["type"] = static_cast<int>(nal.type);
                     d["streamType"] = static_cast<int>(nal.streamType);
                     d["size"] = nal.size;
                     d["isMVC"] = nal.isMVC;
                     d["spsId"] = nal.spsId;
                     result.append(d);
                 }
                 return result;
             },
             "Parse buffer and extract NAL units",
             py::arg("buffer"))
        .def("get_mvc_sps_ids", &H264NALParser::getMVCSPSIDs,
             "Get MVC SPS IDs that have been detected");

    // Expose Matroska-specific classes (NEW - with libmatroska support)

    // MatroskaTrack
    py::class_<MatroskaTrack>(m, "MatroskaTrack")
        .def(py::init<>())
        .def_readwrite("trackNumber", &MatroskaTrack::trackNumber)
        .def_readwrite("trackUID", &MatroskaTrack::trackUID)
        .def_readwrite("trackType", &MatroskaTrack::trackType)
        .def_readwrite("codecId", &MatroskaTrack::codecId)
        .def_readwrite("pixelWidth", &MatroskaTrack::pixelWidth)
        .def_readwrite("pixelHeight", &MatroskaTrack::pixelHeight)
        .def_readwrite("frameRate", &MatroskaTrack::frameRate)
        .def_readwrite("isMVC", &MatroskaTrack::isMVC)
        .def_readwrite("mvcSubTrack", &MatroskaTrack::mvcSubTrack);

    // MVCMatroskaDemuxer::VideoInfo
    py::class_<MVCMatroskaDemuxer::VideoInfo>(m, "MVCMatroskaVideoInfo")
        .def(py::init<>())
        .def_readwrite("width", &MVCMatroskaDemuxer::VideoInfo::width)
        .def_readwrite("height", &MVCMatroskaDemuxer::VideoInfo::height)
        .def_readwrite("fps", &MVCMatroskaDemuxer::VideoInfo::fps)
        .def_readwrite("hasMVC", &MVCMatroskaDemuxer::VideoInfo::hasMVC)
        .def_readwrite("baseTrackNumber", &MVCMatroskaDemuxer::VideoInfo::baseTrackNumber)
        .def_readwrite("mvcTrackNumber", &MVCMatroskaDemuxer::VideoInfo::mvcTrackNumber);

    // ========== SUBTITLE STREAMING SUPPORT ==========

    // MVCMatroskaDemuxer::SubtitleTrackInfo
    py::class_<MVCMatroskaDemuxer::SubtitleTrackInfo>(m, "SubtitleTrackInfo")
        .def(py::init<>())
        .def_readwrite("trackNumber", &MVCMatroskaDemuxer::SubtitleTrackInfo::trackNumber)
        .def_readwrite("codecId", &MVCMatroskaDemuxer::SubtitleTrackInfo::codecId)
        .def_readwrite("language", &MVCMatroskaDemuxer::SubtitleTrackInfo::language)
        .def_readwrite("name", &MVCMatroskaDemuxer::SubtitleTrackInfo::name)
        .def_readwrite("isPGS", &MVCMatroskaDemuxer::SubtitleTrackInfo::isPGS);

    // MVCMatroskaDemuxer::SubtitleBlock
    py::class_<MVCMatroskaDemuxer::SubtitleBlock>(m, "SubtitleBlock")
        .def(py::init<>())
        .def_readwrite("trackNumber", &MVCMatroskaDemuxer::SubtitleBlock::trackNumber)
        .def_readwrite("timestampMs", &MVCMatroskaDemuxer::SubtitleBlock::timestampMs)
        .def_property("data",
            [](MVCMatroskaDemuxer::SubtitleBlock& self) {
                return py::array_t<uint8_t>(self.data.size(), self.data.data());
            },
            [](MVCMatroskaDemuxer::SubtitleBlock& self, py::array_t<uint8_t> arr) {
                auto buf = arr.request();
                self.data.assign(
                    static_cast<uint8_t*>(buf.ptr),
                    static_cast<uint8_t*>(buf.ptr) + buf.size
                );
            });

    // ================================================

    // MVCMatroskaDemuxer::FramePair
    py::class_<MVCMatroskaDemuxer::FramePair>(m, "MVCFramePair")
        .def(py::init<>())
        .def_property("baseData",
            [](MVCMatroskaDemuxer::FramePair& self) {
                return py::array_t<uint8_t>(self.baseData.size(), self.baseData.data());
            },
            [](MVCMatroskaDemuxer::FramePair& self, py::array_t<uint8_t> arr) {
                auto buf = arr.request();
                self.baseData.assign(
                    static_cast<uint8_t*>(buf.ptr),
                    static_cast<uint8_t*>(buf.ptr) + buf.size
                );
            })
        .def_property("dependentData",
            [](MVCMatroskaDemuxer::FramePair& self) {
                return py::array_t<uint8_t>(self.dependentData.size(), self.dependentData.data());
            },
            [](MVCMatroskaDemuxer::FramePair& self, py::array_t<uint8_t> arr) {
                auto buf = arr.request();
                self.dependentData.assign(
                    static_cast<uint8_t*>(buf.ptr),
                    static_cast<uint8_t*>(buf.ptr) + buf.size
                );
            })
        .def_readwrite("timestamp", &MVCMatroskaDemuxer::FramePair::timestamp)
        .def_readwrite("isKeyframe", &MVCMatroskaDemuxer::FramePair::isKeyframe);

    // MVCMatroskaDemuxer (NEW - Recommended for full MVC support)
    py::class_<MVCMatroskaDemuxer>(m, "MVCMatroskaDemuxer")
        .def(py::init<>())
        .def("open", &MVCMatroskaDemuxer::open,
             "Open an MKV file with full Matroska parsing",
             py::arg("file_path"))
        .def("close", &MVCMatroskaDemuxer::close,
             "Close the file")
        .def("is_open", &MVCMatroskaDemuxer::isOpen,
             "Check if file is open")
        .def("get_video_info", &MVCMatroskaDemuxer::getVideoInfo,
             "Get video metadata (with MVC track detection)")
        .def("get_codec_private",
             [](MVCMatroskaDemuxer& self) -> py::bytes {
                 auto data = self.getCodecPrivate();
                 return py::bytes(reinterpret_cast<const char*>(data.data()), data.size());
             },
             "Get codec private data (SPS/PPS in AVCC format)")
        .def("set_external_duration_ms", &MVCMatroskaDemuxer::set_external_duration_ms,
             py::arg("duration_ms"),
             "Provide external duration hint in milliseconds when container lacks Duration")
        .def("rewind_after_failed_seek_ms", &MVCMatroskaDemuxer::rewind_after_failed_seek_ms,
             py::arg("timestamp_ms"), py::arg("backoff_ms") = 5000,
             "Rewind slightly after a failed seek to recover an IDR frame")
        .def("read_next_frame_pair",
             [](MVCMatroskaDemuxer& self) -> py::tuple {
                 MVCMatroskaDemuxer::FramePair pair;
                bool success = false;
                // V7b STABILITY FIX: Do NOT release GIL here.
                // readNextFramePair may involve allocations or operations that are safer under GIL lock,
                // especially when dealing with shared state or if memoryview/bytes creation happens immediately after.
                // The previous GIL release caused random Access Violations during seek/scan.
                success = self.readNextFramePair(pair);

                if (!success) {
                    return py::make_tuple(false, py::none(), py::none());
                }

                 py::dict base;
                 base["data"] = py::array_t<uint8_t>(pair.baseData.size(), pair.baseData.data());
                 base["timestamp"] = pair.timestamp;
                 base["isKeyframe"] = pair.isKeyframe;

                 py::dict dep;
                 dep["data"] = py::array_t<uint8_t>(pair.dependentData.size(), pair.dependentData.data());
                 dep["timestamp"] = pair.timestamp;
                 dep["isKeyframe"] = pair.isKeyframe;

                 return py::make_tuple(true, base, dep);
             },
             "Read next frame pair (base + dependent). Returns (success, base_dict, dependent_dict)")
        .def("read_next_into_ring",
             [](MVCMatroskaDemuxer& self, FrameRingBuffer& ring) {
                 MVCMatroskaDemuxer::FramePair pair;
                 bool success = false;
                 {
                     py::gil_scoped_release release;
                     success = self.readNextFramePair(pair);
                 }
                 if (!success) {
                     return false;
                 }
                 ring.push(pair.baseData, pair.dependentData, pair.timestamp, pair.isKeyframe);
                 return true;
             },
             py::arg("ring_buffer"),
             "Demux next frame pair into a native ring buffer (zero-copy)")
        .def("seek", &MVCMatroskaDemuxer::seek,
             "Seek to timestamp in milliseconds",
             py::arg("timestamp_ms"))
        .def("getLastCueTimestamp", &MVCMatroskaDemuxer::getLastCueTimestamp,
             "V8 INDEX-BASED SYNC: Get authoritative Cue timestamp from last seek. Returns -1 if unavailable.")
        .def("getCuesTimestamps", &MVCMatroskaDemuxer::getCuesTimestamps,
             "V8 SEEK OPTIMIZATION: Get all keyframe timestamps from Cues index (sorted list in ms)")
        .def("seekToCue", &MVCMatroskaDemuxer::seekToCue,
             py::arg("cue_timestamp_ms"),
             "V8 SEEK OPTIMIZATION: Seek directly to a known Cue timestamp (faster than seek())")
        // ========== SUBTITLE STREAMING METHODS ==========
        .def("get_subtitle_tracks",
             [](MVCMatroskaDemuxer& self) -> py::list {
                 py::list result;
                 for (const auto& track : self.getSubtitleTracks()) {
                     py::dict d;
                     d["trackNumber"] = track.trackNumber;
                     d["codecId"] = track.codecId;
                     d["language"] = track.language;
                     d["name"] = track.name;
                     d["isPGS"] = track.isPGS;
                     result.append(d);
                 }
                 return result;
             },
             "Get all subtitle tracks in the file. Returns list of dicts with trackNumber, codecId, language, name, isPGS.")
        .def("set_subtitle_track", &MVCMatroskaDemuxer::setActiveSubtitleTrack,
             py::arg("track_number"),
             "Enable streaming for a specific subtitle track (0 = disable)")
        .def("get_active_subtitle_track", &MVCMatroskaDemuxer::getActiveSubtitleTrack,
             "Get currently active subtitle track number (0 = none)")
        .def("has_subtitle_data", &MVCMatroskaDemuxer::hasSubtitleData,
             "Check if subtitle data is available in the queue")
        .def("read_subtitle_block",
             [](MVCMatroskaDemuxer& self) -> py::tuple {
                 MVCMatroskaDemuxer::SubtitleBlock block;
                 if (!self.readNextSubtitleBlock(block)) {
                     return py::make_tuple(false, py::none());
                 }
                 py::dict d;
                 d["trackNumber"] = block.trackNumber;
                 d["timestampMs"] = block.timestampMs;
                 d["data"] = py::array_t<uint8_t>(block.data.size(), block.data.data());
                 return py::make_tuple(true, d);
             },
             "Read next subtitle block. Returns (success, dict with trackNumber, timestampMs, data).");
        // ================================================

    // === M2TS DEMUXER (Blu-ray 3D support) ===

    // MVCM2TSDemuxer::VideoInfo
    py::class_<MVCM2TSDemuxer::VideoInfo>(m, "MVCM2TSVideoInfo")
        .def(py::init<>())
        .def_readwrite("width", &MVCM2TSDemuxer::VideoInfo::width)
        .def_readwrite("height", &MVCM2TSDemuxer::VideoInfo::height)
        .def_readwrite("fps", &MVCM2TSDemuxer::VideoInfo::fps)
        .def_readwrite("hasMVC", &MVCM2TSDemuxer::VideoInfo::hasMVC)
        .def_readwrite("baseVideoPid", &MVCM2TSDemuxer::VideoInfo::baseVideoPid)
        .def_readwrite("mvcVideoPid", &MVCM2TSDemuxer::VideoInfo::mvcVideoPid);

    // MVCM2TSDemuxer (NEW - for Blu-ray 3D M2TS files)
    py::class_<MVCM2TSDemuxer>(m, "MVCM2TSDemuxer")
        .def(py::init<>())
        .def("open", &MVCM2TSDemuxer::open,
             "Open an M2TS or TS file (Blu-ray 3D)",
             py::arg("file_path"))
        .def("close", &MVCM2TSDemuxer::close,
             "Close the file")
        .def("is_open", &MVCM2TSDemuxer::isOpen,
             "Check if file is open")
        .def("get_video_info", &MVCM2TSDemuxer::getVideoInfo,
             "Get video metadata (with MVC PID detection)")
        .def("get_codec_private",
             [](MVCM2TSDemuxer& self) -> py::bytes {
                 auto data = self.getCodecPrivate();
                 return py::bytes(reinterpret_cast<const char*>(data.data()), data.size());
             },
             "Get codec private data (SPS/PPS extracted from stream)")
        .def("read_next_frame_pair",
             [](MVCM2TSDemuxer& self) -> py::tuple {
                 MVCM2TSDemuxer::FramePair pair;
                bool success = false;
                {
                    py::gil_scoped_release release;
                    success = self.readNextFramePair(pair);
                }

                if (!success) {
                    return py::make_tuple(false, py::none(), py::none());
                }

                 py::dict base;
                 base["data"] = py::array_t<uint8_t>(pair.baseData.size(), pair.baseData.data());
                 base["timestamp"] = pair.timestamp;
                 base["isKeyframe"] = pair.isKeyframe;

                 py::dict dep;
                 dep["data"] = py::array_t<uint8_t>(pair.dependentData.size(), pair.dependentData.data());
                 dep["timestamp"] = pair.timestamp;
                 dep["isKeyframe"] = pair.isKeyframe;

                 return py::make_tuple(true, base, dep);
             },
             "Read next frame pair from M2TS stream. Returns (success, base_dict, dependent_dict)")
        .def("read_next_into_ring",
             [](MVCM2TSDemuxer& self, FrameRingBuffer& ring) {
                 MVCM2TSDemuxer::FramePair pair;
                 bool success = false;
                 {
                     py::gil_scoped_release release;
                     success = self.readNextFramePair(pair);
                 }
                 if (!success) {
                     return false;
                 }
                 ring.push(pair.baseData, pair.dependentData, pair.timestamp, pair.isKeyframe);
                 return true;
             },
             py::arg("ring_buffer"),
             "Demux next frame pair into a native ring buffer (zero-copy)")
        .def("seek", &MVCM2TSDemuxer::seek,
             "Seek to a nearby clean IDR for a normalized timestamp",
             py::arg("timestamp_ms"))
        .def("set_external_duration_ms",
             &MVCM2TSDemuxer::setExternalDurationMs,
             "Provide media duration for timestamp-to-byte M2TS seeking",
             py::arg("duration_ms"))
        .def("getLastCueTimestamp",
             &MVCM2TSDemuxer::getLastCueTimestamp,
             "Return the normalized IDR timestamp selected by seek")
        .def("request_abort", &MVCM2TSDemuxer::requestAbort,
             "Cooperatively abort an in-flight probe/frame read")
        .def("clear_abort", &MVCM2TSDemuxer::clearAbort,
             "Clear the cooperative abort flag before reuse");

    // === SSIF DEMUXER (Blu-ray 3D with separate streams) ===

    // SSIFParser
    py::class_<SSIFParser>(m, "SSIFParser")
        .def(py::init<>())
        .def("parse", &SSIFParser::parse,
             "Parse an SSIF file",
             py::arg("ssif_path"))
        .def_static("detect_ssif_path", &SSIFParser::detectSSIFPath,
             "Auto-detect SSIF path from M2TS path",
             py::arg("m2ts_path"))
        .def_static("has_ssif", &SSIFParser::hasSSIF,
             "Check if SSIF file exists for given M2TS",
             py::arg("m2ts_path"));

    // MVCSSIFDemuxer::VideoInfo
    py::class_<MVCSSIFDemuxer::VideoInfo>(m, "MVCSSIFVideoInfo")
        .def(py::init<>())
        .def_readwrite("width", &MVCSSIFDemuxer::VideoInfo::width)
        .def_readwrite("height", &MVCSSIFDemuxer::VideoInfo::height)
        .def_readwrite("fps", &MVCSSIFDemuxer::VideoInfo::fps)
        .def_readwrite("hasMVC", &MVCSSIFDemuxer::VideoInfo::hasMVC)
        .def_readwrite("baseVideoPid", &MVCSSIFDemuxer::VideoInfo::baseVideoPid)
        .def_readwrite("mvcVideoPid", &MVCSSIFDemuxer::VideoInfo::mvcVideoPid);

    // MVCSSIFDemuxer (NEW - for Blu-ray 3D with separate M2TS streams)
    py::class_<MVCSSIFDemuxer>(m, "MVCSSIFDemuxer")
        .def(py::init<>())
        .def("open", &MVCSSIFDemuxer::open,
             "Open an SSIF file or M2TS file (auto-detects SSIF)",
             py::arg("file_path"))
        .def("open_dual", &MVCSSIFDemuxer::openDual,
             py::call_guard<py::gil_scoped_release>(),
             "Open a DUAL-SOURCE BD3D pair: base view and dependent view in SEPARATE .m2ts "
             "files (MakeMKV backup with no interleaved .ssif). base_path -> PID 0x1011 video "
             "AUs, dep_path -> PID 0x1012 AUs; base/dep pairs are matched by PTS and delivered "
             "through the identical read_next_frame_pair / seek / get_codec_private / "
             "request_abort interface as the SSIF path (Python pipeline unchanged).",
             py::arg("base_path"), py::arg("dep_path"))
        .def("close", &MVCSSIFDemuxer::close,
             "Close the demuxer")
        .def("get_video_info", &MVCSSIFDemuxer::getVideoInfo,
             "Get video metadata")
        .def("get_codec_private",
             [](MVCSSIFDemuxer& self) -> py::bytes {
                 auto data = self.getCodecPrivate();
                 return py::bytes(reinterpret_cast<const char*>(data.data()), data.size());
             },
             "Get codec private data (SPS/PPS)")
        .def("has_codec_private", &MVCSSIFDemuxer::hasCodecPrivate,
             "Check if codec private data is available")
        .def("read_next_frame_pair",
             [](MVCSSIFDemuxer& self) -> py::tuple {
                 MVCSSIFDemuxer::FramePair pair;
                bool success = false;
                {
                    py::gil_scoped_release release;
                    success = self.readNextFramePair(pair);
                }

                if (!success) {
                    return py::make_tuple(false, py::none(), py::none());
                }

                 py::dict base;
                 base["data"] = py::array_t<uint8_t>(pair.baseData.size(), pair.baseData.data());
                 base["timestamp"] = pair.timestamp;
                 base["isKeyframe"] = pair.isKeyframe;

                 py::dict dep;
                 dep["data"] = py::array_t<uint8_t>(pair.dependentData.size(), pair.dependentData.data());
                 // Review fix DF-2 (finding 1): use the dependent view's OWN PTS (depTimestamp),
                 // not a copy of the base timestamp. Previously this line read `pair.timestamp`,
                 // so base/dep pair-timestamp deltas were structurally 0 (comparing a value to
                 // itself) for BOTH the dual-file and SSIF-streaming paths (they share the same
                 // FramePair producer, tryMatchFramePair()). Consumers checked (mvc_decoder.py
                 // _push_pair_pts / IDR-scan timestamp extraction) only ever read dep['timestamp']
                 // as a fallback when `base` itself is absent/falsy, which never happens for these
                 // demuxer pairs — so this is safe for existing consumers.
                 dep["timestamp"] = pair.depTimestamp;
                 dep["isKeyframe"] = pair.isKeyframe;

                 return py::make_tuple(true, base, dep);
             },
             "Read next frame pair (left + right eyes). Returns (success, base_dict, dependent_dict)")
        .def("read_next_into_ring",
             [](MVCSSIFDemuxer& self, FrameRingBuffer& ring) {
                 MVCSSIFDemuxer::FramePair pair;
                 bool success = false;
                 {
                     py::gil_scoped_release release;
                     success = self.readNextFramePair(pair);
                 }
                 if (!success) {
                     return false;
                 }
                 ring.push(pair.baseData, pair.dependentData, pair.timestamp, pair.isKeyframe);
                 return true;
             },
             py::arg("ring_buffer"),
             "Demux next SSIF frame pair into a native ring buffer (zero-copy)")
        .def("seek", &MVCSSIFDemuxer::seek,
             "Seek to timestamp in milliseconds",
             py::arg("timestamp_ms"))
        .def("set_external_duration_ms", &MVCSSIFDemuxer::setExternalDurationMs,
             "Provide media duration (ms) so seek() can map timestamp -> byte offset",
             py::arg("duration_ms"))
        .def("set_base_seek_table", &MVCSSIFDemuxer::setBaseSeekTable,
             "Provide base EP_map seek table (pts_ms[], byte[]) for exact base seeking",
             py::arg("pts_ms"), py::arg("bytes"))
        .def("set_ssif_seek_table", &MVCSSIFDemuxer::setSsifSeekTable,
             "Provide the BD3D Extent-Start-Point seek map (pts_ms[], ssif_byte[]) for "
             "byte-exact, both-views-aligned streaming seeks",
             py::arg("pts_ms"), py::arg("ssif_bytes"))
        .def("enable_stream_tap", &MVCSSIFDemuxer::enableStreamTap,
             "Tee the raw bytes the active reader already pulls off the disk into a "
             "bounded buffer (drop-oldest), so cast audio can be demuxed WITHOUT a "
             "second reader on the optical head. Returns False if no reader is open.",
             py::arg("capacity_bytes") = size_t(32) * 1024 * 1024)
        .def("disable_stream_tap", &MVCSSIFDemuxer::disableStreamTap,
             "Detach and clear the cast-audio stream tap.")
        .def("read_stream_tap",
             [](MVCSSIFDemuxer& d, size_t maxBytes) {
                 auto v = d.readStreamTap(maxBytes);
                 return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
             },
             "Pop up to max_bytes of teed stream bytes (b'' when none pending).",
             py::arg("max_bytes") = size_t(1) * 1024 * 1024)
        .def("stream_tap_dropped", &MVCSSIFDemuxer::streamTapDropped,
             "Total teed bytes discarded on overflow (drop-oldest) since enable.")
        .def("at_eof", &MVCSSIFDemuxer::atEof,
             "True when the last failed read hit the real end of file (vs a bounded "
             "no-pair scan). Lets the player retry instead of tearing down mid-movie.")
        .def("request_abort", &MVCSSIFDemuxer::requestAbort,
             "Cooperatively abort an in-flight read/scan. Safe to call from another thread "
             "(read_next_* releases the GIL); the read returns early so a slow cold/contended "
             "disc read can never pin the decoder thread into a watchdog force-terminate. "
             "Call before stopping the thread or when a newer seek supersedes the scan.")
        .def("clear_abort", &MVCSSIFDemuxer::clearAbort,
             "Clear the abort flag. Call at the start of each seek/scan.")
        .def("get_subtitle_pids",
             [](MVCSSIFDemuxer& self) -> py::list {
                 py::list result;
                 for (uint16_t pid : self.getSubtitlePids()) result.append(pid);
                 return result;
             },
             "Get PGS subtitle PIDs found in the base PMT (stream_type 0x90)")
        .def("set_subtitle_pid", &MVCSSIFDemuxer::setSubtitlePid,
             "Select a PGS subtitle PID to stream (0 = disable)",
             py::arg("pid"))
        .def("set_subtitle_track", &MVCSSIFDemuxer::setSubtitlePid,
             "Alias of set_subtitle_pid (the decoder calls set_subtitle_track first)",
             py::arg("pid"))
        .def("has_subtitle_data", &MVCSSIFDemuxer::hasSubtitleData,
             "Check if a reassembled PGS subtitle block is queued")
        .def("read_subtitle_block",
             [](MVCSSIFDemuxer& self) -> py::tuple {
                 int64_t ts = 0;
                 std::vector<uint8_t> data;
                 if (!self.readSubtitleBlock(ts, data)) {
                     return py::make_tuple(false, py::none());
                 }
                 py::dict d;
                 d["timestampMs"] = ts;
                 d["data"] = py::array_t<uint8_t>(data.size(), data.data());
                 return py::make_tuple(true, d);
             },
             "Read next PGS subtitle block. Returns (success, dict with timestampMs, data).");

    // === MVC DECODER (edge264 integration) ===

#ifdef EDGE264_AVAILABLE
    // edge264 itself is loaded dynamically by MVCDecoder. The Python extension
    // therefore stays MSVC-only and does not embed/link the MinGW static archive.

    // Decoded view structure
    py::class_<DecodedMVCFrame::View>(m, "DecodedView")
        .def_readonly("width", &DecodedMVCFrame::View::width)
        .def_readonly("height", &DecodedMVCFrame::View::height)
        .def_readonly("chroma_width", &DecodedMVCFrame::View::chroma_width)
        .def_readonly("chroma_height", &DecodedMVCFrame::View::chroma_height)
        .def_readonly("stride_y", &DecodedMVCFrame::View::stride_y)
        .def_readonly("stride_c", &DecodedMVCFrame::View::stride_c)
        .def_property_readonly("y_plane",
            [](py::object self_object) -> py::array_t<uint8_t> {
                const auto& self = self_object.cast<const DecodedMVCFrame::View&>();
                if (self.y_plane.empty()) return py::array_t<uint8_t>();
                return py::array_t<uint8_t>(
                    {static_cast<py::ssize_t>(self.height),
                     static_cast<py::ssize_t>(self.width)},
                    {static_cast<py::ssize_t>(self.stride_y),
                     static_cast<py::ssize_t>(1)},
                    self.y_plane.data(), self_object);
            })
        .def_property_readonly("cb_plane",
            [](py::object self_object) -> py::array_t<uint8_t> {
                const auto& self = self_object.cast<const DecodedMVCFrame::View&>();
                if (self.cb_plane.empty()) return py::array_t<uint8_t>();
                return py::array_t<uint8_t>(
                    {static_cast<py::ssize_t>(self.chroma_height),
                     static_cast<py::ssize_t>(self.chroma_width)},
                    {static_cast<py::ssize_t>(self.stride_c),
                     static_cast<py::ssize_t>(1)},
                    self.cb_plane.data(), self_object);
            })
        .def_property_readonly("cr_plane",
            [](py::object self_object) -> py::array_t<uint8_t> {
                const auto& self = self_object.cast<const DecodedMVCFrame::View&>();
                if (self.cr_plane.empty()) return py::array_t<uint8_t>();
                return py::array_t<uint8_t>(
                    {static_cast<py::ssize_t>(self.chroma_height),
                     static_cast<py::ssize_t>(self.chroma_width)},
                    {static_cast<py::ssize_t>(self.stride_c),
                     static_cast<py::ssize_t>(1)},
                    self.cr_plane.data(), self_object);
            });

    // Decoded MVC frame (both views)
    py::class_<DecodedMVCFrame, std::shared_ptr<DecodedMVCFrame>>(m, "DecodedMVCFrame")
        .def(py::init<>())
        .def_property_readonly(
            "base_view",
            [](DecodedMVCFrame& self) -> DecodedMVCFrame::View& {
                return self.base_view;
            },
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "dependent_view",
            [](DecodedMVCFrame& self) -> DecodedMVCFrame::View& {
                return self.dependent_view;
            },
            py::return_value_policy::reference_internal)
        .def_readonly("has_mvc", &DecodedMVCFrame::has_mvc)
        .def_readonly("frame_id", &DecodedMVCFrame::frame_id)
        .def_readonly("frame_id_mvc", &DecodedMVCFrame::frame_id_mvc)
        .def_readonly("picture_order_cnt", &DecodedMVCFrame::picture_order_cnt)
        .def_readonly("picture_order_cnt_mvc", &DecodedMVCFrame::picture_order_cnt_mvc)
        .def_readonly("display_width", &DecodedMVCFrame::display_width)
        .def_readonly("display_height", &DecodedMVCFrame::display_height);

    // MVC Decoder (using edge264)
    py::class_<MVCDecoder>(m, "MVCDecoder")
        .def(py::init<>())
        .def("init", &MVCDecoder::init,
             "Initialize decoder with specified number of threads (-1 for auto)",
             py::arg("n_threads") = -1)
        .def("decode_nal",
             [](MVCDecoder& self, py::array_t<uint8_t> nal_data) -> int {
                 auto buf = nal_data.request();
                py::gil_scoped_release release;
                return self.decodeNAL(
                    static_cast<const uint8_t*>(buf.ptr),
                    buf.size
                );
             },
             "Decode a NAL unit. Feed both base and dependent NAL units to the same decoder.",
             py::arg("nal_data"))
        .def("decode_annexb_stream",
             [](MVCDecoder& self, py::buffer buffer) -> int {
                 auto info = buffer.request();
                 if (info.itemsize != 1 || info.ndim != 1 || info.strides[0] != 1) {
                     throw py::value_error("Annex B data must be a contiguous byte buffer");
                 }
                 py::gil_scoped_release release;
                 return self.decodeAnnexBStream(
                     static_cast<const uint8_t*>(info.ptr),
                     static_cast<size_t>(info.size)
                 );
             },
             "Decode a full Annex B access unit without copying",
             py::arg("data"))
        .def("decode_access_unit_pair",
             [](MVCDecoder& self, py::buffer base, py::buffer dependent) -> int {
                 auto base_info = base.request();
                 auto dep_info = dependent.request();
                 if (base_info.itemsize != 1 || base_info.ndim != 1 ||
                     base_info.strides[0] != 1 || dep_info.itemsize != 1 ||
                     dep_info.ndim != 1 || dep_info.strides[0] != 1) {
                     throw py::value_error(
                         "MVC access units must be contiguous byte buffers");
                 }
                 py::gil_scoped_release release;
                 return self.decodeAccessUnitPair(
                     static_cast<const uint8_t*>(base_info.ptr),
                     static_cast<size_t>(base_info.size),
                     static_cast<const uint8_t*>(dep_info.ptr),
                     static_cast<size_t>(dep_info.size));
             },
             "Decode a base/dependent MVC access-unit pair without copying",
             py::arg("base"), py::arg("dependent"))
        .def("get_frame",
             [](MVCDecoder& self) -> py::tuple {
                 auto frame = std::make_shared<DecodedMVCFrame>();
                 bool success = false;
                 {
                     py::gil_scoped_release release;
                     success = self.getFrame(*frame);
                 }
                 if (!success) return py::make_tuple(false, py::none());
                 return py::make_tuple(true, py::cast(frame));
             },
             "Get next decoded frame if available. Returns (success, frame)")
        .def("flush", &MVCDecoder::flush,
             py::call_guard<py::gil_scoped_release>(),
             "Flush decoder (for seeking)")
        .def("bump_frames", &MVCDecoder::bumpFrames,
             py::call_guard<py::gil_scoped_release>(),
             "Make all delayed pictures eligible for end-of-stream output")
        .def("request_abort", &MVCDecoder::requestAbort,
             "Request cooperative cancellation at the next safe decode boundary")
        .def("clear_abort", &MVCDecoder::clearAbort,
             "Clear a previous cooperative cancellation request")
        .def("close", &MVCDecoder::close,
             py::call_guard<py::gil_scoped_release>(),
             "Release the edge264 decoder deterministically")
        .def("is_initialized", &MVCDecoder::isInitialized,
             "Check if decoder is initialized")
        .def("get_last_error", &MVCDecoder::getLastError,
             "Get last error message");

    m.def("edge264_runtime_status", []() {
        std::string diagnostic;
        const bool available = MVCDecoder::runtimeAvailable(&diagnostic);
        return py::make_tuple(available, diagnostic);
    }, "Probe edge264.dll and return (available, diagnostic)");
    m.attr("EDGE264_DYNAMIC") = true;
#else
    // Edge264 not available - do NOT expose MVCDecoder
    // Python code will check hasattr(module, 'MVCDecoder') and fallback to ctypes
    m.attr("EDGE264_UNAVAILABLE") = true;
#endif

    // --- Native D3D11 renderer (Tokyo #3), STAGE S1 -------------------------
    // Exposed only when built with SYLC_NATIVE_RENDERER (Windows + d3d11/dxgi).
    // Python checks hasattr(module, 'NativeRenderer') / NATIVE_RENDERER_AVAILABLE.
#ifdef SYLC_NATIVE_RENDERER
    py::class_<sylc::NativeRenderer>(m, "NativeRenderer")
        .def(py::init<>())
        // GIL released around device/swapchain creation and the blocking Present,
        // per the design's GIL discipline (never freeze the UI / starve audio).
        .def("initialize", &sylc::NativeRenderer::initialize,
             py::arg("hwnd"), py::arg("width"), py::arg("height"), py::arg("hdr") = false,
             py::call_guard<py::gil_scoped_release>(),
             "Create the flip-model swapchain on an existing HWND (int). "
             "hdr=False -> 8-bit SDR (G22/gamma, no EOTF; matches Qt on an SDR display); "
             "hdr=True -> FP16 scRGB linear (needs in-shader EOTF + SDR-white scaling).")
        .def("resize", &sylc::NativeRenderer::resize,
             py::arg("width"), py::arg("height"),
             py::call_guard<py::gil_scoped_release>(),
             "ResizeBuffers to the new physical size (flip-model requirement).")
        .def("present", &sylc::NativeRenderer::present,
             py::arg("sync_interval") = 1,
             py::call_guard<py::gil_scoped_release>(),
             "Draw and Present. interval 1 is the timing authority; interval 0 "
             "is intended for a simultaneous non-blocking secondary preview.")
        .def("is_hdr", &sylc::NativeRenderer::is_hdr,
             "True if the scRGB HDR color space was accepted.")
        .def("set_uniforms", &sylc::NativeRenderer::set_uniforms,
             py::arg("stereo_mode"), py::arg("subtitle_enabled"),
             py::arg("rect_x") = 0.f, py::arg("rect_y") = 0.f,
             py::arg("rect_w") = 1.f, py::arg("rect_h") = 1.f,
             py::arg("sdr_white_level") = 1.f, py::arg("output_gamma") = 0.f,
             py::arg("subtitle_disparity") = 0.f,
             "Set shader uniforms (stereo_mode 0=2D/1=framepack/2=SBS/3=TAB; "
             "output_gamma>0 linearizes the gamma-domain RGB before scaling; "
             "subtitle_disparity = stereoscopic overlay depth, normalized eye-width, "
             ">0 = in front of the screen).")
        .def("set_hud_state", &sylc::NativeRenderer::set_hud_state,
             py::arg("enabled"),
             py::arg("rect_x") = 0.f, py::arg("rect_y") = 0.f,
             py::arg("rect_w") = 1.f, py::arg("rect_h") = 1.f,
             py::arg("disparity") = 0.f, py::arg("opacity") = 1.f,
             "Set the independent playback-HUD composition state. The rect is "
             "normalized in one-eye video coordinates and is duplicated by the "
             "shader for SBS/TAB/FramePack.")
        .def("set_source_aspect", &sylc::NativeRenderer::set_source_aspect,
             py::arg("aspect"),
             "C2: display-aspect override (width/height). >0 forces the letterbox/pillarbox "
             "aspect (half-SBS/half-TAB, where the packed frame carries the original 2D "
             "dims); 0.0 = derive from the uploaded eye dimensions (full formats/MVC/2D).")
        .def("set_color_params", &sylc::NativeRenderer::set_color_params,
             py::arg("yuv_matrix_sel"), py::arg("transfer_sel"),
             "HDR10/PQ color selectors (HEVC). yuv_matrix_sel: 0=BT.601 limited (legacy), "
             "1=BT.709 limited, 2=BT.2020nc limited. transfer_sel: 0=legacy gamma/sdr_white, "
             "1=PQ->scRGB absolute (HDR display), 2=PQ->tone-mapped SDR. Both 0 (DEFAULT) is "
             "byte-identical to the pre-HDR path.")
        .def("set_video_time_ms", &sylc::NativeRenderer::set_video_time_ms,
             py::arg("video_time_ms"),
             "Attach the exact media PTS (milliseconds) to the planes uploaded "
             "for the next present. Negative/non-finite selects the fallback "
             "clock; inference cadence is never substituted for video time.")
        .def("set_synth3d_output_eye", &sylc::NativeRenderer::set_synth3d_output_eye,
             py::arg("eye"),
             "Select the synthesized eye exposed by a stereo_mode=0 surface: "
             "0=left, 1=right. Used by Dual Projector eye windows only.")
        .def("set_yuv_frame",
             [](sylc::NativeRenderer& r, py::object yl, py::object ul, py::object vl,
                py::object yr, py::object ur, py::object vr) {
                 r.set_plane_scale(1.0f);   // 8-bit R8: identity scale
                 py::object planes[6] = { yl, ul, vl, yr, ur, vr };
                 bool ok = true;
                 for (int i = 0; i < 6; ++i) {
                     if (planes[i].is_none()) continue;
                     // ZERO-COPY fast path: any uint8 2-D array whose rows are
                     // element-contiguous (strides(1)==1) is uploaded in place —
                     // upload_plane honors the row stride. This covers the
                     // column-sliced views of the packed-stereo (FSBS/FTAB) split,
                     // which the old c_style cast silently deep-copied every frame.
                     if (py::isinstance<py::array>(planes[i])) {
                         auto a = planes[i].cast<py::array>();
                         if (a.ndim() == 2 && a.dtype().is(py::dtype::of<uint8_t>())
                                 && a.strides(1) == 1
                                 && a.strides(0) >= a.shape(1)) {
                             const uint32_t h = static_cast<uint32_t>(a.shape(0));
                             const uint32_t w = static_cast<uint32_t>(a.shape(1));
                             const uint32_t stride = static_cast<uint32_t>(a.strides(0));
                             const auto* data = static_cast<const uint8_t*>(a.data());
                             py::gil_scoped_release nogil;   // memcpy without holding the GIL
                             if (!r.upload_plane(i, data, w, h, stride)) ok = false;
                             continue;
                         }
                     }
                     // Fallback: force a contiguous uint8 copy (previous behavior).
                     auto a = planes[i].cast<py::array_t<uint8_t,
                                  py::array::c_style | py::array::forcecast>>();
                     if (a.ndim() != 2)
                         throw std::runtime_error("YUV plane must be a 2-D uint8 array");
                     const uint32_t h = static_cast<uint32_t>(a.shape(0));
                     const uint32_t w = static_cast<uint32_t>(a.shape(1));
                     const uint32_t stride = static_cast<uint32_t>(a.strides(0));
                     const auto* data = a.data();
                     py::gil_scoped_release nogil;
                     if (!r.upload_plane(i, data, w, h, stride)) ok = false;
                 }
                 return ok;
             },
             py::arg("y_l"), py::arg("u_l"), py::arg("v_l"),
             py::arg("y_r") = py::none(), py::arg("u_r") = py::none(), py::arg("v_r") = py::none(),
             "Upload the 6 YUV planes (uint8 2-D arrays; right planes optional for 2D).")
        .def("set_yuv_frame16",
             [](sylc::NativeRenderer& r, py::object yl, py::object ul, py::object vl,
                py::object yr, py::object ur, py::object vr, float plane_scale) {
                 // 10-bit (yuv420p10le) path: uint16 planes -> R16_UNORM textures.
                 // plane_scale rescales the 10-bit-in-low-bits sample to [0,1] in
                 // the shader before YUV->RGB. Same shape/stride contract as the
                 // 8-bit path, right planes None-capable for 2D.
                 r.set_plane_scale(plane_scale);
                 py::object planes[6] = { yl, ul, vl, yr, ur, vr };
                 bool ok = true;
                 for (int i = 0; i < 6; ++i) {
                     if (planes[i].is_none()) continue;
                     // ZERO-COPY fast path: uint16 2-D array whose rows are
                     // element-contiguous (strides(1)==2) is uploaded in place;
                     // upload_plane16 honors the (byte) row stride.
                     if (py::isinstance<py::array>(planes[i])) {
                         auto a = planes[i].cast<py::array>();
                         if (a.ndim() == 2 && a.dtype().is(py::dtype::of<uint16_t>())
                                 && a.strides(1) == static_cast<py::ssize_t>(sizeof(uint16_t))
                                 && a.strides(0) >= a.shape(1) * static_cast<py::ssize_t>(sizeof(uint16_t))) {
                             const uint32_t h = static_cast<uint32_t>(a.shape(0));
                             const uint32_t w = static_cast<uint32_t>(a.shape(1));
                             const uint32_t stride = static_cast<uint32_t>(a.strides(0)); // bytes
                             const auto* data = static_cast<const uint16_t*>(a.data());
                             py::gil_scoped_release nogil;   // memcpy without holding the GIL
                             if (!r.upload_plane16(i, data, w, h, stride)) ok = false;
                             continue;
                         }
                     }
                     // Fallback: force a contiguous uint16 copy.
                     auto a = planes[i].cast<py::array_t<uint16_t,
                                  py::array::c_style | py::array::forcecast>>();
                     if (a.ndim() != 2)
                         throw std::runtime_error("YUV16 plane must be a 2-D uint16 array");
                     const uint32_t h = static_cast<uint32_t>(a.shape(0));
                     const uint32_t w = static_cast<uint32_t>(a.shape(1));
                     const uint32_t stride = static_cast<uint32_t>(a.strides(0)); // bytes (c_style => w*2)
                     const auto* data = a.data();
                     py::gil_scoped_release nogil;
                     if (!r.upload_plane16(i, data, w, h, stride)) ok = false;
                 }
                 return ok;
             },
             py::arg("y_l"), py::arg("u_l"), py::arg("v_l"),
             py::arg("y_r") = py::none(), py::arg("u_r") = py::none(), py::arg("v_r") = py::none(),
             py::arg("plane_scale") = 1.0f,
             "Upload the 6 YUV planes as uint16 2-D arrays (10-bit in low bits) into "
             "R16_UNORM textures; right planes optional for 2D. plane_scale multiplies "
             "each sample before YUV->RGB (65535/1023 ~= 64.06 for yuv420p10le).")
        .def("set_subtitle_rgba",
             [](sylc::NativeRenderer& r,
                py::array_t<uint8_t, py::array::c_style | py::array::forcecast> a) {
                 if (a.ndim() != 3 || a.shape(2) != 4)
                     throw std::runtime_error("subtitle must be HxWx4 uint8 RGBA");
                 const uint32_t h = static_cast<uint32_t>(a.shape(0));
                 const uint32_t w = static_cast<uint32_t>(a.shape(1));
                 const uint32_t stride = static_cast<uint32_t>(a.strides(0));
                 return r.upload_subtitle(a.data(), w, h, stride);
             },
             py::arg("rgba"), "Upload an HxWx4 uint8 RGBA subtitle overlay.")
        .def("set_hud_rgba",
             [](sylc::NativeRenderer& r,
                py::array_t<uint8_t, py::array::c_style | py::array::forcecast> a) {
                 if (a.ndim() != 3 || a.shape(2) != 4)
                     throw std::runtime_error("HUD must be HxWx4 uint8 RGBA");
                 const uint32_t h = static_cast<uint32_t>(a.shape(0));
                 const uint32_t w = static_cast<uint32_t>(a.shape(1));
                 const uint32_t stride = static_cast<uint32_t>(a.strides(0));
                 return r.upload_hud(a.data(), w, h, stride);
             },
             py::arg("rgba"), "Upload an HxWx4 uint8 RGBA playback HUD.")
        .def("clear_frame", &sylc::NativeRenderer::clear_frame,
             "Forget the current frame (present() falls back to black).")
        .def("pause", &sylc::NativeRenderer::pause,
             py::call_guard<py::gil_scoped_release>(),
             "Seek/pause gate: present() holds the last frame (no GPU work).")
        .def("resume", &sylc::NativeRenderer::resume,
             py::call_guard<py::gil_scoped_release>(),
             "Resume presenting.")
        .def("is_paused", &sylc::NativeRenderer::is_paused)
        .def("backend_info", &sylc::NativeRenderer::backend_info)
        .def("last_error", &sylc::NativeRenderer::last_error)
        .def("shutdown", &sylc::NativeRenderer::shutdown)
        // --- SyLC Cast (PC sender): NV12-SBS pack + NVENC HEVC encode ----------
        .def("cast_available", &sylc::NativeRenderer::cast_available,
             "True if NVENC (nvEncodeAPI64.dll) is present. Safe on non-NVIDIA hosts "
             "(returns False, no crash).")
        .def("cast_start", &sylc::NativeRenderer::cast_start,
             py::arg("mode"), py::arg("fps"), py::arg("bitrate_bps"),
             py::arg("main10") = false,
             py::call_guard<py::gil_scoped_release>(),
             "Build the cast pipeline (Task-2 NV12-SBS packer + Task-1 NVENC encoder) on "
             "the renderer's D3D11 device. mode='lossless' (bit-exact) or 'cbr'; "
             "bitrate_bps applies to CBR only. main10=True encodes HEVC Main10 from a "
             "P010 pack with BT.2020/ST2084 (PQ) VUI signalling — the HDR cast path. "
             "Returns False on error (see last_error()).")
        .def("cast_encode",
             [](sylc::NativeRenderer& r, int64_t pts_ms, bool force_idr) -> py::list {
                 std::vector<std::vector<uint8_t>> pkts;
                 {
                     py::gil_scoped_release nogil;   // GPU pack + NVENC encode off the GIL
                     pkts = r.cast_encode(pts_ms, force_idr);
                 }
                 py::list out;
                 for (const auto& p : pkts)
                     out.append(py::bytes(reinterpret_cast<const char*>(p.data()), p.size()));
                 return out;
             },
             py::arg("pts_ms"), py::arg("force_idr"),
             "Pack the last-uploaded YUV planes (set_yuv_frame) into NV12 SBS and NVENC-encode "
             "ONE frame. Returns a list of HEVC Annex-B packets (bytes); empty on error.")
        .def("cast_reconfigure", &sylc::NativeRenderer::cast_reconfigure,
             py::arg("mode"), py::arg("bitrate_bps"),
             py::call_guard<py::gil_scoped_release>(),
             "Hot-change the running cast encoder to a new bitrate/mode mid-stream. A same-mode "
             "bitrate change is seamless (NvEncReconfigureEncoder, no teardown); a mode switch "
             "the driver won't reconfigure falls back to an encoder-only reopen (packer/texture "
             "stay). The next encoded frame is an IDR. Returns False on error (see last_error()).")
        .def("cast_stop", &sylc::NativeRenderer::cast_stop,
             py::call_guard<py::gil_scoped_release>(),
             "Flush + tear down the cast pipeline. Idempotent.")
        // --- synth3d (2D->3D): AI depth + DIBR warp ----------------------------
        .def("set_synth3d",
              [](sylc::NativeRenderer& r, bool enabled, float strength_pct, float convergence,
                 bool depth_view, const std::wstring& model_path, const std::wstring& ort_dir,
                 bool diagnostics, int side, int grid_width, int grid_height,
                 float crop_top, float crop_bottom, bool auto_convergence,
                 bool temporal_fill, bool stereo_lab, bool comfort_enabled,
                 float comfort_soft_pct, float comfort_hard_pct) {
                  if (side <= 0)
                      throw std::runtime_error("side must be a positive inference grid");
                  if ((grid_width > 0) != (grid_height > 0))
                      throw std::runtime_error(
                          "grid_width and grid_height must be provided together");
                  py::gil_scoped_release nogil;   // shader compile + resource creation
                  return r.set_synth3d(enabled, strength_pct, convergence, depth_view,
                                       model_path, ort_dir, diagnostics, side,
                                       grid_width, grid_height, crop_top, crop_bottom,
                                       auto_convergence, temporal_fill, stereo_lab,
                                       comfort_enabled, comfort_soft_pct,
                                       comfort_hard_pct);
             },
             py::arg("enabled"), py::arg("strength_pct") = 1.5f,
             py::arg("convergence") = 0.5f, py::arg("depth_view") = false,
              py::arg("model_path") = std::wstring(), py::arg("ort_dir") = std::wstring(),
              py::arg("diagnostics") = false, py::arg("side") = kDefaultDepthSide,
              py::arg("grid_width") = 0, py::arg("grid_height") = 0,
              py::arg("crop_top") = 0.0f, py::arg("crop_bottom") = 0.0f,
              py::arg("auto_convergence") = false,
              py::arg("temporal_fill") = false,
              py::arg("stereo_lab") = true,
              py::arg("comfort_enabled") = false,
              py::arg("comfort_soft_pct") = 0.0f,
              py::arg("comfort_hard_pct") = 0.0f,
             "Enable the real-time 2D->3D conversion. strength_pct = max disparity as a "
             "% of image width; convergence = normalized nearness at zero parallax (0..1); "
             "depth_view replaces the warp with a false-color depth visualization; "
             "diagnostics overlays depth and disocclusion confidence on the movie; "
              "side = the backward-compatible square grid; positive grid_width and "
              "grid_height override it for a fixed rectangular export; crop_top and "
              "crop_bottom remove normalized encoded mattes before inference; "
              "stereo_lab adds a reversible final coherence pass above the "
              "immutable v5.2.1c raw pair; the calibrated comfort envelope "
              "remains bypassed. "
             "Renderers share one asynchronous ORT service, so playback never waits "
             "for the model. False on error (see last_error()).")
        .def("synth3d_set_lookahead_frame",
             [](sylc::NativeRenderer& r,
                py::array y, py::array u, py::array v,
                py::array_t<float,
                    py::array::c_style | py::array::forcecast> flow_x,
                py::array_t<float,
                    py::array::c_style | py::array::forcecast> flow_y,
                py::array_t<float,
                    py::array::c_style | py::array::forcecast> flow_q,
                double current_pts_ms, double future_pts_ms,
                float plane_scale) {
                 if (y.ndim() != 2 || u.ndim() != 2 || v.ndim() != 2 ||
                     flow_x.ndim() != 2 || flow_y.ndim() != 2 ||
                     flow_q.ndim() != 2) {
                     throw std::runtime_error(
                         "lookahead YUV and flow channels must be 2-D arrays");
                 }
                 const bool is8 = y.dtype().is(py::dtype::of<uint8_t>());
                 const bool is16 = y.dtype().is(py::dtype::of<uint16_t>());
                 if ((!is8 && !is16) || !u.dtype().is(y.dtype()) ||
                     !v.dtype().is(y.dtype())) {
                     throw std::runtime_error(
                         "lookahead Y/U/V must share uint8 or uint16 dtype");
                 }
                 const py::ssize_t bytes = is16 ? 2 : 1;
                 auto plane_ok = [bytes](const py::array& a) {
                     return a.shape(0) > 0 && a.shape(1) > 0 &&
                            a.strides(1) == bytes &&
                            a.strides(0) >= a.shape(1) * bytes;
                 };
                 if (!plane_ok(y) || !plane_ok(u) || !plane_ok(v) ||
                     u.shape(0) != v.shape(0) || u.shape(1) != v.shape(1)) {
                     throw std::runtime_error(
                         "lookahead planes must have element-contiguous rows; U/V shapes must match");
                 }
                 if (flow_y.shape(0) != flow_x.shape(0) ||
                     flow_y.shape(1) != flow_x.shape(1) ||
                     flow_q.shape(0) != flow_x.shape(0) ||
                     flow_q.shape(1) != flow_x.shape(1) ||
                     flow_x.shape(0) <= 0 || flow_x.shape(1) <= 0) {
                     throw std::runtime_error(
                         "lookahead flow x/y/reliability shapes must match");
                 }
                 const uint32_t flow_h = static_cast<uint32_t>(flow_x.shape(0));
                 const uint32_t flow_w = static_cast<uint32_t>(flow_x.shape(1));
                 const size_t flow_count = static_cast<size_t>(flow_w) * flow_h;
                 bool ok = false;
                 {
                     py::gil_scoped_release nogil;
                     ok = r.synth3d_set_lookahead_frame(
                         y.data(), static_cast<uint32_t>(y.shape(1)),
                         static_cast<uint32_t>(y.shape(0)),
                         static_cast<uint32_t>(y.strides(0)),
                         u.data(), static_cast<uint32_t>(u.shape(1)),
                         static_cast<uint32_t>(u.shape(0)),
                         static_cast<uint32_t>(u.strides(0)),
                         v.data(), static_cast<uint32_t>(v.shape(1)),
                         static_cast<uint32_t>(v.shape(0)),
                         static_cast<uint32_t>(v.strides(0)),
                         static_cast<int>(bytes), plane_scale,
                         flow_x.data(), flow_y.data(), flow_q.data(),
                         flow_w, flow_h, flow_count,
                         current_pts_ms, future_pts_ms);
                 }
                 return ok;
             },
             py::arg("future_y"), py::arg("future_u"), py::arg("future_v"),
             py::arg("flow_x"), py::arg("flow_y"),
             py::arg("flow_reliability"),
             py::arg("current_pts_ms"), py::arg("future_pts_ms"),
             py::arg("plane_scale") = 1.0f,
             "Upload one future YUV frame plus current->future NVOFA flow. "
             "The next synth3d present may use it only inside DIBR holes; "
             "the evidence is copied and consumed once.")
        .def("synth3d_clear_lookahead",
             &sylc::NativeRenderer::synth3d_clear_lookahead,
             py::call_guard<py::gil_scoped_release>(),
             "Discard pending future-frame evidence (seek/cut/format reset).")
        .def("synth3d_status", &sylc::NativeRenderer::synth3d_status,
             py::call_guard<py::gil_scoped_release>(),
             "One line with state/provider/fps, map age, shared-client and cut counts, "
             "motion/adaptive-alpha/scene metrics, aspect certification, and err. State "
             "reports renderer-local GPU failures before the shared depth engine; a "
             "debug test depth drives the warp independently of it.")
        .def("synth3d_notify_seek", &sylc::NativeRenderer::synth3d_notify_seek,
             py::call_guard<py::gil_scoped_release>(),
             "Tell synth3d a seek happened: the temporal depth filter re-primes on the "
             "next inference instead of blending across the discontinuity. No-op when "
             "synth3d has never been enabled.")
        .def("synth3d_set_ramp_ms", &sylc::NativeRenderer::synth3d_set_ramp_ms,
             py::arg("ramp_ms"), py::call_guard<py::gil_scoped_release>(),
             "Debug: override the post-cut/seek ease-out ramp duration in ms "
             "(default 300). 0 disables the ramp (full disparity always applied). "
             "No-op when synth3d has never been enabled.")
        .def("synth3d_set_test_depth",
             [](sylc::NativeRenderer& r, py::object depth) {
                 if (depth.is_none()) {
                     py::gil_scoped_release nogil;
                     r.synth3d_set_test_depth(nullptr, 0);
                     return;
                 }
                 auto a = depth.cast<py::array_t<uint16_t,
                              py::array::c_style | py::array::forcecast>>();
                 // Shape sanity only: the grid this must match is the
                 // renderer's LIVE one (it follows the depth preset, and a
                 // 756-sized map on a 900 grid used to be read past its end),
                 // and only synth3d_set_test_depth can compare against it
                 // under the same lock that performs the copy.
                  if (a.ndim() != 2)
                      throw std::runtime_error(
                          "test depth must be a 2-D uint16 array");
                 const uint16_t* p = a.data();
                 const size_t count =
                     static_cast<size_t>(a.shape(0)) * a.shape(1);
                 bool ok = false;
                  int live_w = 0, live_h = 0;
                 {
                     py::gil_scoped_release nogil;
                     ok = r.synth3d_set_test_depth(p, count);
                      if (!ok) {
                          live_w = r.synth3d_grid_width();
                          live_h = r.synth3d_grid_height();
                      }
                 }
                 if (!ok)
                     throw std::runtime_error(
                         "test depth is " + std::to_string(a.shape(0)) + "x" +
                         std::to_string(a.shape(1)) +
                         " but the renderer's inference grid is " +
                          std::to_string(live_h) + "x" + std::to_string(live_w));
             },
             py::arg("depth"),
              "Debug: arm a rectangular uint16 nearness map that BYPASSES inference (None "
             "returns to the engine). It must match the renderer's live inference "
             "grid exactly (see synth3d_grid()); any other shape raises. Copied "
             "internally; the array is not retained.")
        .def("synth3d_set_test_geometry",
             [](sylc::NativeRenderer& r,
                py::array_t<uint16_t,
                    py::array::c_style | py::array::forcecast> depth,
                py::array_t<uint16_t,
                    py::array::c_style | py::array::forcecast> owned,
                py::array_t<uint16_t,
                    py::array::c_style | py::array::forcecast> safety,
                py::array_t<uint16_t,
                    py::array::c_style | py::array::forcecast> ownership) {
                 if (depth.ndim() != 2 || owned.ndim() != 2 ||
                     safety.ndim() != 2 || ownership.ndim() != 2 ||
                     owned.shape(0) != depth.shape(0) ||
                     owned.shape(1) != depth.shape(1) ||
                     safety.shape(0) != depth.shape(0) ||
                     safety.shape(1) != depth.shape(1) ||
                     ownership.shape(0) != depth.shape(0) ||
                     ownership.shape(1) != depth.shape(1)) {
                     throw std::runtime_error(
                         "test geometry channels must be equal-shape 2-D uint16 arrays");
                 }
                 const size_t count = static_cast<size_t>(depth.shape(0)) *
                                      depth.shape(1);
                 bool ok = false;
                 {
                     py::gil_scoped_release nogil;
                     ok = r.synth3d_set_test_geometry(
                         depth.data(), owned.data(), safety.data(),
                         ownership.data(), count);
                 }
                 if (!ok)
                     throw std::runtime_error(
                         "test geometry does not match the renderer's live grid");
             },
             py::arg("depth"), py::arg("owned_depth"),
             py::arg("safety"), py::arg("ownership"),
             "Debug: inject the four production ownership-analysis channels "
             "while bypassing inference. They are precomposed to the live "
             "RG16 renderer map; all arrays must match its grid.")
        .def("synth3d_set_human_matte",
             [](sylc::NativeRenderer& r, py::object matte,
                const std::string& mode, py::object reliability) -> bool {
                 int mode_id = 0;
                 if (mode == "guard") mode_id = 1;
                 else if (mode == "contour") mode_id = 2;
                 else throw std::runtime_error(
                     "matte mode must be 'guard' or 'contour'");

                 if (matte.is_none()) {
                     py::gil_scoped_release nogil;
                     return r.synth3d_set_test_matte(nullptr, 0, 0, 0, 0);
                 }
                 auto a = matte.cast<py::array_t<uint8_t,
                              py::array::c_style | py::array::forcecast>>();
                 if (a.ndim() != 2 || a.shape(0) <= 0 || a.shape(1) <= 0)
                     throw std::runtime_error(
                         "test matte must be a non-empty 2-D uint8 alpha array");
                 const auto height = static_cast<uint32_t>(a.shape(0));
                 const auto width = static_cast<uint32_t>(a.shape(1));
                 const size_t count = static_cast<size_t>(width) * height;
                 bool ok = false;
                 if (reliability.is_none()) {
                     py::gil_scoped_release nogil;
                     ok = r.synth3d_set_test_matte(
                         a.data(), width, height, count, mode_id);
                 } else {
                     auto local = reliability.cast<py::array_t<uint8_t,
                         py::array::c_style | py::array::forcecast>>();
                     if (local.ndim() != 2 ||
                         local.shape(0) != a.shape(0) ||
                         local.shape(1) != a.shape(1)) {
                         throw std::runtime_error(
                             "matte reliability must match the alpha shape");
                     }
                     py::gil_scoped_release nogil;
                     ok = r.synth3d_set_test_matte(
                         a.data(), width, height, count, mode_id,
                         local.data(), count);
                 }
                 if (!ok)
                     throw std::runtime_error(
                          "human matte dimensions or mode are invalid for D3D11");
                 return true;
              },
              py::arg("matte"), py::arg("mode") = "contour",
              py::arg("reliability") = py::none(),
              "Attach an asynchronously generated uint8 human alpha matte. "
              "'guard' uses it only for layer ownership, hole-fill veto and local "
              "stereo safety; 'contour' also decontaminates fractional foreground "
              "pixels and recomposes their immediate background. The matte may be "
              "lower resolution than the video because sampling uses normalized UV. "
              "None disables all matte behavior and restores the historical path.")
        .def("synth3d_set_test_matte",
             [](sylc::NativeRenderer& r, py::object matte,
                const std::string& mode, py::object reliability) -> bool {
                 int mode_id = 0;
                 if (mode == "guard") mode_id = 1;
                 else if (mode == "contour") mode_id = 2;
                 else throw std::runtime_error(
                     "matte mode must be 'guard' or 'contour'");

                 if (matte.is_none()) {
                     py::gil_scoped_release nogil;
                     return r.synth3d_set_test_matte(nullptr, 0, 0, 0, 0);
                 }
                 auto a = matte.cast<py::array_t<uint8_t,
                              py::array::c_style | py::array::forcecast>>();
                 if (a.ndim() != 2 || a.shape(0) <= 0 || a.shape(1) <= 0)
                     throw std::runtime_error(
                         "test matte must be a non-empty 2-D uint8 alpha array");
                 const auto height = static_cast<uint32_t>(a.shape(0));
                 const auto width = static_cast<uint32_t>(a.shape(1));
                 const size_t count = static_cast<size_t>(width) * height;
                 bool ok = false;
                 if (reliability.is_none()) {
                     py::gil_scoped_release nogil;
                     ok = r.synth3d_set_test_matte(
                         a.data(), width, height, count, mode_id);
                 } else {
                     auto local = reliability.cast<py::array_t<uint8_t,
                         py::array::c_style | py::array::forcecast>>();
                     if (local.ndim() != 2 ||
                         local.shape(0) != a.shape(0) ||
                         local.shape(1) != a.shape(1)) {
                         throw std::runtime_error(
                             "matte reliability must match the alpha shape");
                     }
                     py::gil_scoped_release nogil;
                     ok = r.synth3d_set_test_matte(
                         a.data(), width, height, count, mode_id,
                         local.data(), count);
                 }
                 if (!ok)
                     throw std::runtime_error(
                          "test matte dimensions or mode are invalid for D3D11");
                 return true;
             },
             py::arg("matte"), py::arg("mode") = "guard",
             py::arg("reliability") = py::none(),
             "Compatibility alias retained for the offline A/B prototype.")
        .def("synth3d_side", &sylc::NativeRenderer::synth3d_side,
             py::call_guard<py::gil_scoped_release>(),
             "Backward-compatible horizontal inference-grid dimension. Use "
             "synth3d_grid() for rectangular shapes. 0 before synth3d has ever "
             "been enabled.")
        .def("synth3d_grid",
             [](const sylc::NativeRenderer& r) {
                 int width = 0, height = 0;
                 {
                     py::gil_scoped_release nogil;
                     width = r.synth3d_grid_width();
                     height = r.synth3d_grid_height();
                 }
                 return py::make_tuple(width, height);
             },
             "Return the live synth3d inference grid as (width, height).")
        .def("synth3d_set_lookahead", &sylc::NativeRenderer::synth3d_set_lookahead,
             py::arg("cut_in_ms"), py::arg("storm_in_ms"),
             py::arg("cut_pts_ms") = -1.0,
             "Two-filter look-ahead advisory: delays (ms) from the presented "
             "position to the next cut / motion-storm onset observed in the "
             "decoded future. Slightly NEGATIVE values (hold window, ~-120ms) "
             "mean the event just landed and its effect must persist; pass "
             "-1e9 for none (NOT -1: that reads as 'cut one ms ago'). "
             "cut_pts_ms is the ABSOLUTE media PTS of the same reported cut "
             "(shot identity for the cross-shot gate; -1.0 = none — an "
             "absolute PTS is always >= 0, unlike the relative delays). False "
             "when synth3d is off. SYLC_LOOKAHEAD=0 disables the intake.")
        .def("synth3d_set_motion_hints",
             [](sylc::NativeRenderer& r, double pts_ms, double frame_ms,
                int blocks_w, int blocks_h,
                int source_width, int source_height,
                py::array_t<int16_t,
                    py::array::c_style | py::array::forcecast> mv_xy,
                py::object valid_obj) {
                 const size_t blocks =
                     static_cast<size_t>(blocks_w) * blocks_h;
                 if (mv_xy.ndim() != 1 ||
                     static_cast<size_t>(mv_xy.shape(0)) != 2 * blocks)
                     throw std::runtime_error(
                         "mv_xy must be a flat int16 array of 2*blocks");
                 std::vector<int16_t> mv(mv_xy.data(),
                                         mv_xy.data() + 2 * blocks);
                 std::vector<uint8_t> valid;
                 if (!valid_obj.is_none()) {
                     auto v = valid_obj.cast<py::array_t<uint8_t,
                         py::array::c_style | py::array::forcecast>>();
                     if (v.ndim() != 1 ||
                         static_cast<size_t>(v.shape(0)) != blocks)
                         throw std::runtime_error(
                             "valid must be a flat uint8 array of blocks");
                     valid.assign(v.data(), v.data() + blocks);
                 }
                 py::gil_scoped_release nogil;
                 return r.synth3d_set_motion_hints(
                     pts_ms, frame_ms, blocks_w, blocks_h,
                     source_width, source_height,
                     std::move(mv), std::move(valid));
             },
             py::arg("pts_ms"), py::arg("frame_ms"),
             py::arg("blocks_w"), py::arg("blocks_h"),
             py::arg("source_width"), py::arg("source_height"),
             py::arg("mv_xy"), py::arg("valid") = py::none(),
             "Phase 1 (04/08): decoder motion hints for ONE frame, keyed by "
             "media pts. Quarter-pel per display frame, production flow "
             "convention (cur(p) ~ prev(p - flow)); valid=0 marks intra/"
             "absent blocks. Forwarded to the depth service ring; the worker "
             "matches by pts and feeds fuse_bidirectional's third candidate. "
             "SYLC_SYNTH3D_MV_HINTS=0 disables the intake.")
        .def("synth3d_read_plane",
             [](sylc::NativeRenderer& r, int slot) -> py::object {
                 std::vector<uint8_t> buf;
                 uint32_t w = 0, h = 0, bpp = 0;
                 std::string err;
                 bool ok;
                 {
                     py::gil_scoped_release nogil;   // CopyResource + blocking Map
                     ok = r.synth3d_read_plane(slot, buf, w, h, bpp, err);
                 }
                 if (!ok) throw std::runtime_error("synth3d_read_plane: " + err);
                 const py::ssize_t sh = static_cast<py::ssize_t>(h);
                 const py::ssize_t sw = static_cast<py::ssize_t>(w);
                 if (bpp == 2) {
                     py::array_t<uint16_t> out({sh, sw});
                     std::memcpy(out.mutable_data(), buf.data(), buf.size());
                     return std::move(out);
                 }
                 py::array_t<uint8_t> out({sh, sw});
                 std::memcpy(out.mutable_data(), buf.data(), buf.size());
                 return std::move(out);
             },
             py::arg("slot"),
             "Debug: read back one synth3d warp output (slot 0..5 = Y_L,U_L,V_L,Y_R,U_R,"
             "V_R) as a 2-D numpy array (uint8 for R8 planes, uint16 for R16).")
        .def("synth3d_read_plate",
             [](sylc::NativeRenderer& r) -> py::object {
                 std::vector<uint8_t> buf;
                 uint32_t w = 0, h = 0;
                 std::string err;
                 bool ok;
                 {
                     py::gil_scoped_release nogil;   // CopyResource + blocking Map
                     ok = r.synth3d_read_plate(buf, w, h, err);
                 }
                 if (!ok) throw std::runtime_error("synth3d_read_plate: " + err);
                 const py::ssize_t sh = static_cast<py::ssize_t>(h);
                 const py::ssize_t sw = static_cast<py::ssize_t>(w);
                 py::array_t<uint16_t> out({sh, sw, static_cast<py::ssize_t>(4)});
                 std::memcpy(out.mutable_data(), buf.data(), buf.size());
                 return std::move(out);
             },
             "Debug (round 5a): the temporal background plate as (H, W, 4) uint16 "
             "(Y, U, V, confidence in plane space). Raises until temporal_fill ran.");

    // --- synth3d: optional NVIDIA Optical Flow Accelerator ---------------------
#ifdef SYLC_NVOF_CUDA
    py::class_<sylc::NvofFlow>(m, "NvofFlow")
        .def(py::init([](int width, int height, const std::string& perf,
                         int grid) {
            auto flow = std::make_unique<sylc::NvofFlow>();
            std::string error;
            if (!flow->initialize(width, height, perf, grid, error))
                throw std::runtime_error("NvofFlow init: " + error);
            return flow;
        }), py::arg("width"), py::arg("height"),
            py::arg("perf") = "medium", py::arg("grid") = 4)
        .def("estimate",
             [](sylc::NvofFlow& self,
                py::array_t<uint8_t,
                    py::array::c_style | py::array::forcecast> input,
                py::array_t<uint8_t,
                    py::array::c_style | py::array::forcecast> reference) {
                 if (input.ndim() != 2 || reference.ndim() != 2 ||
                     input.shape(1) != self.width() ||
                     input.shape(0) != self.height() ||
                     reference.shape(1) != self.width() ||
                     reference.shape(0) != self.height())
                     throw std::runtime_error(
                         "input/reference must match the NvofFlow HxW grid");
                 std::vector<float> fx, fy, reliability;
                 std::string error;
                 bool ok = false;
                 {
                     py::gil_scoped_release nogil;
                     ok = self.estimate(
                         input.data(), static_cast<int>(input.strides(0)),
                         reference.data(),
                         static_cast<int>(reference.strides(0)),
                         fx, fy, reliability, error);
                 }
                 if (!ok) throw std::runtime_error(
                     "NvofFlow estimate: " + error);
                 py::array_t<float> x({self.output_height(),
                                       self.output_width()});
                 py::array_t<float> y({self.output_height(),
                                       self.output_width()});
                 py::array_t<float> q({self.output_height(),
                                       self.output_width()});
                 std::memcpy(x.mutable_data(), fx.data(),
                             fx.size() * sizeof(float));
                 std::memcpy(y.mutable_data(), fy.data(),
                             fy.size() * sizeof(float));
                 std::memcpy(q.mutable_data(), reliability.data(),
                             reliability.size() * sizeof(float));
                 return py::make_tuple(x, y, q);
             }, py::arg("input"), py::arg("reference"),
             "Execute NVOFA on two grayscale8 frames. Returns raw flow x/y "
             "(S10.5 converted to pixels) and inverse-cost reliability.")
        .def_property_readonly("width", &sylc::NvofFlow::width)
        .def_property_readonly("height", &sylc::NvofFlow::height)
        .def_property_readonly("output_width", &sylc::NvofFlow::output_width)
        .def_property_readonly("output_height", &sylc::NvofFlow::output_height)
        .def_property_readonly("grid", &sylc::NvofFlow::grid)
        .def_property_readonly("backend", &sylc::NvofFlow::backend);
    m.attr("NVOF_CUDA_AVAILABLE") = true;
#else
    m.attr("NVOF_CUDA_AVAILABLE") = false;
#endif

    // --- synth3d (2D->3D): DepthEngine test-only binding ------------------------
    // Test-only entry point (used by tests/synth3d/test_depth_engine.py) that
    // proves DLL loading, the provider ladder and I/O binding end-to-end,
    // before any renderer/warp work exists. The CPU-side preprocessing (nearest
    // resize + ImageNet normalization, HWC->CHW) mirrors what Task 4's real
    // pipeline will eventually do on the GPU.
    m.def("depth_infer_test",
          [](const std::wstring& model, const std::wstring& ort_dir,
             py::array_t<uint8_t, py::array::c_style | py::array::forcecast> rgb,
              int side, int grid_width, int grid_height) {
              if (rgb.ndim() != 3 || rgb.shape(2) != 3)
                  throw std::runtime_error("rgb must be HxWx3 uint8");
              if (side <= 0)
                  throw std::runtime_error("side must be a positive inference grid");
               if ((grid_width > 0) != (grid_height > 0))
                   throw std::runtime_error(
                       "grid_width and grid_height must be provided together");
               const int W = grid_width > 0 ? grid_width : side;
               const int H = grid_height > 0 ? grid_height : side;
               std::vector<float> chw(3 * static_cast<size_t>(W) * H);
              static const float kMean[3] = {0.485f, 0.456f, 0.406f};
              static const float kStd[3] = {0.229f, 0.224f, 0.225f};
              auto r = rgb.unchecked<3>();
               const double sy = double(rgb.shape(0)) / H, sx = double(rgb.shape(1)) / W;
               for (int y = 0; y < H; ++y)
                   for (int x = 0; x < W; ++x) {
                      const int py_ = std::min<int>(int(y * sy), static_cast<int>(rgb.shape(0)) - 1);
                      const int px_ = std::min<int>(int(x * sx), static_cast<int>(rgb.shape(1)) - 1);
                      for (int c = 0; c < 3; ++c)
                           chw[c * W * H + y * W + x] =
                              (r(py_, px_, c) / 255.0f - kMean[c]) / kStd[c];
                  }
              DepthEngine eng;
              std::string err;
              DepthConfig cfg;
              cfg.model_path = model;
              cfg.ort_dir = ort_dir;
               cfg.side = side;
               cfg.width = W;
               cfg.height = H;
              // F3: must match Synth3D::worker_main's cfg.invert_output (synth3d.cpp) so
              // this test binding exercises the same shipped orientation as production.
              cfg.invert_output = true;
              if (!eng.init(cfg, err)) throw std::runtime_error("init: " + err);
              std::vector<float> model_input(
                  static_cast<size_t>(eng.input_views()) * chw.size());
              for (int view = 0; view < eng.input_views(); ++view)
                  std::copy(
                      chw.begin(), chw.end(),
                      model_input.begin() +
                          static_cast<size_t>(view) * chw.size());
               py::array_t<float> out({H, W});
              {
                  py::gil_scoped_release nogil;
                  if (!eng.infer(model_input.data(), out.mutable_data(), err))
                      throw std::runtime_error("infer: " + err);
              }
              return out;
          },
          py::arg("model_path"), py::arg("ort_dir"), py::arg("rgb"),
          py::arg("side") = kDefaultDepthSide,
          py::arg("grid_width") = 0, py::arg("grid_height") = 0,
          "Test-only: uint8 HxWx3 RGB -> side x side float32 depth map via DepthEngine "
          "(ORT provider ladder TensorRT/DirectML/CPU). CPU-side resize+normalize. "
          "side must match the grid the model was exported for (756 default, 518 for "
          "the faster presets); a fixed-shape mismatch fails init with a named error.");

    // Profiling-only companion to depth_infer_test.  The legacy helper creates
    // and destroys an ORT session for every call, so timing it mostly measures
    // DLL/session/cache setup rather than the hot per-map inference path.  This
    // entry point keeps one DepthEngine alive across warm-up and measured runs,
    // requests confidence exactly like SharedDepthService, and returns every
    // sample so tools_dev/profile_synth3d.py can report robust percentiles.
    m.def("_depth_infer_benchmark",
          [](const std::wstring& model, const std::wstring& ort_dir,
             py::array_t<uint8_t, py::array::c_style | py::array::forcecast> rgb,
             int side, int grid_width, int grid_height,
             int warmup, int iterations) {
              if (rgb.ndim() != 3 || rgb.shape(2) != 3)
                  throw std::runtime_error("rgb must be HxWx3 uint8");
              if (side <= 0)
                  throw std::runtime_error("side must be a positive inference grid");
              if ((grid_width > 0) != (grid_height > 0))
                  throw std::runtime_error(
                      "grid_width and grid_height must be provided together");
              if (warmup < 0 || iterations <= 0 || iterations > 10000)
                  throw std::runtime_error(
                      "warmup must be >= 0 and iterations must be in 1..10000");

              using BenchClock = std::chrono::steady_clock;
              const int W = grid_width > 0 ? grid_width : side;
              const int H = grid_height > 0 ? grid_height : side;
              const size_t frame = static_cast<size_t>(W) * H;
              const auto prep_start = BenchClock::now();
              std::vector<float> chw(3 * frame);
              static const float kMean[3] = {0.485f, 0.456f, 0.406f};
              static const float kStd[3] = {0.229f, 0.224f, 0.225f};
              auto pixels = rgb.unchecked<3>();
              const double sy = double(rgb.shape(0)) / H;
              const double sx = double(rgb.shape(1)) / W;
              for (int y = 0; y < H; ++y) {
                  const int py_ = std::min<int>(
                      int(y * sy), static_cast<int>(rgb.shape(0)) - 1);
                  for (int x = 0; x < W; ++x) {
                      const int px_ = std::min<int>(
                          int(x * sx), static_cast<int>(rgb.shape(1)) - 1);
                      for (int c = 0; c < 3; ++c) {
                          chw[static_cast<size_t>(c) * frame +
                              static_cast<size_t>(y) * W + x] =
                              (pixels(py_, px_, c) / 255.0f - kMean[c]) /
                              kStd[c];
                      }
                  }
              }
              const double reference_cpu_preprocess_ms =
                  std::chrono::duration<double, std::milli>(
                      BenchClock::now() - prep_start).count();

              DepthEngine engine;
              DepthConfig cfg;
              cfg.model_path = model;
              cfg.ort_dir = ort_dir;
              cfg.side = side;
              cfg.width = W;
              cfg.height = H;
              cfg.invert_output = true;
              std::string err;
              const auto init_start = BenchClock::now();
              bool init_ok = false;
              {
                  py::gil_scoped_release nogil;
                  init_ok = engine.init(cfg, err);
              }
              const double init_ms = std::chrono::duration<double, std::milli>(
                  BenchClock::now() - init_start).count();
              if (!init_ok) throw std::runtime_error("init: " + err);

              const int input_views = (std::max)(1, engine.input_views());
              std::vector<float> model_input(
                  static_cast<size_t>(input_views) * chw.size());
              for (int view = 0; view < input_views; ++view) {
                  std::copy(chw.begin(), chw.end(),
                            model_input.begin() +
                                static_cast<size_t>(view) * chw.size());
              }
              std::vector<float> out(frame), confidence(frame);
              std::vector<double> warmup_ms;
              std::vector<double> samples_ms;
              warmup_ms.reserve(static_cast<size_t>(warmup));
              samples_ms.reserve(static_cast<size_t>(iterations));
              bool infer_ok = true;
              {
                  py::gil_scoped_release nogil;
                  for (int i = 0; i < warmup && infer_ok; ++i) {
                      const auto t0 = BenchClock::now();
                      infer_ok = engine.infer(
                          model_input.data(), out.data(), err,
                          confidence.data());
                      warmup_ms.push_back(
                          std::chrono::duration<double, std::milli>(
                              BenchClock::now() - t0).count());
                  }
                  for (int i = 0; i < iterations && infer_ok; ++i) {
                      const auto t0 = BenchClock::now();
                      infer_ok = engine.infer(
                          model_input.data(), out.data(), err,
                          confidence.data());
                      samples_ms.push_back(
                          std::chrono::duration<double, std::milli>(
                              BenchClock::now() - t0).count());
                  }
              }
              const std::string provider = engine.provider();
              engine.shutdown();
              if (!infer_ok) throw std::runtime_error("infer: " + err);

              std::vector<double> sorted = samples_ms;
              std::sort(sorted.begin(), sorted.end());
              double total = 0.0;
              for (double sample : samples_ms) total += sample;
              const double mean = total / samples_ms.size();
              auto percentile = [&](double p) {
                  const size_t index = (std::min)(
                      sorted.size() - 1,
                      static_cast<size_t>(
                          std::ceil(p * static_cast<double>(sorted.size()))) - 1);
                  return sorted[index];
              };

              py::dict result;
              result["provider"] = provider;
              result["grid_width"] = W;
              result["grid_height"] = H;
              result["input_views"] = input_views;
              result["init_ms"] = init_ms;
              result["reference_cpu_preprocess_ms"] =
                  reference_cpu_preprocess_ms;
              result["warmup_ms"] = warmup_ms;
              result["samples_ms"] = samples_ms;
              result["min_ms"] = sorted.front();
              result["p50_ms"] = percentile(0.50);
              result["p95_ms"] = percentile(0.95);
              result["max_ms"] = sorted.back();
              result["mean_ms"] = mean;
              result["fps"] = mean > 0.0 ? 1000.0 / mean : 0.0;
              return result;
          },
          py::arg("model_path"), py::arg("ort_dir"), py::arg("rgb"),
          py::arg("side") = kDefaultDepthSide,
          py::arg("grid_width") = 0, py::arg("grid_height") = 0,
          py::arg("warmup") = 3, py::arg("iterations") = 30,
          "Profile one persistent production DepthEngine session. Returns "
          "initialization, warm-up and per-inference timings; confidence is "
          "requested exactly like SharedDepthService.");

    m.def("_synth3d_detect_letterbox_test",
          [](py::array_t<uint8_t,
                         py::array::c_style | py::array::forcecast> luma) {
              if (luma.ndim() != 2)
                  throw std::runtime_error("luma must be a 2-D uint8 array");
              const int height = static_cast<int>(luma.shape(0));
              const int width = static_cast<int>(luma.shape(1));
              const size_t n = static_cast<size_t>(width) * height;
              std::vector<float> normalized(n);
              const uint8_t* src = luma.data();
              for (size_t i = 0; i < n; ++i)
                  normalized[i] = src[i] / 255.0f;
              synth3d_aspect::HorizontalBars bars;
              {
                  py::gil_scoped_release nogil;
                  bars = synth3d_aspect::detect_horizontal_letterbox(
                      normalized, width, height);
              }
              return py::make_tuple(bars.top, bars.bottom, bars.valid);
          },
          py::arg("luma"),
          "Test the production encoded-horizontal-letterbox detector. Returns "
          "(top_rows, bottom_rows, valid).");

    m.def("_synth3d_reduce_lab_metrics_test",
          [](py::array_t<uint8_t,
                         py::array::c_style | py::array::forcecast> guards) {
              if (guards.ndim() != 3 || guards.shape(2) != 2)
                  throw std::runtime_error(
                      "guards must be a contiguous HxWx2 uint8 array");
              const size_t count = static_cast<size_t>(guards.shape(0)) *
                                   static_cast<size_t>(guards.shape(1));
              const auto summary = StereoLab::reduce_metric_rg8(
                  guards.data(), count);
              py::dict out;
              out["mean"] = summary.mean;
              out["p95_active"] = summary.p95_active;
              out["asym"] = summary.asym;
              out["coverage"] = summary.coverage;
              return out;
          },
          py::arg("guards"),
          "Test the production Stereo Lab metric reducer on an HxWx2 RG8 "
          "guard map. p95_active excludes inactive samples.");

    m.def("_synth3d_reduce_lab_shadow_metrics_test",
          [](py::array_t<uint8_t,
                         py::array::c_style | py::array::forcecast> metrics,
             float comfort_scale_px) {
              if (metrics.ndim() != 3 || metrics.shape(2) != 4)
                  throw std::runtime_error(
                      "metrics must be a contiguous HxWx4 uint8 array");
              const size_t count = static_cast<size_t>(metrics.shape(0)) *
                                   static_cast<size_t>(metrics.shape(1));
              const auto summary = StereoLab::reduce_metric_rgba8(
                  metrics.data(), count, comfort_scale_px);
              py::dict out;
              out["mean"] = summary.mean;
              out["p95_active"] = summary.p95_active;
              out["asym"] = summary.asym;
              out["coverage"] = summary.coverage;
              out["comfort_mean_loss_px"] = summary.comfort_mean_loss_px;
              out["comfort_p95_loss_px"] = summary.comfort_p95_loss_px;
              out["comfort_coverage"] = summary.comfort_coverage;
              out["edge_veto_p95"] = summary.edge_veto_p95;
              out["edge_veto_coverage"] = summary.edge_veto_coverage;
              return out;
          },
          py::arg("metrics"), py::arg("comfort_scale_px"),
          "Test the production RGBA8 Lab/comfort shadow reducer. B is "
          "normalized comfort loss and A is protected-edge strength.");

    m.def("_synth3d_registry_stats", [] {
              size_t services = 0, active = 0, idle = 0;
              SharedDepthService::debug_registry_stats(
                  services, active, idle);
              py::dict result;
              result["services"] = services;
              result["active"] = active;
              result["idle"] = idle;
              return result;
          },
          "Test/support diagnostics for the bounded shared-depth registry. "
          "Idle is guaranteed to be at most one.");

    m.def("_synth3d_advisory_decay_test",
          [](double value, int64_t age_ms) {
              return SharedDepthService::advisory_decay(value, age_ms);
          },
          py::arg("value"), py::arg("age_ms"),
          "Test surface for the look-ahead advisory dead-reckoning "
          "(production code path): the stored pump delay minus its "
          "steady-clock age; the NONE sentinel (-1e9) passes through "
          "undecayed.");

    m.def("_synth3d_shot_gate_test",
          [](double map_video_ms, double presented_video_ms,
             double cut_pts_ms) {
              return SharedDepthService::cross_shot_gate(
                  map_video_ms, presented_video_ms, cut_pts_ms);
          },
          py::arg("map_video_ms"), py::arg("presented_video_ms"),
          py::arg("cut_pts_ms"),
          "Test surface for the cross-shot gate decision (production code "
          "path, 04/08): true while the cut boundary separates the published "
          "map's source observation from the presented frame (map < cut <= "
          "presented, with the +-4 ms media-clock jitter tolerance). Any "
          "negative/invalid clock never gates.");

    m.def("_synth3d_boundary_signal_test",
          [](float histogram_distance, float scene_cut_threshold) {
              return SharedDepthService::boundary_scene_signal(
                  histogram_distance, scene_cut_threshold);
          },
          py::arg("histogram_distance"), py::arg("scene_cut_threshold"),
          "Test surface for the boundary-crossing scene-cut signal "
          "(production code path, 04/08 round 2): the signal handed to "
          "DepthStabilizer::step() when a worker observation crossed a "
          "recorded cut boundary — raised to the stabilizer's scene-cut "
          "threshold so the snap is guaranteed even when both content "
          "detectors are blind (similar compositions); a higher measured "
          "distance passes through untouched.");

    m.def("_synth3d_realign_test",
          [](py::array_t<uint16_t,
                 py::array::c_style | py::array::forcecast> depth,
             py::array_t<float,
                 py::array::c_style | py::array::forcecast> luma,
             int max_threads) {
              if (depth.ndim() != 2 || luma.ndim() != 2 ||
                  depth.shape(0) != luma.shape(0) ||
                  depth.shape(1) != luma.shape(1))
                  throw std::runtime_error(
                      "depth/luma must be equal-shape 2D arrays");
              const int height = static_cast<int>(depth.shape(0));
              const int width = static_cast<int>(depth.shape(1));
              const size_t n = static_cast<size_t>(width) * height;
              std::vector<uint16_t> d(depth.data(), depth.data() + n);
              std::vector<float> l(luma.data(), luma.data() + n);
              std::vector<uint16_t> scratch;
              {
                  py::gil_scoped_release nogil;
                  synth3d_surface::realign_contours(
                      d, l, width, height, scratch, max_threads);
              }
              py::array_t<uint16_t> out({height, width});
              std::copy(d.begin(), d.end(), out.mutable_data());
              return out;
          },
          py::arg("depth_q16"), py::arg("luma"), py::arg("max_threads") = 1,
          "Test surface for contour re-anchoring (production code path, "
          "04/08): snaps the depth discontinuity onto the dominant unique "
          "image edge — a near-halo painted over the background beside a "
          "silhouette returns to its own side's depth; aligned edges, weak "
          "or ambiguous image edges and filaments are untouched.");

    // --- synth3d (2D->3D): DepthStabilizer (temporal fit/EMA/cut) ----------------
    // Test-only-visible helper (the closed-form fit is also used internally by
    // DepthStabilizer::step) plus the stabilizer itself, consumed by Task 4's
    // inference thread. See tests/synth3d/test_depth_stabilizer.py.
    m.def("_synth3d_fit_scale_shift",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> ref,
             py::array_t<float, py::array::c_style | py::array::forcecast> cur) {
              if (ref.ndim() != 1 || cur.ndim() != 1 || ref.shape(0) != cur.shape(0))
                  throw std::runtime_error(
                      "ref and cur must be 1D float32 arrays of equal length");
              const size_t n = static_cast<size_t>(ref.shape(0));
              float a = 1.0f, b = 0.0f;
              {
                  py::gil_scoped_release nogil;
                  DepthStabilizer::fit_scale_shift(ref.data(), cur.data(), n, a, b);
              }
              return py::make_tuple(a, b);
          },
          py::arg("ref"), py::arg("cur"),
          "Least-squares closed form: argmin_(a,b) sum (a*cur+b - ref)^2 -> (a, b).");

    m.def("_synth3d_estimate_flow_test",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> source,
             py::array_t<float, py::array::c_style | py::array::forcecast> destination,
             int width, int height, int threads) {
              const size_t expected =
                  static_cast<size_t>(width) * static_cast<size_t>(height);
              if (source.ndim() != 1 || destination.ndim() != 1 ||
                  static_cast<size_t>(source.shape(0)) != expected ||
                  static_cast<size_t>(destination.shape(0)) != expected)
                  throw std::runtime_error(
                      "source/destination must be 1D float32 arrays of length "
                      "width*height");
              std::vector<float> src(source.data(), source.data() + expected);
              std::vector<float> dst(destination.data(),
                                     destination.data() + expected);
              synth3d_flow::DenseFlow flow;
              {
                  py::gil_scoped_release nogil;
                  flow = synth3d_flow::estimate_flow(
                      src, dst, width, height, threads);
              }
              const py::ssize_t n = static_cast<py::ssize_t>(expected);
              py::array_t<float> fx(n), fy(n), fq(n);
              std::memcpy(fx.mutable_data(), flow.x.data(),
                          expected * sizeof(float));
              std::memcpy(fy.mutable_data(), flow.y.data(),
                          expected * sizeof(float));
              std::memcpy(fq.mutable_data(), flow.quality.data(),
                          expected * sizeof(float));
              return py::make_tuple(fx, fy, fq);
          },
          py::arg("source"), py::arg("destination"),
          py::arg("width"), py::arg("height"), py::arg("threads") = 1,
          "Test surface for the worker's CPU optical flow (production code "
          "path). Returns (flow_x, flow_y, quality); source->destination "
          "displacement at destination coordinates, full resolution.");

    m.def("_synth3d_fuse_flow_test",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> prev,
             py::array_t<float, py::array::c_style | py::array::forcecast> cur,
             py::array_t<float, py::array::c_style | py::array::forcecast> next,
             int width, int height, float video_time_scale, int threads) {
              const size_t expected =
                  static_cast<size_t>(width) * static_cast<size_t>(height);
              for (auto* a : {&prev, &cur, &next})
                  if (a->ndim() != 1 ||
                      static_cast<size_t>(a->shape(0)) != expected)
                      throw std::runtime_error(
                          "prev/cur/next must be 1D float32 arrays of length "
                          "width*height");
              std::vector<float> p(prev.data(), prev.data() + expected);
              std::vector<float> c(cur.data(), cur.data() + expected);
              std::vector<float> f(next.data(), next.data() + expected);
              std::vector<float> ox, oy, orel, omotion;
              double motion_sum = 0.0;
              float mean_flow = 0.0f;
              {
                  py::gil_scoped_release nogil;
                  synth3d_flow::DenseFlow forward =
                      synth3d_flow::estimate_flow(p, c, width, height, threads);
                  synth3d_flow::DenseFlow future =
                      synth3d_flow::estimate_flow(f, c, width, height, threads);
                  synth3d_flow::fuse_bidirectional(
                      p, c, forward, future, width, height, video_time_scale,
                      ox, oy, orel, omotion, motion_sum, mean_flow, threads);
              }
              const py::ssize_t n = static_cast<py::ssize_t>(expected);
              py::array_t<float> fx(n), fy(n), frel(n), fm(n);
              std::memcpy(fx.mutable_data(), ox.data(), expected * sizeof(float));
              std::memcpy(fy.mutable_data(), oy.data(), expected * sizeof(float));
              std::memcpy(frel.mutable_data(), orel.data(),
                          expected * sizeof(float));
              std::memcpy(fm.mutable_data(), omotion.data(),
                          expected * sizeof(float));
              return py::make_tuple(fx, fy, frel, fm, motion_sum, mean_flow);
          },
          py::arg("prev"), py::arg("cur"), py::arg("next"),
          py::arg("width"), py::arg("height"),
          py::arg("video_time_scale") = 1.0f, py::arg("threads") = 1,
          "Round 7 test surface: bidirectional flow fusion (production code "
          "path). Returns (flow_x, flow_y, reliability, motion, motion_sum, "
          "mean_flow) — the fused prev->cur transport at cur coordinates.");

    m.def("_synth3d_fuse_candidates_test",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> prev,
             py::array_t<float, py::array::c_style | py::array::forcecast> cur,
             py::array_t<float, py::array::c_style | py::array::forcecast> fwd_x,
             py::array_t<float, py::array::c_style | py::array::forcecast> fwd_y,
             py::array_t<float, py::array::c_style | py::array::forcecast> fwd_q,
             py::array_t<float, py::array::c_style | py::array::forcecast> fut_x,
             py::array_t<float, py::array::c_style | py::array::forcecast> fut_y,
             py::array_t<float, py::array::c_style | py::array::forcecast> fut_q,
             py::object hint_x_obj, py::object hint_y_obj,
             py::object hint_q_obj,
             int width, int height, float video_time_scale, int threads) {
              const size_t expected =
                  static_cast<size_t>(width) * static_cast<size_t>(height);
              auto take = [&](py::array_t<
                      float, py::array::c_style | py::array::forcecast>& a) {
                  if (a.ndim() != 1 ||
                      static_cast<size_t>(a.shape(0)) != expected)
                      throw std::runtime_error(
                          "all fields must be flat float32 width*height");
                  return std::vector<float>(a.data(), a.data() + expected);
              };
              std::vector<float> p = take(prev), c = take(cur);
              synth3d_flow::DenseFlow forward, future, hint;
              forward.x = take(fwd_x); forward.y = take(fwd_y);
              forward.quality = take(fwd_q);
              future.x = take(fut_x); future.y = take(fut_y);
              future.quality = take(fut_q);
              synth3d_flow::DenseFlow* hint_ptr = nullptr;
              if (!hint_x_obj.is_none()) {
                  auto hx = hint_x_obj.cast<py::array_t<
                      float, py::array::c_style | py::array::forcecast>>();
                  auto hy = hint_y_obj.cast<py::array_t<
                      float, py::array::c_style | py::array::forcecast>>();
                  auto hq = hint_q_obj.cast<py::array_t<
                      float, py::array::c_style | py::array::forcecast>>();
                  hint.x = take(hx); hint.y = take(hy);
                  hint.quality = take(hq);
                  hint_ptr = &hint;
              }
              std::vector<float> ox, oy, orel, omotion;
              double motion_sum = 0.0;
              float mean_flow = 0.0f;
              {
                  py::gil_scoped_release nogil;
                  synth3d_flow::fuse_bidirectional(
                      p, c, forward, future, width, height,
                      video_time_scale, ox, oy, orel, omotion,
                      motion_sum, mean_flow, threads, hint_ptr);
              }
              const py::ssize_t n = static_cast<py::ssize_t>(expected);
              py::array_t<float> fx(n), fy(n), frel(n), fm(n);
              std::memcpy(fx.mutable_data(), ox.data(), expected * sizeof(float));
              std::memcpy(fy.mutable_data(), oy.data(), expected * sizeof(float));
              std::memcpy(frel.mutable_data(), orel.data(),
                          expected * sizeof(float));
              std::memcpy(fm.mutable_data(), omotion.data(),
                          expected * sizeof(float));
              return py::make_tuple(fx, fy, frel, fm);
          },
          py::arg("prev"), py::arg("cur"),
          py::arg("fwd_x"), py::arg("fwd_y"), py::arg("fwd_q"),
          py::arg("fut_x"), py::arg("fut_y"), py::arg("fut_q"),
          py::arg("hint_x") = py::none(), py::arg("hint_y") = py::none(),
          py::arg("hint_q") = py::none(),
          py::arg("width"), py::arg("height"),
          py::arg("video_time_scale") = 1.0f, py::arg("threads") = 1,
          "Phase-1 test surface: triple fusion with EXPLICIT candidate "
          "fields (production code path). The optional hint triple is the "
          "decoder motion-vector candidate; quality 0 = no candidate.");

    m.def("_synth3d_rasterize_hints_test",
          [](py::array_t<int16_t,
                 py::array::c_style | py::array::forcecast> mv_xy,
             py::array_t<uint8_t,
                 py::array::c_style | py::array::forcecast> valid,
             int blocks_w, int blocks_h,
             int source_width, int source_height,
             int grid_width, int grid_height, float time_scale) {
              const size_t blocks =
                  static_cast<size_t>(blocks_w) * blocks_h;
              if (mv_xy.ndim() != 1 ||
                  static_cast<size_t>(mv_xy.shape(0)) != 2 * blocks ||
                  valid.ndim() != 1 ||
                  static_cast<size_t>(valid.shape(0)) != blocks)
                  throw std::runtime_error(
                      "mv_xy must be flat 2*blocks int16, valid flat blocks");
              std::vector<float> ox, oy, oq;
              synth3d_flow::rasterize_motion_hints(
                  mv_xy.data(), valid.data(), blocks_w, blocks_h,
                  source_width, source_height, grid_width, grid_height,
                  time_scale, ox, oy, oq);
              const py::ssize_t n =
                  static_cast<py::ssize_t>(grid_width) * grid_height;
              py::array_t<float> fx(n), fy(n), fq(n);
              std::memcpy(fx.mutable_data(), ox.data(), ox.size() * sizeof(float));
              std::memcpy(fy.mutable_data(), oy.data(), oy.size() * sizeof(float));
              std::memcpy(fq.mutable_data(), oq.data(), oq.size() * sizeof(float));
              return py::make_tuple(fx, fy, fq);
          },
          py::arg("mv_xy"), py::arg("valid"),
          py::arg("blocks_w"), py::arg("blocks_h"),
          py::arg("source_width"), py::arg("source_height"),
          py::arg("grid_width"), py::arg("grid_height"),
          py::arg("time_scale") = 1.0f,
          "Phase-1 test surface: decoder block motion field -> inference "
          "grid (production code path). Quarter-pel per display frame, "
          "anisotropic scaling, invalid blocks -> quality 0.");

    m.def("_synth3d_mean_divergence_test",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> fx,
             py::array_t<float, py::array::c_style | py::array::forcecast> fy,
             int width, int height) {
              const size_t expected =
                  static_cast<size_t>(width) * static_cast<size_t>(height);
              if (fx.ndim() != 1 || fy.ndim() != 1 ||
                  static_cast<size_t>(fx.shape(0)) != expected ||
                  static_cast<size_t>(fy.shape(0)) != expected)
                  throw std::runtime_error(
                      "fx/fy must be flat float32 width*height");
              std::vector<float> vx(fx.data(), fx.data() + expected);
              std::vector<float> vy(fy.data(), fy.data() + expected);
              return synth3d_flow::mean_divergence(vx, vy, width, height);
          },
          py::arg("flow_x"), py::arg("flow_y"),
          py::arg("width"), py::arg("height"),
          "Phase-2 test surface: mean divergence of a flow field "
          "(production code path); positive = expansion/looming.");

    m.def("_synth3d_expand_motion_test",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> motion,
             py::array_t<float, py::array::c_style | py::array::forcecast> boundary,
             py::array_t<float, py::array::c_style | py::array::forcecast> fx,
             py::array_t<float, py::array::c_style | py::array::forcecast> fy,
             int width, int height, bool directional, int threads) {
              const size_t expected =
                  static_cast<size_t>(width) * static_cast<size_t>(height);
              for (auto* a : {&motion, &boundary, &fx, &fy})
                  if (a->ndim() != 1 ||
                      static_cast<size_t>(a->shape(0)) != expected)
                      throw std::runtime_error(
                          "all fields must be flat float32 width*height");
              std::vector<float> m_(motion.data(), motion.data() + expected);
              std::vector<float> b(boundary.data(),
                                   boundary.data() + expected);
              std::vector<float> vx(fx.data(), fx.data() + expected);
              std::vector<float> vy(fy.data(), fy.data() + expected);
              std::vector<float> scratch;
              {
                  py::gil_scoped_release nogil;
                  synth3d_flow::expand_boundary_motion(
                      m_, b, vx, vy, width, height, scratch, threads,
                      directional);
              }
              py::array_t<float> out(static_cast<py::ssize_t>(expected));
              std::memcpy(out.mutable_data(), m_.data(),
                          expected * sizeof(float));
              return out;
          },
          py::arg("motion"), py::arg("boundary"),
          py::arg("flow_x"), py::arg("flow_y"),
          py::arg("width"), py::arg("height"),
          py::arg("directional") = true, py::arg("threads") = 1,
          "Phase-2 test surface: silhouette motion expansion (production "
          "code path). directional=false reproduces the historical 3x3 "
          "guard-band max exactly; true adds the upstream anti-trail reach.");

    m.def("_synth3d_refine_depth_test",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> depth,
             py::array_t<float, py::array::c_style | py::array::forcecast> luma,
             py::object confidence_obj, int threads) {
              if (depth.ndim() != 2 || luma.ndim() != 2 ||
                  depth.shape(0) != luma.shape(0) ||
                  depth.shape(1) != luma.shape(1)) {
                  throw std::runtime_error(
                      "depth and luma must be equal-shape 2D float32 arrays");
              }
              const int height = static_cast<int>(depth.shape(0));
              const int width = static_cast<int>(depth.shape(1));
              const size_t n = static_cast<size_t>(width) * height;
              std::vector<float> refined(depth.data(), depth.data() + n);
              std::vector<float> guide(luma.data(), luma.data() + n);
              std::vector<float> confidence;
              if (!confidence_obj.is_none()) {
                  auto c = confidence_obj.cast<py::array_t<
                      float, py::array::c_style | py::array::forcecast>>();
                  if (c.ndim() != 2 || c.shape(0) != height ||
                      c.shape(1) != width) {
                      throw std::runtime_error(
                          "confidence must match the depth shape");
                  }
                  confidence.assign(c.data(), c.data() + n);
              }
              std::vector<float> boundary, scratch;
              {
                  py::gil_scoped_release nogil;
                  synth3d_surface::compute_boundary(
                      refined, guide, width, height, boundary, scratch);
                  synth3d_surface::refine_observation(
                      refined, guide, confidence, boundary,
                      width, height, scratch, threads);
              }
              py::array_t<float> refined_out({height, width});
              py::array_t<float> boundary_out({height, width});
              std::memcpy(refined_out.mutable_data(), refined.data(),
                          n * sizeof(float));
              std::memcpy(boundary_out.mutable_data(), boundary.data(),
                          n * sizeof(float));
              return py::make_tuple(refined_out, boundary_out);
          },
          py::arg("depth"), py::arg("luma"),
          py::arg("confidence") = py::none(), py::arg("threads") = 1,
          "Exercise the production edge-aware depth-observation refinement. "
          "Returns (refined_depth, three-pixel_surface_boundary).");

    m.def("_synth3d_build_geometry_test",
          [](py::array_t<uint16_t,
                    py::array::c_style | py::array::forcecast> depth,
             py::array_t<uint8_t,
                    py::array::c_style | py::array::forcecast> rgb,
             py::object confidence_obj, int threads) {
              if (depth.ndim() != 2 || rgb.ndim() != 3 ||
                  rgb.shape(0) != depth.shape(0) ||
                  rgb.shape(1) != depth.shape(1) || rgb.shape(2) != 3) {
                  throw std::runtime_error(
                      "depth must be HxW uint16 and rgb must be matching HxWx3 uint8");
              }
              const int height = static_cast<int>(depth.shape(0));
              const int width = static_cast<int>(depth.shape(1));
              const size_t n = static_cast<size_t>(width) * height;
              std::vector<uint16_t> q16(depth.data(), depth.data() + n);
              std::vector<float> chw(3 * n), luma(n), confidence;
              const uint8_t* pixels = rgb.data();
              for (size_t i = 0; i < n; ++i) {
                  const float r = pixels[3 * i + 0] / 255.0f;
                  const float g = pixels[3 * i + 1] / 255.0f;
                  const float b = pixels[3 * i + 2] / 255.0f;
                  chw[i] = (r - 0.485f) / 0.229f;
                  chw[n + i] = (g - 0.456f) / 0.224f;
                  chw[2 * n + i] = (b - 0.406f) / 0.225f;
                  luma[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
              }
              if (!confidence_obj.is_none()) {
                  auto c = confidence_obj.cast<py::array_t<
                      float, py::array::c_style | py::array::forcecast>>();
                  if (c.ndim() != 2 || c.shape(0) != height ||
                      c.shape(1) != width)
                      throw std::runtime_error(
                          "confidence must match the depth shape");
                  confidence.assign(c.data(), c.data() + n);
              }
              std::vector<float> depth_float(n), boundary, scratch;
              for (size_t i = 0; i < n; ++i)
                  depth_float[i] = q16[i] / 65535.0f;
              std::vector<uint16_t> geometry;
              {
                  py::gil_scoped_release nogil;
                  synth3d_surface::compute_boundary(
                      depth_float, luma, width, height, boundary, scratch);
                  synth3d_surface::build_geometry_map(
                      q16, chw, luma, confidence, boundary,
                      width, height, geometry, scratch, threads);
              }
              py::array_t<uint16_t> owned_out({height, width});
              py::array_t<uint16_t> safety_out({height, width});
              py::array_t<uint16_t> ownership_out({height, width});
              uint16_t* owned_dst = owned_out.mutable_data();
              uint16_t* safety_dst = safety_out.mutable_data();
              uint16_t* ownership_dst = ownership_out.mutable_data();
              for (size_t i = 0; i < n; ++i) {
                  owned_dst[i] = geometry[4 * i + 1];
                  safety_dst[i] = geometry[4 * i + 2];
                  ownership_dst[i] = geometry[4 * i + 3];
              }
              return py::make_tuple(owned_out, safety_out, ownership_out);
          },
          py::arg("depth"), py::arg("rgb"),
          py::arg("confidence") = py::none(), py::arg("threads") = 1,
          "Build the production foreground-ownership and stereo-safety maps.");

    py::class_<DepthStabilizer>(m, "DepthStabilizer")
        .def(py::init<size_t>(), py::arg("n"))
        .def("reset", &DepthStabilizer::reset,
             "Re-prime on the next step() (seek / stream restart).")
        .def("reproject",
             [](DepthStabilizer& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> flow_x,
                py::array_t<float, py::array::c_style | py::array::forcecast> flow_y,
                py::object reliability_obj, size_t width, size_t height) {
                 const size_t n = width * height;
                 if (n != self.size() || flow_x.ndim() != 1 ||
                     flow_y.ndim() != 1 ||
                     static_cast<size_t>(flow_x.shape(0)) != n ||
                     static_cast<size_t>(flow_y.shape(0)) != n)
                     throw std::runtime_error(
                         "flow_x/flow_y must be flat float32 arrays matching width*height");
                 py::array_t<float, py::array::c_style | py::array::forcecast>
                     reliability;
                 const float* reliability_ptr = nullptr;
                 if (!reliability_obj.is_none()) {
                     reliability = reliability_obj.cast<py::array_t<
                         float, py::array::c_style | py::array::forcecast>>();
                     if (reliability.ndim() != 1 ||
                         static_cast<size_t>(reliability.shape(0)) != n)
                         throw std::runtime_error(
                             "reliability must be None or a flat float32 array "
                             "matching width*height");
                     reliability_ptr = reliability.data();
                 }
                 py::gil_scoped_release nogil;
                 self.reproject(flow_x.data(), flow_y.data(),
                                reliability_ptr, width, height);
             },
             py::arg("flow_x"), py::arg("flow_y"),
             py::arg("reliability") = py::none(),
             py::arg("width"), py::arg("height"),
             "Transport the running depth state with previous->current flow.")
        .def("step",
             [](DepthStabilizer& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> raw,
                py::object motion_obj, float scene_change,
                py::object confidence_obj, py::object boundary_obj) {
                 if (raw.ndim() != 1 || static_cast<size_t>(raw.shape(0)) != self.size())
                     throw std::runtime_error(
                         "raw must be a 1D float32 array of length " +
                         std::to_string(self.size()));
                 py::array_t<float, py::array::c_style | py::array::forcecast> motion;
                 const float* motion_ptr = nullptr;
                 if (!motion_obj.is_none()) {
                     motion = motion_obj.cast<py::array_t<float,
                              py::array::c_style | py::array::forcecast>>();
                     if (motion.ndim() != 1 ||
                         static_cast<size_t>(motion.shape(0)) != self.size())
                         throw std::runtime_error(
                             "motion must be None or a 1D float32 array of length " +
                             std::to_string(self.size()));
                     motion_ptr = motion.data();
                 }
                 py::array_t<float, py::array::c_style | py::array::forcecast> confidence;
                 const float* confidence_ptr = nullptr;
                 if (!confidence_obj.is_none()) {
                     confidence = confidence_obj.cast<py::array_t<float,
                                  py::array::c_style | py::array::forcecast>>();
                     if (confidence.ndim() != 1 ||
                         static_cast<size_t>(confidence.shape(0)) != self.size())
                         throw std::runtime_error(
                             "confidence must be None or a 1D float32 array of length " +
                             std::to_string(self.size()));
                     confidence_ptr = confidence.data();
                 }
                 py::array_t<float, py::array::c_style | py::array::forcecast> boundary;
                 const float* boundary_ptr = nullptr;
                 if (!boundary_obj.is_none()) {
                     boundary = boundary_obj.cast<py::array_t<float,
                               py::array::c_style | py::array::forcecast>>();
                     if (boundary.ndim() != 1 ||
                         static_cast<size_t>(boundary.shape(0)) != self.size())
                         throw std::runtime_error(
                             "surface_boundary must be None or a 1D float32 "
                             "array of length " + std::to_string(self.size()));
                     boundary_ptr = boundary.data();
                 }
                 py::array_t<uint16_t> out(static_cast<py::ssize_t>(self.size()));
                 bool was_cut = false;
                 {
                     py::gil_scoped_release nogil;
                     was_cut = self.step(raw.data(), out.mutable_data(),
                                         motion_ptr, scene_change,
                                         confidence_ptr, boundary_ptr);
                 }
                 return py::make_tuple(out, was_cut);
             },
             py::arg("raw"), py::arg("motion") = py::none(),
             py::arg("scene_change") = 0.0f,
             py::arg("confidence") = py::none(),
             py::arg("surface_boundary") = py::none(),
             "raw model output (float32 1D, length n) -> (out_q16 uint16 1D, was_cut bool).")
        .def_readwrite("alpha", &DepthStabilizer::alpha)
        .def_readwrite("alpha_static", &DepthStabilizer::alpha_static)
        .def_readwrite("alpha_motion", &DepthStabilizer::alpha_motion)
        .def_readwrite("motion_low", &DepthStabilizer::motion_low)
        .def_readwrite("motion_high", &DepthStabilizer::motion_high)
        .def_readwrite("tone_alpha", &DepthStabilizer::tone_alpha)
        .def_readwrite("depth_contrast", &DepthStabilizer::depth_contrast)
        .def_readwrite("depth_scurve", &DepthStabilizer::depth_scurve)
        .def_readwrite("confidence_floor", &DepthStabilizer::confidence_floor)
        .def_readwrite("local_stability_low",
                       &DepthStabilizer::local_stability_low)
        .def_readwrite("local_stability_high",
                       &DepthStabilizer::local_stability_high)
        .def_readwrite("local_jitter_low", &DepthStabilizer::local_jitter_low)
        .def_readwrite("local_jitter_high", &DepthStabilizer::local_jitter_high)
        .def_readwrite("local_jitter_alpha_scale",
                       &DepthStabilizer::local_jitter_alpha_scale)
        .def_readwrite("temporal_surface_memory",
                       &DepthStabilizer::temporal_surface_memory)
        .def_readwrite("temporal_stable_low",
                       &DepthStabilizer::temporal_stable_low)
        .def_readwrite("temporal_stable_high",
                       &DepthStabilizer::temporal_stable_high)
        .def_readwrite("boundary_fast_motion",
                       &DepthStabilizer::boundary_fast_motion)
        .def_readwrite("cut_threshold", &DepthStabilizer::cut_threshold)
        .def_readwrite("scene_cut_threshold", &DepthStabilizer::scene_cut_threshold)
        .def_readwrite("snap_frac", &DepthStabilizer::snap_frac)
        .def_readwrite("auto_convergence_percentile",
                       &DepthStabilizer::auto_convergence_percentile)
        .def_readwrite("convergence_alpha",
                       &DepthStabilizer::convergence_alpha)
        .def_property_readonly("suggested_convergence",
                               &DepthStabilizer::suggested_convergence)
        .def_property_readonly("last_motion", &DepthStabilizer::last_motion)
        .def_property_readonly("last_effective_alpha",
                               &DepthStabilizer::last_effective_alpha)
        .def_property_readonly("last_scene_change",
                               &DepthStabilizer::last_scene_change)
        .def_property_readonly("last_confidence",
                               &DepthStabilizer::last_confidence)
        .def_property_readonly("last_cut", &DepthStabilizer::last_cut)
        .def_property_readonly("last_depth_cut",
                               &DepthStabilizer::last_depth_cut)
        .def_property_readonly("last_scene_cut",
                               &DepthStabilizer::last_scene_cut)
        .def_property_readonly("last_depth_residual",
                               &DepthStabilizer::last_depth_residual)
        .def_property_readonly("last_stability",
                               &DepthStabilizer::last_stability)
        .def_property_readonly("last_history_support",
                               &DepthStabilizer::last_history_support)
        .def_property_readonly("cut_count", &DepthStabilizer::cut_count)
        .def("snap_count", &DepthStabilizer::snap_count)
        .def("set_dt_ms", &DepthStabilizer::set_dt_ms, py::arg("dt_ms"),
             "Backward-compatible single-clock setter: applies the same "
             "effective interval to map updates and source observations.")
        .def("set_update_dt_ms", &DepthStabilizer::set_update_dt_ms,
             py::arg("dt_ms"),
             "Compute wall-clock interval between stabilized map updates, in "
             "ms (clamped to [20, 500]); diagnostic only and deliberately "
             "does not alter temporal geometry.")
        .def("set_source_dt_ms", &DepthStabilizer::set_source_dt_ms,
             py::arg("dt_ms"),
             "Video PTS interval between the source frames being compared, in "
             "ms (clamped to [4, 500]); drives every content-temporal rule.")
        .def_property_readonly("dt_ms", &DepthStabilizer::dt_ms)
        .def_property_readonly("update_dt_ms", &DepthStabilizer::update_dt_ms)
        .def_property_readonly("source_dt_ms", &DepthStabilizer::source_dt_ms)
        .def_property_readonly("history_count", &DepthStabilizer::history_count)
        .def_readwrite("snap_confirm_ms", &DepthStabilizer::snap_confirm_ms)
        .def_readwrite("history_commit_ms", &DepthStabilizer::history_commit_ms)
        .def_readwrite("worker_threads", &DepthStabilizer::worker_threads)
        .def_property_readonly_static("kReferenceDtMs",
             [](const py::object&) { return DepthStabilizer::kReferenceDtMs; });

    // --- synth3d (2D->3D): CutGate (anti-flash histogram confirmation) ----------
    // See include/cut_gate.h. tests/synth3d/test_cut_gate.py drives this directly;
    // SharedDepthService's worker uses the same type internally (shared_depth_service.cpp).
    py::class_<CutGate>(m, "CutGate")
        .def(py::init<>())
        .def("update", &CutGate::update, py::arg("depth_cut"), py::arg("histogram_distance"),
             "depth_cut passes instantly and re-arms; histogram_distance alone needs "
             "two CONSECUTIVE calls >= histogram_threshold to confirm (a single spike "
             "sets pending() without confirming). Returns True exactly on the "
             "confirming cycle.")
        .def("pending", &CutGate::pending,
             "True while a single histogram spike awaits a second confirming frame.")
        .def_readwrite("histogram_threshold", &CutGate::histogram_threshold);

    m.attr("NATIVE_RENDERER_AVAILABLE") = true;
#else
    m.attr("NATIVE_RENDERER_AVAILABLE") = false;
#endif
}
