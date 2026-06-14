#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <cstring>
#include "mvc_demuxer.h"
#include "mvc_matroska_demuxer.h"
#include "mvc_m2ts_demuxer.h"
#include "mvc_ssif_demuxer.h"
#include "ssif_parser.h"
#include "matroska_reader.h"
#include "mvc_decoder.h"
#include "frame_ring_buffer.h"

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
        .def("pop",
             [](std::shared_ptr<FrameRingBuffer> self) {
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
             "Seek to timestamp in milliseconds (resets to start)",
             py::arg("timestamp_ms"));

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
                 dep["timestamp"] = pair.timestamp;
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
             py::arg("timestamp_ms"));

    // === MVC DECODER (edge264 integration) ===

#ifdef EDGE264_AVAILABLE
    // Only expose MVCDecoder if edge264 is properly linked.
    // If disabled (MSVC build for AV safety), this class will be missing,
    // forcing Python to fallback to the ctypes implementation using edge264.dll.

    // Decoded view structure
    py::class_<DecodedMVCFrame::View>(m, "DecodedView")
        .def_readonly("width", &DecodedMVCFrame::View::width)
        .def_readonly("height", &DecodedMVCFrame::View::height)
        .def_readonly("stride_y", &DecodedMVCFrame::View::stride_y)
        .def_readonly("stride_c", &DecodedMVCFrame::View::stride_c)
        .def_property_readonly("y_plane",
            [](const DecodedMVCFrame::View& self) -> py::array_t<uint8_t> {
                if (!self.y_plane) return py::array_t<uint8_t>();
                return py::array_t<uint8_t>({self.height, self.stride_y}, self.y_plane);
            })
        .def_property_readonly("cb_plane",
            [](const DecodedMVCFrame::View& self) -> py::array_t<uint8_t> {
                if (!self.cb_plane) return py::array_t<uint8_t>();
                int h_c = self.height / 2;
                return py::array_t<uint8_t>({h_c, self.stride_c}, self.cb_plane);
            })
        .def_property_readonly("cr_plane",
            [](const DecodedMVCFrame::View& self) -> py::array_t<uint8_t> {
                if (!self.cr_plane) return py::array_t<uint8_t>();
                int h_c = self.height / 2;
                return py::array_t<uint8_t>({h_c, self.stride_c}, self.cr_plane);
            });

    // Decoded MVC frame (both views)
    py::class_<DecodedMVCFrame>(m, "DecodedMVCFrame")
        .def(py::init<>())
        .def_readonly("base_view", &DecodedMVCFrame::base_view)
        .def_readonly("dependent_view", &DecodedMVCFrame::dependent_view)
        .def_readonly("has_mvc", &DecodedMVCFrame::has_mvc)
        .def_readonly("frame_id", &DecodedMVCFrame::frame_id)
        .def_readonly("frame_id_mvc", &DecodedMVCFrame::frame_id_mvc)
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
                 py::gil_scoped_release release;
                 return self.decodeAnnexBStream(
                     static_cast<const uint8_t*>(info.ptr),
                     static_cast<size_t>(info.size)
                 );
             },
             "Decode a full Annex B access unit without copying",
             py::arg("data"))
        .def("get_frame",
             [](MVCDecoder& self) -> py::tuple {
                 DecodedMVCFrame frame;
                bool success = false;
                {
                    py::gil_scoped_release release;
                    success = self.getFrame(frame);
                }
                if (!success) {
                    return py::make_tuple(false, py::none());
                }
                return py::make_tuple(true, frame);
             },
             "Get next decoded frame if available. Returns (success, frame)")
        .def("flush", &MVCDecoder::flush,
             "Flush decoder (for seeking)")
        .def("is_initialized", &MVCDecoder::isInitialized,
             "Check if decoder is initialized")
        .def("get_last_error", &MVCDecoder::getLastError,
             "Get last error message");
#else
    // Edge264 not available - do NOT expose MVCDecoder
    // Python code will check hasattr(module, 'MVCDecoder') and fallback to ctypes
    m.attr("EDGE264_UNAVAILABLE") = true;
#endif
}
