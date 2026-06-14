#include "matroska_reader.h"
#include <fstream>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <map>
#include <cstdlib>  // V7b++ SYNC FIX: for std::abs

// DIAGNOSTIC: Check if libmatroska is available at compile time
#ifdef HAVE_LIBMATROSKA
    #pragma message("COMPILING WITH LIBMATROSKA SUPPORT")
#else
    #pragma message("WARNING: COMPILING WITHOUT LIBMATROSKA - FALLBACK MODE")
#endif

// Check if libmatroska is available
#ifdef HAVE_LIBMATROSKA
    #include <ebml/StdIOCallback.h>
    #include <ebml/EbmlHead.h>
    #include <ebml/EbmlSubHead.h>
    #include <ebml/EbmlStream.h>
    #include <ebml/EbmlContexts.h>
    #include <matroska/KaxSegment.h>
    #include <matroska/KaxCluster.h>
    #include <matroska/KaxTracks.h>
    #include <matroska/KaxTrackEntryData.h>
    #include <matroska/KaxTrackAudio.h>
    #include <matroska/KaxTrackVideo.h>
    #include <matroska/KaxBlockData.h>
    #include <matroska/KaxSeekHead.h>
    #include <matroska/KaxInfo.h>
    #include <matroska/KaxBlock.h>
    #include <matroska/KaxCues.h>
    #include <matroska/KaxCuesData.h>

    using namespace libebml;
    using namespace libmatroska;

    // --- CUSTOM IO CALLBACK FOR LARGE FILE SUPPORT (>2GB) ---
    class LargeFileIOCallback : public IOCallback {
    private:
        FILE* m_file;
        bool m_owner;

    public:
        LargeFileIOCallback(const char* path, const char* mode) : m_file(nullptr), m_owner(true) {
#ifdef _WIN32
            fopen_s(&m_file, path, mode);
#else
            m_file = fopen(path, mode);
#endif
            if (!m_file) {
                // Handle error or throw
                std::cerr << "[LargeFileIOCallback] Failed to open file: " << path << std::endl;
            }
        }

        ~LargeFileIOCallback() override {
            close();
        }

        // Read bytes
        uint32 read(void* buffer, size_t size) override {
            if (!m_file) return 0;
            return (uint32)fread(buffer, 1, size, m_file);
        }

        // Set file pointer (seek)
        void setFilePointer(int64 offset, seek_mode mode = seek_beginning) override {
            if (!m_file) return;
            int origin = SEEK_SET;
            if (mode == seek_current) origin = SEEK_CUR;
            else if (mode == seek_end) origin = SEEK_END;

#ifdef _WIN32
            _fseeki64(m_file, offset, origin);
#else
            fseeko(m_file, offset, origin);
#endif
        }

        // Write bytes (dummy implementation for read-only)
        size_t write(const void* buffer, size_t size) override {
            return 0;
        }

        // Get file pointer (tell)
        uint64 getFilePointer() override {
            if (!m_file) return 0;
#ifdef _WIN32
            return (uint64)_ftelli64(m_file);
#else
            return (uint64)ftello(m_file);
#endif
        }

        // Close file
        void close() override {
            if (m_file && m_owner) {
                fclose(m_file);
                m_file = nullptr;
            }
        }

        // Optional: set owner
        void setOwner(bool owner) { m_owner = owner; }
    };
#endif

namespace mvc_demux {

#ifdef HAVE_LIBMATROSKA

// Implementation using libmatroska
struct MatroskaReader::Impl {
    std::unique_ptr<IOCallback> ioHandler; // Changed from StdIOCallback to IOCallback
    std::unique_ptr<EbmlStream> ebmlStream;
    KaxSegment* segment;

    std::vector<MatroskaTrack> tracks;
    uint64_t timecodeScale;
    int64_t duration;
    uint64_t fileSize;

    // Current reading position
    KaxCluster* currentCluster;
    uint64_t clusterPosition;
    uint64_t firstClusterPosition;  // V7b FIX: Store first cluster position for seek
    size_t blockIndex;

    bool isOpen;
    int64_t pendingSeekTimestampMs;
    int32_t seekTrackNumber; // -1 for any track
    bool cuesBasedSeek; // V7b+: If true, skip timestamp filtering (Cues already positioned us correctly)
    int64_t startTimestampMs;
    int64_t firstBlockTimestampMs;
    int64_t maxSeenTimestampMs;

    // V7b+ CRITICAL FIX: Cues index for precise seeking
    std::map<int64_t, uint64_t> cuesIndex; // timestamp_ms -> cluster file position
    uint64_t cuesPosition;  // Position of Cues element in file

    // V7b++ SYNC FIX: Store CuePoint timestamp for seek recovery (cluster timecodes can be corrupted)
    int64_t cuesSeekTimestampMs;  // The timestamp from the CuePoint we seeked to
    int blocksAfterSeek;          // Count blocks read after seek to detect first block

    // V7b+++++++ FIX: Store actual frame duration for accurate timestamp recovery
    int64_t frameDurationMs;      // Calculated from video track frameRate (default 33ms = 30fps)

    Impl()
        : segment(nullptr)
        , timecodeScale(1000000)  // Default: 1ms
        , duration(0)
        , fileSize(0)
        , currentCluster(nullptr)
        , clusterPosition(0)
        , firstClusterPosition(0)  // V7b FIX
        , blockIndex(0)
        , isOpen(false)
        , pendingSeekTimestampMs(-1)
        , seekTrackNumber(-1)
        , cuesBasedSeek(false)
        , startTimestampMs(-1)
        , firstBlockTimestampMs(-1)
        , maxSeenTimestampMs(-1)
        , cuesPosition(0)
        , cuesSeekTimestampMs(-1)  // V7b++ SYNC FIX: Store CuePoint timestamp for seek recovery
        , blocksAfterSeek(0)       // V7b++ SYNC FIX: Count blocks read after seek
        , frameDurationMs(33)      // V7b+++++++ FIX: Default 33ms (30fps), updated when track parsed
    {}

    ~Impl() {
        if (currentCluster) {
            delete currentCluster;
        }
        if (segment) {
            delete segment;
        }
    }

    // Helper method to parse track entry
    MatroskaTrack parseTrackEntry(KaxTrackEntry* entry);

    // V7b+ CRITICAL FIX: Load Cues index for precise seeking
    void loadCuesIndex();
};

MatroskaReader::MatroskaReader()
    : impl_(std::make_unique<Impl>())
{
    fprintf(stderr, "========== [MatroskaReader] Constructor WITH libmatroska (LFS PATCHED) ==========\n");
    fflush(stderr);
}

MatroskaReader::~MatroskaReader() {
    close();
}

bool MatroskaReader::open(const std::string& filePath) {
    close();

    try {
        // Use our custom LargeFileIOCallback instead of StdIOCallback
        impl_->ioHandler = std::make_unique<LargeFileIOCallback>(filePath.c_str(), "rb");
        
        // Verify file open
        impl_->ioHandler->setFilePointer(0, seek_end);
        impl_->fileSize = impl_->ioHandler->getFilePointer();
        impl_->ioHandler->setFilePointer(0, seek_beginning);

        if (impl_->fileSize == 0) {
             std::cerr << "[MatroskaReader] Error: File is empty or could not be opened." << std::endl;
             return false;
        }
        
        // Debug output for LFS
        if (impl_->fileSize > 2147483647) {
             std::cerr << "[MatroskaReader] Large file detected (" << (impl_->fileSize / 1024 / 1024) << " MB). Using 64-bit I/O." << std::endl;
        }

        impl_->ebmlStream = std::make_unique<EbmlStream>(*impl_->ioHandler);

        // Read EBML header
        EbmlElement* ebmlHead = impl_->ebmlStream->FindNextID(EBML_INFO(EbmlHead), 0xFFFFFFFFL);
        if (!ebmlHead) {
            return false;
        }

        ebmlHead->SkipData(*impl_->ebmlStream, EBML_CONTEXT(ebmlHead));
        delete ebmlHead;

        // Find segment
        EbmlElement* segment = impl_->ebmlStream->FindNextID(EBML_INFO(KaxSegment), 0xFFFFFFFFFFFFFFFFL);
        if (!segment) {
            return false;
        }

        impl_->segment = static_cast<KaxSegment*>(segment);

        // Read segment info and tracks
        int upperLevel = 0;
        EbmlElement* level1 = impl_->ebmlStream->FindNextElement(EBML_CONTEXT(impl_->segment), upperLevel, 0xFFFFFFFFL, true);


        while (level1) {

            if (EbmlId(*level1) == EBML_ID(KaxInfo)) {
                // Read segment info
                KaxInfo* segInfo = static_cast<KaxInfo*>(level1);

                // CRITICAL: Use a local upperLevel variable for reading KaxInfo
                // to prevent it from affecting the main loop's upperLevel
                int infoUpperLevel = 0;
                EbmlElement* infoChild = impl_->ebmlStream->FindNextElement(EBML_CONTEXT(segInfo), infoUpperLevel, 0xFFFFFFFFL, true);

                while (infoChild) {
                    if (EbmlId(*infoChild) == EBML_ID(KaxTimecodeScale)) {
                        KaxTimecodeScale* tcScale = static_cast<KaxTimecodeScale*>(infoChild);
                        impl_->timecodeScale = uint64(*tcScale);
                    } else if (EbmlId(*infoChild) == EBML_ID(KaxDuration)) {
                        KaxDuration* dur = static_cast<KaxDuration*>(infoChild);
                        impl_->duration = static_cast<int64_t>(double(*dur) * impl_->timecodeScale / 1000000);
                        fprintf(stderr, "[MatroskaReader] Duration found: %lld ms\n", (long long)impl_->duration);
                    }

                    if (infoUpperLevel > 0) {
                        delete infoChild;
                        break;
                    }
                    infoChild = impl_->ebmlStream->FindNextElement(EBML_CONTEXT(segInfo), infoUpperLevel, 0xFFFFFFFFL, true);
                }

            } else if (EbmlId(*level1) == EBML_ID(KaxTracks)) {

                // Read tracks
                KaxTracks* tracks = static_cast<KaxTracks*>(level1);

                // Important: Use a local upperLevel variable for reading KaxTracks
                // to prevent it from affecting the main loop's upperLevel
                int tracksUpperLevel = 0;
                tracks->Read(*impl_->ebmlStream, EBML_CONTEXT(tracks), tracksUpperLevel, level1, true);

                // Now iterate through the loaded track entries
                for (size_t i = 0; i < tracks->ListSize(); i++) {
                    EbmlElement* trackChild = (*tracks)[static_cast<unsigned int>(i)];

                    if (EbmlId(*trackChild) == EBML_ID(KaxTrackEntry)) {
                        KaxTrackEntry* trackEntry = static_cast<KaxTrackEntry*>(trackChild);

                        // Read the track entry to load its children
                        int trackEntryUpperLevel = 0;
                        trackEntry->Read(*impl_->ebmlStream, EBML_CONTEXT(trackEntry), trackEntryUpperLevel, trackChild, true);

                        MatroskaTrack track = impl_->parseTrackEntry(trackEntry);
                        impl_->tracks.push_back(track);

                        // V7b+++++++ FIX: Calculate frame duration from video track
                        // This fixes the hardcoded 33ms assumption for timestamp recovery
                        if (track.trackType == 1 && track.frameRate > 0.0) {
                            impl_->frameDurationMs = static_cast<int64_t>(1000.0 / track.frameRate + 0.5);
                            fprintf(stderr, "[MatroskaReader] V7b+++++++ FIX: Video track %u has frameRate=%.3f fps -> frameDurationMs=%lld\n",
                                    track.trackNumber, track.frameRate, (long long)impl_->frameDurationMs);
                            fflush(stderr);
                        }

                    }
                }

            } else if (EbmlId(*level1) == EBML_ID(KaxCluster)) {
                // Found first cluster, stop here
                impl_->clusterPosition = level1->GetElementPosition();
                impl_->firstClusterPosition = impl_->clusterPosition;  // V7b FIX: Save for seek
                delete level1;
                break;
            }

            if (upperLevel > 0) {
                delete level1;
                break;
            }

            level1->SkipData(*impl_->ebmlStream, EBML_CONTEXT(level1));
            delete level1;
            level1 = impl_->ebmlStream->FindNextElement(EBML_CONTEXT(impl_->segment), upperLevel, 0xFFFFFFFFL, true);
        }

        impl_->isOpen = true;

        // V7b+ CRITICAL FIX: Load Cues index for precise seeking (fixes crash on seek in large files)
        try {
            impl_->loadCuesIndex();
        } catch (const std::exception& e) {
            fprintf(stderr, "[MatroskaReader] WARNING: Could not load Cues index: %s\n", e.what());
            fprintf(stderr, "[MatroskaReader] Seeking will use heuristic fallback (less reliable)\n");
            fflush(stderr);
        } catch (...) {
            fprintf(stderr, "[MatroskaReader] WARNING: Could not load Cues index (unknown error)\n");
            fprintf(stderr, "[MatroskaReader] Seeking will use heuristic fallback (less reliable)\n");
            fflush(stderr);
        }

        return true;

    } catch (const std::exception& e) {
        fprintf(stderr, "[MatroskaReader] ERROR in open(): %s\n", e.what());
        fflush(stderr);
        return false;
    }
}

MatroskaTrack MatroskaReader::Impl::parseTrackEntry(KaxTrackEntry* entry) {
    MatroskaTrack track;
    track.trackNumber = 0;
    track.trackUID = 0;
    track.trackType = 0;
    track.pixelWidth = 0;
    track.pixelHeight = 0;
    track.frameRate = 0.0;
    track.isMVC = false;
    track.mvcSubTrack = 0;

    // Parse track entry elements
    for (size_t i = 0; i < entry->ListSize(); i++) {
        EbmlElement* elem = (*entry)[static_cast<unsigned int>(i)];
        if (EbmlId(*elem) == EBML_ID(KaxTrackNumber)) {
            track.trackNumber = uint32(*static_cast<KaxTrackNumber*>(elem));
        } else if (EbmlId(*elem) == EBML_ID(KaxTrackUID)) {
            track.trackUID = uint32(*static_cast<KaxTrackUID*>(elem));
        } else if (EbmlId(*elem) == EBML_ID(KaxTrackType)) {
            track.trackType = uint8(*static_cast<KaxTrackType*>(elem));
        } else if (EbmlId(*elem) == EBML_ID(KaxCodecID)) {
            track.codecId = std::string(*static_cast<KaxCodecID*>(elem));
            if (track.codecId.size() >= 15 && 
                track.codecId.compare(0, 15, "V_MPEG4/ISO/MVC") == 0) {
                track.isMVC = true;
            }
        } else if (EbmlId(*elem) == EBML_ID(KaxTrackVideo)) {
            KaxTrackVideo* video = static_cast<KaxTrackVideo*>(elem);
            for (size_t j = 0; j < video->ListSize(); j++) {
                EbmlElement* velem = (*video)[static_cast<unsigned int>(j)];
                if (EbmlId(*velem) == EBML_ID(KaxVideoPixelWidth)) {
                    track.pixelWidth = uint32(*static_cast<KaxVideoPixelWidth*>(velem));
                } else if (EbmlId(*velem) == EBML_ID(KaxVideoPixelHeight)) {
                    track.pixelHeight = uint32(*static_cast<KaxVideoPixelHeight*>(velem));
                } else if (EbmlId(*velem) == EBML_ID(KaxVideoFrameRate)) {
                    track.frameRate = double(*static_cast<KaxVideoFrameRate*>(velem));
                }
            }
        }
    }

    // CodecPrivate is a binary element not loaded by the loop above.
    // We must find it and read its data from the stream manually.
    KaxCodecPrivate* codecPriv = FindChild<KaxCodecPrivate>(*entry);
    if (codecPriv) {
        // FIX: The stream pointer is likely in the wrong place after the loop.
        // Seek explicitly to the start of the CodecPrivate data payload.
        ioHandler->setFilePointer(codecPriv->GetElementPosition() + codecPriv->HeadSize());
        
        // Now read the data from the correct position.
        codecPriv->ReadData(*ioHandler);
        const binary* data = codecPriv->GetBuffer();
        size_t size = codecPriv->GetSize();
        if (data && size > 0) {
            track.codecPrivate.assign(data, data + size);
        }
    }

    return track;
}

void MatroskaReader::Impl::loadCuesIndex() {
    // V7b+ CRITICAL FIX: Load Cues element to build precise seek index
    // This solves the crash when seeking to late positions in large files (>2GB)

    fprintf(stderr, "[MatroskaReader] loadCuesIndex() called\n");
    fflush(stderr);

    if (!segment || !isOpen) {
        fprintf(stderr, "[MatroskaReader] WARNING: Cannot load Cues (segment=%p, isOpen=%d)\n",
                (void*)segment, isOpen);
        fflush(stderr);
        return;
    }

    try {

    // First, try to find Cues position from SeekHead
    ioHandler->setFilePointer(firstClusterPosition);

    // Rewind to start of segment to search for SeekHead
    ioHandler->setFilePointer(segment->GetElementPosition() + segment->HeadSize());

    int upperLevel = 0;
    EbmlElement* seekHeadElem = ebmlStream->FindNextID(EBML_INFO(KaxSeekHead), 0xFFFFFFFFL);

    if (seekHeadElem) {
        KaxSeekHead* seekHead = static_cast<KaxSeekHead*>(seekHeadElem);
        seekHead->Read(*ebmlStream, EBML_CONTEXT(seekHead), upperLevel, seekHeadElem, true);

        // Find Cues position in SeekHead
        for (size_t i = 0; i < seekHead->ListSize(); i++) {
            EbmlElement* seekElem = (*seekHead)[static_cast<unsigned int>(i)];
            if (EbmlId(*seekElem) == EBML_ID(KaxSeek)) {
                KaxSeek* seek = static_cast<KaxSeek*>(seekElem);

                KaxSeekID* seekID = FindChild<KaxSeekID>(*seek);
                KaxSeekPosition* seekPos = FindChild<KaxSeekPosition>(*seek);

                if (seekID && seekPos) {
                    EbmlId searchID(seekID->GetBuffer(), seekID->GetSize());
                    if (searchID == EBML_ID(KaxCues)) {
                        cuesPosition = segment->GetElementPosition() + segment->HeadSize() + uint64(*seekPos);
                        fprintf(stderr, "[MatroskaReader] Found Cues position from SeekHead: %llu\n",
                                (unsigned long long)cuesPosition);
                        break;
                    }
                }
            }
        }
        delete seekHeadElem;
    }

    // If we found Cues position, load it
    if (cuesPosition > 0) {
        ioHandler->setFilePointer(cuesPosition);

        EbmlElement* cuesElem = ebmlStream->FindNextID(EBML_INFO(KaxCues), 0xFFFFFFFFL);
        if (cuesElem) {
            KaxCues* cues = static_cast<KaxCues*>(cuesElem);
            upperLevel = 0;
            cues->Read(*ebmlStream, EBML_CONTEXT(cues), upperLevel, cuesElem, true);

            // Parse CuePoints
            int cueCount = 0;
            for (size_t i = 0; i < cues->ListSize(); i++) {
                EbmlElement* cueChild = (*cues)[static_cast<unsigned int>(i)];
                if (EbmlId(*cueChild) == EBML_ID(KaxCuePoint)) {
                    KaxCuePoint* cuePoint = static_cast<KaxCuePoint*>(cueChild);

                    KaxCueTime* cueTime = FindChild<KaxCueTime>(*cuePoint);
                    if (!cueTime) continue;

                    // Convert CueTime (in timecodeScale units) to milliseconds
                    int64_t timestamp_ms = (uint64(*cueTime) * timecodeScale) / 1000000;

                    // Find CueTrackPositions
                    KaxCueTrackPositions* trackPos = FindChild<KaxCueTrackPositions>(*cuePoint);
                    if (trackPos) {
                        KaxCueClusterPosition* clusterPos = FindChild<KaxCueClusterPosition>(*trackPos);
                        if (clusterPos) {
                            uint64_t segmentPos = segment->GetElementPosition();
                            uint64_t segmentHeadSize = segment->HeadSize();
                            uint64_t clusterOffset = uint64(*clusterPos);
                            uint64_t filePos = segmentPos + segmentHeadSize + clusterOffset;

                            if (cueCount < 3) {  // Debug first 3 entries
                                fprintf(stderr, "[MatroskaReader] CuePoint #%d: timestamp=%lld ms\n", cueCount, (long long)timestamp_ms);
                                fprintf(stderr, "  segmentPos=%llu, segmentHeadSize=%llu, clusterOffset=%llu\n",
                                        (unsigned long long)segmentPos, (unsigned long long)segmentHeadSize, (unsigned long long)clusterOffset);
                                fprintf(stderr, "  calculated filePos=%llu\n", (unsigned long long)filePos);
                            }

                            cuesIndex[timestamp_ms] = filePos;
                            cueCount++;
                        }
                    }
                }
            }

            fprintf(stderr, "[MatroskaReader] Loaded %d CuePoints from Cues index\n", cueCount);
            delete cuesElem;
        } else {
            fprintf(stderr, "[MatroskaReader] WARNING: Cues element not found at expected position\n");
        }
    } else {
        fprintf(stderr, "[MatroskaReader] WARNING: No Cues index found (SeekHead missing or incomplete)\n");
        fprintf(stderr, "[MatroskaReader] Seeking will use heuristic fallback (less reliable for large files)\n");
    }

    // Restore file pointer to first cluster for normal reading
    ioHandler->setFilePointer(firstClusterPosition);

    } catch (const std::exception& e) {
        fprintf(stderr, "[MatroskaReader] ERROR in loadCuesIndex(): %s\n", e.what());
        fflush(stderr);
    } catch (...) {
        fprintf(stderr, "[MatroskaReader] ERROR in loadCuesIndex(): Unknown exception\n");
        fflush(stderr);
    }
}

void MatroskaReader::close() {
    if (impl_->isOpen) {
        impl_->isOpen = false;
        // Cleanup handled by smart pointers and destructor
    }
}

bool MatroskaReader::isOpen() const {
    return impl_->isOpen;
}

std::vector<MatroskaTrack> MatroskaReader::getTracks() const {
    return impl_->tracks;
}

MatroskaTrack MatroskaReader::getTrack(uint32_t trackNumber) const {
    for (const auto& track : impl_->tracks) {
        if (track.trackNumber == trackNumber) {
            return track;
        }
    }
    return MatroskaTrack();
}

bool MatroskaReader::readNextBlock(MatroskaBlock& block) {
    if (!impl_->isOpen) {
        return false;
    }

    auto should_skip_timestamp = [this](int64_t timestamp_ms, uint32_t track_num) -> bool {
        if (impl_->pendingSeekTimestampMs < 0) {
            return false;
        }

        // V7b+ CUES FIX: When using Cues-based seek, skip timestamp filtering entirely
        // Cues already positioned us at the correct cluster, and timestamps may be corrupted
        if (impl_->cuesBasedSeek) {
            // Just accept the first keyframe we find, no timestamp check
            impl_->pendingSeekTimestampMs = -1;
            impl_->seekTrackNumber = -1;
            impl_->cuesBasedSeek = false;
            fprintf(stderr, "[MatroskaReader] Cues-based seek: Accepting first keyframe (timestamp validation skipped)\n");
            return false;
        }

        // If filtering by track, skip any block that doesn't match
        if (impl_->seekTrackNumber != -1 && (int32_t)track_num != impl_->seekTrackNumber) {
            return true;
        }

        // 1. Check if we are too early (standard seek behavior)
        // Allow 2ms tolerance for float rounding
        if (timestamp_ms + 2 < impl_->pendingSeekTimestampMs) {
            return true;
        }

        // 2. V7b GARBAGE CHECK: Check if we are WAY too late
        // If the timestamp is > target + 60s, it's likely a garbage timestamp or out-of-order block.
        // We should SKIP it and continue scanning for the real target.
        if (timestamp_ms > impl_->pendingSeekTimestampMs + 60000) {
             // Log occasionally to avoid flooding, but warn about bad blocks
             static int garbage_log_counter = 0;
             if (++garbage_log_counter % 500 == 0) {
                 fprintf(stderr, "[MatroskaReader] Skipping garbage/overshoot: %lld ms (Target: %lld ms)\n",
                         (long long)timestamp_ms, (long long)impl_->pendingSeekTimestampMs);
             }
             return true;
        }

        // 3. Match found (timestamp is >= target AND <= target + 60s)
        fprintf(stderr, "[MatroskaReader] Seek target reached! Found %lld ms (Target: %lld ms)\n",
                (long long)timestamp_ms, (long long)impl_->pendingSeekTimestampMs);

        impl_->pendingSeekTimestampMs = -1;
        impl_->seekTrackNumber = -1;
        return false;
    };

    int garbage_retry_count = 0; // V7b FIX 4: Track consecutive failures in this read attempt

    while (true) {
        // If we don't have a current cluster, seek to the first one or next one
        if (!impl_->currentCluster) {
            uint64_t currentPos = impl_->ioHandler->getFilePointer();

            // V7b+ CUES PROTECTION: Stop reading if we reached the Cues index position
            // Cues are metadata, not video clusters, and must not be parsed as clusters
            if (impl_->cuesPosition > 0 && impl_->clusterPosition >= impl_->cuesPosition) {
                fprintf(stderr, "[MatroskaReader] Reached Cues position (%llu). Stopping sequential read.\n",
                        (unsigned long long)impl_->cuesPosition);
                fflush(stderr);
                return false; // End of video stream
            }

            // V7b FIX: If positioned exactly at cluster (from seek), create it directly
            // Otherwise use FindNextID for sequential reading
            EbmlElement* clusterElem = nullptr;

            if (currentPos == impl_->clusterPosition) {
                // Direct cluster creation at known position (from seek)
                int upperLevel = 0;
                clusterElem = impl_->ebmlStream->FindNextElement(EBML_CONTEXT(impl_->segment), upperLevel, 0xFFFFFFFFL, true);

                if (clusterElem && EbmlId(*clusterElem) == EBML_ID(KaxCluster)) {
                    // Direct creation successful
                } else {
                    if (clusterElem) delete clusterElem;
                    clusterElem = nullptr;
                }
            }

            // Fallback: FindNextID for sequential reading or if direct creation failed
            if (!clusterElem) {
                // V7b FIX 2: If Direct Creation failed (while positioned exactly at clusterPosition),
                // it means FindNextElement rejected the element at this position (False Positive ID?).
                // We MUST NOT force the pointer back to clusterPosition, or FindNextID will find the same garbage.
                // Instead, we should advance slightly to search for the NEXT valid cluster.
                if (currentPos == impl_->clusterPosition) {
                     // We failed at the exact expected position.
                     // Advance 1 byte to avoid finding the same false positive.
                     impl_->ioHandler->setFilePointer(impl_->clusterPosition + 1);
                } else {
                     // Normal sequential reading: ensure we start searching from expected position
                     impl_->ioHandler->setFilePointer(impl_->clusterPosition);
                }

                clusterElem = impl_->ebmlStream->FindNextID(EBML_INFO(KaxCluster), 0xFFFFFFFFL);
                if (!clusterElem) {
                    return false; // No more clusters
                }

                // V7b FIX 3: Sanity Check for False Positives (Garbage Size)
                // Uses the shared garbage_retry_count
                if (clusterElem->IsFiniteSize() && clusterElem->GetSize() > 500 * 1024 * 1024) { // > 500MB
                     garbage_retry_count++;
                     delete clusterElem;
                     clusterElem = nullptr;

                     // Anti-Stuck Logic: If we find too many garbage clusters in a row, jump ahead
                     if (garbage_retry_count > 5) {
                         uint64_t jump_size = 1024 * 1024; // 1MB
                         impl_->ioHandler->setFilePointer(impl_->clusterPosition + jump_size);
                         impl_->clusterPosition = impl_->ioHandler->getFilePointer();
                         garbage_retry_count = 0; // Reset
                     } else {
                         // Update position to continue search from where FindNextID left off (or +1)
                         impl_->clusterPosition = impl_->ioHandler->getFilePointer();
                     }
                     continue; // Restart loop to find next
                }
                
                // Do NOT reset garbage_retry_count here yet, as ListSize check might fail next.

            }

            impl_->currentCluster = static_cast<KaxCluster*>(clusterElem);
            impl_->blockIndex = 0;

            // Read cluster contents
            int upperLevel = 0;
            EbmlElement* dummyElem = nullptr;
            impl_->currentCluster->Read(*impl_->ebmlStream, EBML_CONTEXT(impl_->currentCluster), upperLevel, dummyElem, true);

            // ╔═══════════════════════════════════════════════════════════════════╗
            // ║  V8 CRITICAL FIX: Initialize cluster timecode context             ║
            // ║  Without InitTimecode(), GlobalTimecode() returns garbage!        ║
            // ║  This was causing "corrupted" timestamps after forward seeks.     ║
            // ╚═══════════════════════════════════════════════════════════════════╝

            // Extract cluster timecode from KaxClusterTimecode child element
            uint64_t clusterTimecode = 0;
            KaxClusterTimecode* tcElem = FindChild<KaxClusterTimecode>(*impl_->currentCluster);
            if (tcElem) {
                clusterTimecode = uint64(*tcElem);
            }

            // Initialize the cluster's timing context
            impl_->currentCluster->InitTimecode(clusterTimecode, impl_->timecodeScale);

            // Sanity check for cluster size. A ridiculously large number indicates a parsing error.
            if (impl_->currentCluster->ListSize() > 100000) {
                garbage_retry_count++;
                fprintf(stderr, "[MatroskaReader] ERROR: Cluster too large (%zu elements). Skipping bad cluster at %llu (Count: %d).\n", 
                        impl_->currentCluster->ListSize(), (unsigned long long)impl_->clusterPosition, garbage_retry_count);
                delete impl_->currentCluster;
                impl_->currentCluster = nullptr;
                
                // Anti-Stuck Logic: If we find too many garbage clusters in a row, jump ahead
                if (garbage_retry_count > 5) {
                     uint64_t jump_size = 1024 * 1024; // 1MB
                     fprintf(stderr, "[MatroskaReader] Too many garbage detections. Jumping forward %llu bytes to escape corruption zone.\n", (unsigned long long)jump_size);
                     impl_->ioHandler->setFilePointer(impl_->clusterPosition + jump_size);
                     impl_->clusterPosition = impl_->ioHandler->getFilePointer();
                     garbage_retry_count = 0; // Reset
                } else {
                     // Advance and retry
                     impl_->ioHandler->setFilePointer(impl_->clusterPosition + 1);
                     impl_->clusterPosition = impl_->ioHandler->getFilePointer();
                }
                
                continue; // Retry finding a cluster
            }

            if (impl_->currentCluster->ListSize() == 0) {
                fprintf(stderr, "[MatroskaReader] WARNING: Cluster is empty, moving to next\n");
                delete impl_->currentCluster;
                impl_->currentCluster = nullptr;
                impl_->clusterPosition = impl_->ioHandler->getFilePointer();
                continue; // Try next cluster
            }
            
            // If we reached here, we have a valid loaded cluster!
            garbage_retry_count = 0;
        }

        // Iterate through cluster elements to find blocks
        while (impl_->blockIndex < impl_->currentCluster->ListSize()) {
            EbmlElement* elem = (*impl_->currentCluster)[static_cast<unsigned int>(impl_->blockIndex++)];

                        if (EbmlId(*elem) == EBML_ID(KaxSimpleBlock)) {
                        KaxSimpleBlock* simpleBlock = static_cast<KaxSimpleBlock*>(elem);

                        // V8 CRITICAL FIX: Set parent cluster so GlobalTimecode() works correctly
                        // Without this, the block doesn't know its cluster's timecode!
                        simpleBlock->SetParent(*impl_->currentCluster);

                        block.trackNumber = simpleBlock->TrackNum();
                        int64_t raw_timestamp = simpleBlock->GlobalTimecode() / 1000000; // ns to ms

                        // ╔═══════════════════════════════════════════════════════════════════╗
                        // ║  V8 INDEX-BASED SYNC: Return RAW timestamps to Python              ║
                        // ║  Python will use getLastCueTimestamp() as single source of truth  ║
                        // ║  C++ no longer "corrects" timestamps - simplification              ║
                        // ╚═══════════════════════════════════════════════════════════════════╝

                        // Track blocks after seek (still useful for debugging)
                        if (impl_->cuesSeekTimestampMs >= 0 && impl_->blocksAfterSeek < 500) {
                            impl_->blocksAfterSeek++;
                        }

                // Extract block data
                // Capture start timestamp if not set
                if (impl_->startTimestampMs == -1) {
                    impl_->startTimestampMs = raw_timestamp;
                    fprintf(stderr, "[MatroskaReader] File Start Timestamp: %lld ms\n", (long long)impl_->startTimestampMs);
                }

                // V8 INDEX-BASED SYNC: Return RAW timestamp
                // Python uses getLastCueTimestamp() as authoritative reference
                // No normalization needed - Python handles T_cues = T_audio = T_video
                block.timestamp = raw_timestamp;
                if (raw_timestamp > impl_->maxSeenTimestampMs) {
                    impl_->maxSeenTimestampMs = raw_timestamp;
                    if (impl_->duration == 0 && impl_->maxSeenTimestampMs > 0) {
                        impl_->duration = impl_->maxSeenTimestampMs;
                    }
                }
                block.isKeyframe = simpleBlock->IsKeyframe();
                block.frameCount = simpleBlock->NumberFrames();

                // LACING FIX: Concatenate ALL laced frames, not just the first one
                // MKV lacing packs multiple frames in a single block
                unsigned int numFrames = simpleBlock->NumberFrames();
                block.data.clear();

                // Debug: Log lacing info for first few blocks
                static int lacingDiagCount = 0;
                if (lacingDiagCount < 20) {
                    size_t totalSize = 0;
                    for (unsigned int f = 0; f < numFrames; ++f) {
                        totalSize += simpleBlock->GetBuffer(f).Size();
                    }
                    fprintf(stderr, "[LACING-DIAG] Block: %u laced frames, total size=%zu bytes\n",
                            numFrames, totalSize);
                    lacingDiagCount++;
                }

                for (unsigned int frameIdx = 0; frameIdx < numFrames; ++frameIdx) {
                    DataBuffer& dataBuffer = simpleBlock->GetBuffer(frameIdx);
                    const uint8_t* frameData = dataBuffer.Buffer();
                    size_t frameSize = dataBuffer.Size();
                    block.data.insert(block.data.end(), frameData, frameData + frameSize);
                }

                if (should_skip_timestamp(raw_timestamp, block.trackNumber)) {
                    continue;
                }

                return true;

            } else if (EbmlId(*elem) == EBML_ID(KaxBlockGroup)) {
                KaxBlockGroup* blockGroup = static_cast<KaxBlockGroup*>(elem);

                            // Find the Block element inside BlockGroup
                            KaxBlock* kaxBlock = FindChild<KaxBlock>(*blockGroup);
                            if (kaxBlock) {
                                // V8 CRITICAL FIX: Set parent cluster so GlobalTimecode() works correctly
                                kaxBlock->SetParent(*impl_->currentCluster);

                                block.trackNumber = kaxBlock->TrackNum();
                                int64_t raw_ts_bg = kaxBlock->GlobalTimecode() / 1000000; // ns to ms

                                // ╔═══════════════════════════════════════════════════════════════════╗
                                // ║  V8 INDEX-BASED SYNC: Return RAW timestamps to Python              ║
                                // ║  Python uses getLastCueTimestamp() as single source of truth      ║
                                // ╚═══════════════════════════════════════════════════════════════════╝

                                // Track blocks after seek (debugging)
                                if (impl_->cuesSeekTimestampMs >= 0 && impl_->blocksAfterSeek < 500) {
                                    impl_->blocksAfterSeek++;
                                }

                                // Capture start timestamp if not set
                                if (impl_->startTimestampMs == -1) {
                                    impl_->startTimestampMs = raw_ts_bg;
                                    fprintf(stderr, "[MatroskaReader] File Start Timestamp: %lld ms\n", (long long)impl_->startTimestampMs);
                                }

                                // V8: Return RAW timestamp (no normalization)
                                block.timestamp = raw_ts_bg;
                                if (raw_ts_bg > impl_->maxSeenTimestampMs) {
                                    impl_->maxSeenTimestampMs = raw_ts_bg;
                                    if (impl_->duration == 0 && impl_->maxSeenTimestampMs > 0) {
                                        impl_->duration = impl_->maxSeenTimestampMs;
                                    }
                                }
                                block.frameCount = kaxBlock->NumberFrames();

                                // Check for ReferenceBlock to determine if keyframe
                                KaxReferenceBlock* refBlock = FindChild<KaxReferenceBlock>(*blockGroup);
                                block.isKeyframe = (refBlock == nullptr);

                                // LACING FIX: Concatenate ALL laced frames (same fix as SimpleBlock)
                                unsigned int numFramesBG = kaxBlock->NumberFrames();
                                block.data.clear();
                                for (unsigned int frameIdx = 0; frameIdx < numFramesBG; ++frameIdx) {
                                    DataBuffer& dataBuffer = kaxBlock->GetBuffer(frameIdx);
                                    const uint8_t* frameData = dataBuffer.Buffer();
                                    size_t frameSize = dataBuffer.Size();
                                    block.data.insert(block.data.end(), frameData, frameData + frameSize);
                                }
                
                                if (should_skip_timestamp(raw_ts_bg, block.trackNumber)) {
                                    continue;
                                }
                
                                return true;
                            }            }
        }

        // Current cluster exhausted, move to next
        delete impl_->currentCluster;
        impl_->currentCluster = nullptr;
        impl_->blockIndex = 0;

        // Update cluster position for next read
        impl_->clusterPosition = impl_->ioHandler->getFilePointer();
        
        // Loop continues to load next cluster
    }
}

bool MatroskaReader::seek(int64_t timestampMs, int32_t trackNumber) {
#ifdef HAVE_LIBMATROSKA
    if (!impl_->isOpen || !impl_->segment) {
        return false;
    }

    if (timestampMs < 0) {
        timestampMs = 0;
    }

    if (impl_->currentCluster) {
        delete impl_->currentCluster;
        impl_->currentCluster = nullptr;
    }

    impl_->blockIndex = 0;
    impl_->seekTrackNumber = trackNumber;

    uint64_t targetPos = impl_->firstClusterPosition;

    try {
        // V7b+ CRITICAL FIX: Use Cues index for precise seeking if available
        if (!impl_->cuesIndex.empty()) {
            // When Cues index is available, timestamps in Cues are the authoritative reference
            // Do NOT adjust by startTimestampMs (which may be corrupted in some files)
            // Cues timestamps are already normalized (0-based), matching the input timestampMs
            impl_->pendingSeekTimestampMs = timestampMs;
            impl_->cuesBasedSeek = true; // V7b+: Skip timestamp filtering for Cues-based seeks

            fprintf(stderr, "[MatroskaReader] Seeking to %lld ms using Cues index (%zu entries)\n",
                    (long long)timestampMs, impl_->cuesIndex.size());
            fprintf(stderr, "[MatroskaReader] NOTE: Using Cues timestamps directly (ignoring startTimestampMs=%lld)\n",
                    (long long)impl_->startTimestampMs);

            // Find the closest CuePoint at or before target timestamp
            // Use the input timestamp directly - Cues timestamps match the normalized (0-based) timeline
            auto it = impl_->cuesIndex.upper_bound(timestampMs);
            if (it != impl_->cuesIndex.begin()) {
                --it; // Get the entry at or before target
                targetPos = it->second;
                // V7b++ SYNC FIX: Store the Cue timestamp - cluster timecodes can be corrupted!
                impl_->cuesSeekTimestampMs = it->first;
                impl_->blocksAfterSeek = 0;  // Reset block counter for this seek
                fprintf(stderr, "[MatroskaReader] Found CuePoint: timestamp=%lld ms -> filePos=%llu\n",
                        (long long)it->first, (unsigned long long)targetPos);
                fprintf(stderr, "[MatroskaReader] V7b++ SYNC FIX: cuesSeekTimestampMs=%lld ms (authoritative)\n",
                        (long long)impl_->cuesSeekTimestampMs);
            } else {
                // Target is before first CuePoint, use first cluster
                targetPos = impl_->firstClusterPosition;
                impl_->cuesSeekTimestampMs = 0;  // Beginning of file
                impl_->blocksAfterSeek = 0;
                fprintf(stderr, "[MatroskaReader] Target before first CuePoint, using first cluster: %llu\n",
                        (unsigned long long)targetPos);
            }

            impl_->clusterPosition = targetPos;
            impl_->currentCluster = nullptr;
            impl_->blockIndex = 0;
            impl_->ioHandler->setFilePointer(targetPos);

            fprintf(stderr, "[MatroskaReader] Positioned at cluster %llu (Cues-based seek)\n",
                    (unsigned long long)targetPos);
            fflush(stderr);
            return true;
        }

        // Fallback: Heuristic seek (less reliable for large files, but works without Cues)
        // For heuristic seek, we DO need to adjust by startTimestampMs
        impl_->pendingSeekTimestampMs = timestampMs + (impl_->startTimestampMs > 0 ? impl_->startTimestampMs : 0);

        fprintf(stderr, "[MatroskaReader] WARNING: No Cues index available, using heuristic seek\n");
        fprintf(stderr, "[MatroskaReader] Adjusting by startTimestampMs: %lld ms -> %lld ms\n",
                (long long)timestampMs, (long long)impl_->pendingSeekTimestampMs);

        int64_t seekDuration = impl_->duration;
        if (seekDuration <= 0 && impl_->maxSeenTimestampMs > 0) {
            seekDuration = impl_->maxSeenTimestampMs;
        }

        if (seekDuration > 0 && impl_->fileSize > impl_->firstClusterPosition) {
            double ratio = static_cast<double>(timestampMs) / static_cast<double>(seekDuration);
            if (ratio < 0.0) ratio = 0.0;
            if (ratio > 0.95) ratio = 0.95; // leave room for final clusters
            uint64_t approx = impl_->firstClusterPosition +
                              static_cast<uint64_t>((impl_->fileSize - impl_->firstClusterPosition) * ratio);

            targetPos = approx;
        }

        fprintf(stderr, "[MatroskaReader] Seeking to %lld ms, estimated file pos %llu (duration=%lld ms, size=%llu)\n",
                (long long)timestampMs,
                (unsigned long long)targetPos,
                (long long)impl_->duration,
                (unsigned long long)impl_->fileSize);

        // Search backward from target position to find a valid cluster
        const uint64_t MAX_BACKTRACK = 10 * 1024 * 1024; // 10 MB
        const uint64_t STEP_SIZE = 64 * 1024; // 64 KB steps

        uint64_t searchPos = targetPos;
        uint64_t minPos = (targetPos > MAX_BACKTRACK) ? (targetPos - MAX_BACKTRACK) : impl_->firstClusterPosition;

        bool clusterFound = false;

        while (searchPos >= minPos && searchPos >= impl_->firstClusterPosition) {
            impl_->ioHandler->setFilePointer(searchPos);

            uint64_t scanLimit = STEP_SIZE * 2;
            EbmlElement* testCluster = impl_->ebmlStream->FindNextID(EBML_INFO(KaxCluster), scanLimit);

            if (testCluster) {
                if (testCluster->IsFiniteSize() && testCluster->GetSize() > 500 * 1024 * 1024) {
                    fprintf(stderr, "[MatroskaReader] Seek: Ignoring huge/garbage cluster at %llu (Size: %llu)\n",
                            (unsigned long long)testCluster->GetElementPosition(), (unsigned long long)testCluster->GetSize());
                    delete testCluster;
                    testCluster = nullptr;
                } else {
                    targetPos = testCluster->GetElementPosition();
                    delete testCluster;
                    clusterFound = true;
                    fprintf(stderr, "[MatroskaReader] Found cluster at pos %llu (backtracked %llu bytes from search pos %llu)\n",
                            (unsigned long long)targetPos,
                            (unsigned long long)(searchPos > targetPos ? searchPos - targetPos : 0),
                            (unsigned long long)searchPos);
                    break;
                }
            }

            if (searchPos < STEP_SIZE) break;
            searchPos -= STEP_SIZE;
        }

        if (!clusterFound) {
            fprintf(stderr, "[MatroskaReader] WARNING: No cluster found in backtrack, using first cluster\n");
            targetPos = impl_->firstClusterPosition;
        }

        impl_->clusterPosition = targetPos;
        impl_->currentCluster = nullptr;
        impl_->blockIndex = 0;
        impl_->ioHandler->setFilePointer(targetPos);

        fprintf(stderr, "[MatroskaReader] Positioned at cluster %llu (heuristic seek)\n",
                (unsigned long long)targetPos);

        fflush(stderr);
        return true;
    } catch (std::exception& e) {
        fprintf(stderr, "[MatroskaReader] Seek exception: %s\n", e.what());
        impl_->pendingSeekTimestampMs = -1;
        impl_->seekTrackNumber = -1;
        return false;
    } catch (...) {
        fprintf(stderr, "[MatroskaReader] Seek failed with unknown exception\n");
        impl_->pendingSeekTimestampMs = -1;
        impl_->seekTrackNumber = -1;
        return false;
    }
#else
    (void)timestampMs;
    (void)trackNumber;
    return false;
#endif
}

// ... (rest of functions) ...

int64_t MatroskaReader::getDuration() const {
    return impl_->duration;
}

uint64_t MatroskaReader::getTimecodeScale() const {
    return impl_->timecodeScale;
}

void MatroskaReader::setExternalDurationMs(int64_t durationMs) {
    if (durationMs > 0) {
        // V7b FIX: If internal duration looks bogus (> 10 hours) or is just zero, 
        // prefer the external hint which is usually accurate (from ffprobe/mpv).
        // Some files have timestamps acting as duration, leading to massive values.
        if (impl_->duration > 36000000 || impl_->duration == 0) {
             fprintf(stderr, "[MatroskaReader] Overwriting suspicious internal duration %lld ms with external hint %lld ms\n", 
                     (long long)impl_->duration, (long long)durationMs);
             impl_->duration = durationMs;
        } else {
             // Standard behavior: take the max, but don't let a huge internal value win if it's crazy
             if (impl_->duration > durationMs * 2) {
                 impl_->duration = durationMs; // Internal is likely wrong (e.g. absolute timestamp)
             } else {
                 impl_->duration = std::max<int64_t>(impl_->duration, durationMs);
             }
        }
    }
}

bool MatroskaReader::rewind_after_failed_seek(int64_t timestampMs, uint32_t ms_backoff) {
#ifdef HAVE_LIBMATROSKA
    int64_t newTs = timestampMs - static_cast<int64_t>(ms_backoff);
    if (newTs < 0) newTs = 0;
    return seek(newTs, impl_->seekTrackNumber);
#else
    return false;
#endif
}

int64_t MatroskaReader::getLastCueTimestamp() const {
    // ╔═══════════════════════════════════════════════════════════════════╗
    // ║  V8 INDEX-BASED SYNC: Return Cue timestamp as single source       ║
    // ║  of truth for synchronization (T_cues = T_audio = T_video)        ║
    // ╚═══════════════════════════════════════════════════════════════════╝
    return impl_->cuesSeekTimestampMs;  // -1 if no seek or Cues unavailable
}

std::vector<int64_t> MatroskaReader::getCuesTimestamps() const {
    // ╔═══════════════════════════════════════════════════════════════════╗
    // ║  V8 SEEK OPTIMIZATION: Return all Cue timestamps for Python       ║
    // ║  This allows Python to navigate directly between keyframes        ║
    // ╚═══════════════════════════════════════════════════════════════════╝
    std::vector<int64_t> timestamps;
    timestamps.reserve(impl_->cuesIndex.size());
    // Use explicit iterator for MSVC compatibility
    std::map<int64_t, uint64_t>::const_iterator it;
    for (it = impl_->cuesIndex.begin(); it != impl_->cuesIndex.end(); ++it) {
        timestamps.push_back(it->first);
    }
    return timestamps;  // Already sorted since std::map keys are ordered
}

bool MatroskaReader::seekToCue(int64_t cueTimestampMs) {
    // ╔═══════════════════════════════════════════════════════════════════╗
    // ║  V8 SEEK OPTIMIZATION: Seek directly to a known Cue position      ║
    // ║  This is faster than seek() because we skip the binary search     ║
    // ╚═══════════════════════════════════════════════════════════════════╝
    if (!impl_->isOpen || !impl_->segment) {
        return false;
    }

    std::map<int64_t, uint64_t>::iterator it = impl_->cuesIndex.find(cueTimestampMs);
    if (it == impl_->cuesIndex.end()) {
        fprintf(stderr, "[MatroskaReader] seekToCue: Timestamp %lld ms not found in Cues index\n",
                (long long)cueTimestampMs);
        return false;
    }

    // Clean up current cluster
    if (impl_->currentCluster) {
        delete impl_->currentCluster;
        impl_->currentCluster = nullptr;
    }

    uint64_t targetPos = it->second;
    impl_->clusterPosition = targetPos;
    impl_->blockIndex = 0;
    impl_->cuesSeekTimestampMs = cueTimestampMs;
    impl_->blocksAfterSeek = 0;
    impl_->cuesBasedSeek = true;  // Skip timestamp filtering
    impl_->pendingSeekTimestampMs = cueTimestampMs;
    impl_->seekTrackNumber = -1;

    impl_->ioHandler->setFilePointer(targetPos);

    fprintf(stderr, "[MatroskaReader] seekToCue: Positioned at Cue %lld ms -> filePos %llu\n",
            (long long)cueTimestampMs, (unsigned long long)targetPos);
    fflush(stderr);
    return true;
}

#else  // !HAVE_LIBMATROSKA

// Fallback implementation without libmatroska
struct MatroskaReader::Impl {
    std::ifstream file;
    std::vector<MatroskaTrack> tracks;
    bool isOpen;

    Impl() : isOpen(false) {}
};

MatroskaReader::MatroskaReader()
    : impl_(std::make_unique<Impl>())
{
    fprintf(stderr, "========== [MatroskaReader] Constructor WITHOUT libmatroska (FALLBACK) ==========\n");
    fflush(stderr);
}

MatroskaReader::~MatroskaReader() {
    close();
}

bool MatroskaReader::open(const std::string& filePath) {
    close();

    impl_->file.open(filePath, std::ios::binary);
    if (!impl_->file.is_open()) {
        return false;
    }

    // Create a dummy video track
    MatroskaTrack track;
    track.trackNumber = 1;
    track.trackUID = 1;
    track.trackType = 1;  // Video
    track.codecId = "V_MPEG4/ISO/AVC";
    track.pixelWidth = 1920;
    track.pixelHeight = 1080;
    track.frameRate = 23.976;
    track.isMVC = false;
    track.mvcSubTrack = 0;

    impl_->tracks.push_back(track);
    impl_->isOpen = true;

    return true;
}

void MatroskaReader::close() {
    if (impl_->isOpen) {
        impl_->file.close();
        impl_->isOpen = false;
    }
}

bool MatroskaReader::isOpen() const {
    return impl_->isOpen;
}

std::vector<MatroskaTrack> MatroskaReader::getTracks() const {
    return impl_->tracks;
}

MatroskaTrack MatroskaReader::getTrack(uint32_t trackNumber) const {
    for (const auto& track : impl_->tracks) {
        if (track.trackNumber == trackNumber) {
            return track;
        }
    }
    return MatroskaTrack();
}

bool MatroskaReader::readNextBlock(MatroskaBlock& block) {
    // Simplified fallback - just read chunks
    const size_t chunkSize = 1024 * 1024;
    std::vector<uint8_t> buffer(chunkSize);

    impl_->file.read(reinterpret_cast<char*>(buffer.data()), chunkSize);
    size_t bytesRead = impl_->file.gcount();

    if (bytesRead == 0) {
        return false;
    }

    buffer.resize(bytesRead);
    block.trackNumber = 1;
    block.timestamp = 0;
    block.data = std::move(buffer);
    block.isKeyframe = false;
    block.frameCount = 1;

    return true;
}

bool MatroskaReader::seek(int64_t timestampMs, int32_t trackNumber) {
    return false;
}

int64_t MatroskaReader::getDuration() const {
    return 0;
}

uint64_t MatroskaReader::getTimecodeScale() const {
    return 1000000;
}

void MatroskaReader::setExternalDurationMs(int64_t durationMs) {
    // Fallback: no-op
}

bool MatroskaReader::rewind_after_failed_seek(int64_t timestampMs, uint32_t ms_backoff) {
    // Fallback: no-op, return false
    return false;
}

int64_t MatroskaReader::getLastCueTimestamp() const {
    // Fallback: no Cues support
    return -1;
}

std::vector<int64_t> MatroskaReader::getCuesTimestamps() const {
    // Fallback: no Cues support
    return std::vector<int64_t>();
}

bool MatroskaReader::seekToCue(int64_t cueTimestampMs) {
    // Fallback: no Cues support
    (void)cueTimestampMs;
    return false;
}

#endif  // HAVE_LIBMATROSKA

} // namespace mvc_demux
