#pragma once

#include <cstddef>
#include <string>

namespace sylc {

// Non-owning view over immutable RCDATA stored inside mvc_demuxer_cpp.pyd.
// The bytes remain valid for the lifetime of the loaded module.
struct ShaderBytecode {
    const void* data = nullptr;
    std::size_t size = 0;
};

// Loads one compiled shader from this DLL/PYD rather than from the host EXE.
// Returns a detailed Win32 error in `err` when the resource is absent/corrupt.
bool load_shader_bytecode(int resource_id, ShaderBytecode& out,
                          std::string& err);

}  // namespace sylc
