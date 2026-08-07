#include "shader_resources.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdio>

namespace sylc {
namespace {

// Taking the address of module-owned data gives GetModuleHandleEx an address
// inside the PYD. GetModuleHandle(nullptr) would incorrectly search python.exe.
const unsigned char kModuleAnchor = 0;
constexpr WORD kRcDataResourceType = 10;  // RT_RCDATA, explicitly wide below.

std::string resource_error(const char* operation, int resource_id,
                           DWORD win32_error) {
    char buffer[160] = {};
    std::snprintf(buffer, sizeof(buffer),
                  "%s(shader resource %d) failed (win32=%lu)", operation,
                  resource_id, static_cast<unsigned long>(win32_error));
    return std::string(buffer);
}

}  // namespace

bool load_shader_bytecode(int resource_id, ShaderBytecode& out,
                          std::string& err) {
    out = {};

    HMODULE module = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&kModuleAnchor), &module)) {
        err = resource_error("GetModuleHandleExW", resource_id, GetLastError());
        return false;
    }

    const HRSRC resource = FindResourceW(
        module, MAKEINTRESOURCEW(resource_id),
        MAKEINTRESOURCEW(kRcDataResourceType));
    if (!resource) {
        err = resource_error("FindResourceW", resource_id, GetLastError());
        return false;
    }

    const DWORD size = SizeofResource(module, resource);
    if (size == 0) {
        err = resource_error("SizeofResource", resource_id, GetLastError());
        return false;
    }

    const HGLOBAL loaded = LoadResource(module, resource);
    if (!loaded) {
        err = resource_error("LoadResource", resource_id, GetLastError());
        return false;
    }
    const void* bytes = LockResource(loaded);
    if (!bytes) {
        err = resource_error("LockResource", resource_id, GetLastError());
        return false;
    }

    out.data = bytes;
    out.size = static_cast<std::size_t>(size);
    return true;
}

}  // namespace sylc
