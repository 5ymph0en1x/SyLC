/**
 * MinGW-MSVC Compatibility Layer
 *
 * Provides missing symbols that MinGW libraries expect but MSVC doesn't export.
 * This allows linking MinGW-compiled libraries (like edge264) with MSVC.
 */

#ifdef _MSC_VER
#include <setjmp.h>

// Use C linkage to avoid C++ name mangling
extern "C" {

// MinGW's libwinpthread expects __imp__setjmp (pointer to setjmp function)
// We create a wrapper and export a pointer to it
// Suppress C4611: interaction between '_setjmp' and C++ object destruction is not portable
// This is intentional for MinGW compatibility
#pragma warning(push)
#pragma warning(disable: 4611)
int __cdecl mingw_compat_setjmp(jmp_buf env) {
    return _setjmp(env);
}
#pragma warning(pop)

// Export the function pointer as __imp__setjmp
__declspec(dllexport) void* __imp__setjmp = (void*)mingw_compat_setjmp;

} // extern "C"
#endif
