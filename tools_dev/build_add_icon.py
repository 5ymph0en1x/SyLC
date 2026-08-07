"""Add a .ico's images to a PE's resources (RT_ICON + RT_GROUP_ICON).
Replicates Nuitka's post-processing step that got blocked by the AV scan,
with a patient retry loop. Usage: add_icon.py <exe> <ico>"""
import ctypes
import struct
import sys
import time

RT_ICON = 3
RT_GROUP_ICON = 14
LANG_NEUTRAL = 0

k32 = ctypes.windll.kernel32
k32.BeginUpdateResourceW.restype = ctypes.c_void_p
k32.BeginUpdateResourceW.argtypes = [ctypes.c_wchar_p, ctypes.c_int]
k32.UpdateResourceW.restype = ctypes.c_int
k32.UpdateResourceW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_ushort, ctypes.c_void_p, ctypes.c_uint]
k32.EndUpdateResourceW.restype = ctypes.c_int
k32.EndUpdateResourceW.argtypes = [ctypes.c_void_p, ctypes.c_int]


def add_icons(exe_path, ico_path):
    data = open(ico_path, 'rb').read()
    reserved, ico_type, count = struct.unpack_from('<HHH', data, 0)
    assert reserved == 0 and ico_type == 1 and count > 0, "not a valid .ico"
    entries = []
    for i in range(count):
        off = 6 + 16 * i
        (w, h, colors, res, planes, bitcount,
         size, img_off) = struct.unpack_from('<BBBBHHII', data, off)
        entries.append((w, h, colors, res, planes, bitcount, size,
                        data[img_off:img_off + size]))

    for attempt in range(10):
        h = k32.BeginUpdateResourceW(exe_path, 0)
        if h:
            break
        time.sleep(2.0)
    else:
        raise OSError(f"BeginUpdateResourceW failed: {ctypes.GetLastError()}")

    # RT_ICON: one resource per image, ids 1..count
    for i, (w, hh, colors, res, planes, bitcount, size, img) in enumerate(entries):
        buf = ctypes.create_string_buffer(img, size)
        if not k32.UpdateResourceW(h, RT_ICON, i + 1, LANG_NEUTRAL, buf, size):
            raise OSError(f"UpdateResourceW RT_ICON #{i+1} failed: {ctypes.GetLastError()}")

    # RT_GROUP_ICON: directory referencing the RT_ICON ids
    grp = struct.pack('<HHH', 0, 1, count)
    for i, (w, hh, colors, res, planes, bitcount, size, _img) in enumerate(entries):
        grp += struct.pack('<BBBBHHIH', w, hh, colors, res, planes, bitcount, size, i + 1)
    gbuf = ctypes.create_string_buffer(grp, len(grp))
    if not k32.UpdateResourceW(h, RT_GROUP_ICON, 1, LANG_NEUTRAL, gbuf, len(grp)):
        raise OSError(f"UpdateResourceW RT_GROUP_ICON failed: {ctypes.GetLastError()}")

    for attempt in range(10):
        if k32.EndUpdateResourceW(h, 0):
            print(f"OK: {count} icons written to {exe_path}")
            return
        time.sleep(2.0)
    raise OSError(f"EndUpdateResourceW failed: {ctypes.GetLastError()}")


if __name__ == '__main__':
    add_icons(sys.argv[1], sys.argv[2])
