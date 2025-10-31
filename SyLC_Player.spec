# -*- mode: python ; coding: utf-8 -*-
import os

# Chemin du répertoire du script
script_dir = os.path.dirname(os.path.abspath('SyLC.py'))

# Liste des binaires MPV et FFmpeg à inclure
binaries_to_include = [
    # MPV
    ('mpv-2.dll', '.'),

    # FFmpeg executables
    ('ffmpeg.exe', '.'),
    ('ffprobe.exe', '.'),

    # FFmpeg DLLs
    ('avcodec-62.dll', '.'),
    ('avformat-62.dll', '.'),
    ('avutil-60.dll', '.'),
    ('avdevice-62.dll', '.'),
    ('avfilter-11.dll', '.'),
    ('swresample-6.dll', '.'),
    ('swscale-9.dll', '.'),
]

# Convertir les chemins relatifs en chemins absolus
binaries = [(os.path.join(script_dir, src), dst) for src, dst in binaries_to_include]

# --- Ajout pour inclure les DLLs (méthode pour venv/uv) ---
import sys
import glob

dll_paths = set()
# Ajouter le dossier racine de l'installation Python de base
dll_paths.add(sys.base_prefix)
# Ajouter le dossier DLLs de la stdlib
dll_paths.add(os.path.join(sys.base_prefix, 'DLLs'))

all_dlls = []
for path in dll_paths:
    if os.path.isdir(path):
        dll_files = glob.glob(os.path.join(path, '*.dll'))
        for dll_file in dll_files:
            all_dlls.append((dll_file, '.'))
# --- Fin de l'ajout ---

a = Analysis(
    ['SyLC.py'],
    pathex=[],
    binaries=binaries + all_dlls,    datas=[],
    hiddenimports=['PySide6', 'mpv', '_ctypes'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# Ajouter splash.png aux données embarquées
a.datas += [('splash.png', 'splash.png', 'DATA')]

# EXE unique (onefile) - splash screen géré en Python
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SyLC_Player',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Désactivé pour éviter les problèmes avec MPV/FFmpeg
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Pas de console pour une application GUI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
)
