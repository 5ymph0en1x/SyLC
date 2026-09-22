# Build from the repository root with PyInstaller.
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

root = Path(SPECPATH).parent
datas = [(str(root / 'assets'), 'assets'),
         (str(root / 'models/MANIFEST.json'), 'models'),
         (str(root / 'models/da3_small_518.onnx'), 'models'),
         (str(root / 'LICENSE'), '.')]
binaries = [(str(root / 'runtime/libedge264.dylib'), 'runtime'),
            (str(root / 'runtime/mvc_demuxer_cpp.cpython-314-darwin.so'), 'runtime'),
            ('/opt/homebrew/lib/libmpv.2.dylib', 'runtime'),
            ('/opt/homebrew/bin/ffprobe', 'runtime'),
            ('/opt/homebrew/bin/ffmpeg', 'runtime')]
hiddenimports = collect_submodules('sylc')
for package in ('onnxruntime',):
    package_data, package_bins, package_imports = collect_all(package)
    datas += package_data
    binaries += package_bins
    hiddenimports += package_imports
a = Analysis([str(root / 'SyLC_3D_Player.py')],
             pathex=[str(root / 'src'), str(root / 'runtime')],
             binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             runtime_hooks=[str(root / 'scripts/macos_bundle_hook.py')],
             excludes=['torch', 'tensorflow', 'matplotlib', 'pytest', 'IPython',
                       'PyQt5', 'PyQt6', 'PySide2'], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='SyLC',
          debug=False, strip=False, upx=False, console=False, target_arch='arm64')
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='SyLC')
app = BUNDLE(coll, name='SyLC 3D Player.app',
             icon=str(root / 'build/release-macos/SyLC.icns'),
             bundle_identifier='org.sylc.player', version='7.0.0',
             info_plist={'CFBundleShortVersionString': '7.0.0',
                         'CFBundleVersion': '20260922',
                         'NSHighResolutionCapable': True,
                         'LSMinimumSystemVersion': '14.0',
                         'NSPrincipalClass': 'NSApplication'})
