# tools_dev/make_github_release.py
"""Assembles a GitHub_v5xx publication tree from an explicit manifest.

A script rather than manual copying: GitHub_v503 shipped 7 edge264 .o object
files, six spare edge264 DLLs and a Gradle cache because it was assembled by
hand. This is re-runnable and reviewable.

    G:\\SyLC-main\\.venv\\Scripts\\python.exe tools_dev\\make_github_release.py
    ... --dest GitHub_v530 --dry-run
"""
import argparse
import ast
import fnmatch
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROOT_FILES = [
    # Stable launcher and project metadata. Application sources, runtime files,
    # assets and build scripts keep their repository-relative directories.
    "SyLC_3D_Player.py",
    "pyproject.toml", "requirements.txt", "requirements-dev.txt", "uv.lock", ".gitignore",
    # Documentation and legal notices
    "README.md", "BUILD.md", "RELEASE_NOTES.md", "LICENSE",
    "NOTICE-THIRD-PARTY.md", "REFACTORING_PLAN.md",
]

ROOT_FILES_OPTIONAL = set()

RUNTIME_FILES = [
    "runtime/mvc_demuxer_cpp.cp314-win_amd64.pyd",
    "runtime/edge264.dll", "runtime/libwinpthread-1.dll",
    "runtime/matroska.dll", "runtime/ebml.dll", "runtime/ffprobe.exe",
    "runtime/avdevice-62.dll", "runtime/avformat-62.dll",
    "runtime/avutil-60.dll", "runtime/swresample-6.dll",
    "runtime/swscale-9.dll", "runtime/onnxruntime.dll",
    "runtime/DirectML.dll",
]

# A source publication can fetch these at runtime. A local archival snapshot
# requested with --complete-runtime carries them beside the player so every
# currently active feature is reproducible without a later download.
COMPLETE_ROOT_FILES = [
    "runtime/mpv-2.dll", "runtime/avcodec-62.dll",
    "runtime/avfilter-11.dll",
]
COMPLETE_TREES = [
    "models_dev_nc/matanyone2",
    "models_dev_nc/matanyone2_runtime",
    "ort_tensorrt",
]

# NOT docs/superpowers: 56 internal task briefs and design specs are development
# artifacts, not part of "only what is necessary". README.md links to
# docs/Z2A_VALIDATION.md, so that one file ships explicitly below.
TREES = [
    "src", "assets", "scripts", "quest_app_native", "tools",
    "mvc_realtime_demuxer", "edge264",
]

EXTRA_FILES = ["models/MANIFEST.json", "docs/Z2A_VALIDATION.md",
               "docs/PROJECT_LAYOUT.md",
               "tools_dev/make_github_release.py",
               "tools_dev/analyze.py", "tools_dev/build_add_icon.py",
               "tools_dev/setup_tensorrt.py", "tools_dev/da3_to_onnx.py",
               "tools_dev/fetch_da3_model.py", "tools_dev/gen_model_manifest.py",
               "tools_dev/hf_repo_docs/README.md",
               "tools_dev/hf_repo_docs/trt_README.md",
               "tools_dev/hf_repo_docs/UPLOAD.md"]

# Matched against the path relative to ROOT, with forward slashes.
EXCLUDE = [
    "*/__pycache__/*", "*.pyc", "*.pyo",
    "*/.git/*", "*/.git", "*/.gradle/*",
    # Gradle build OUTPUT, not the cache: 656 MB across 3008 files, and one of
    # them (app/build/outputs/apk/debug/app-debug.apk, 104 MB) is over GitHub's
    # hard per-file limit, so leaving these in does not merely bloat the clone —
    # it makes the push fail. GitHub_v503 shipped neither.
    "quest_app_native/app/build/*", "quest_app_native/build/*",
    # The built release APK (73.9 MB). A binary artifact belongs in a GitHub
    # Release, not in a source tree; GitHub_v503 did not carry it either.
    "quest_app_native/sylc.apk",
    "edge264/*.o", "edge264/a.args.*",
    "edge264/edge264_dbg.dll", "edge264/edge264_logs.dll",
    "edge264/edge264_mtwait.dll", "edge264/edge264_new.dll",
    "edge264/edge264_candidate.dll", "edge264/edge264_prefix_backup_*.dll",
    "mvc_realtime_demuxer/build*/*",
    "*.bak_*", "*.old_*", "*.oldlocked", "*.part",
    "*/CMakeFiles/*", "*.log",
]


def _excluded(relative):
    unix = relative.replace("\\", "/")
    return any(fnmatch.fnmatch(unix, pattern) for pattern in EXCLUDE)


def _copy(src, dst, dry):
    size = os.path.getsize(src)
    if dry:
        return size
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        source_stat = os.stat(src)
        dest_stat = os.stat(dst)
        if (source_stat.st_size == dest_stat.st_size and
                source_stat.st_mtime_ns == dest_stat.st_mtime_ns):
            return size
    shutil.copy2(src, dst)
    return size


def _unlisted_local_imports():
    """Report ``sylc.*`` imports that do not resolve inside ``src``."""
    source_root = os.path.join(ROOT, "src")
    listed = [os.path.join(ROOT, "SyLC_3D_Player.py")]
    for dirpath, _dirs, files in os.walk(os.path.join(source_root, "sylc")):
        listed.extend(os.path.join(dirpath, name)
                      for name in files if name.endswith(".py"))
    unlisted = {}
    for path in sorted(listed):
        name = os.path.relpath(path, ROOT)
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                tree = ast.parse(handle.read(), filename=name)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            modules = ([a.name for a in node.names]
                       if isinstance(node, ast.Import) else
                       [node.module] if isinstance(node, ast.ImportFrom)
                       and node.level == 0 and node.module else [])
            for imported in modules:
                if not imported.startswith("sylc."):
                    continue
                relative = imported.replace(".", os.sep)
                module_file = os.path.join(source_root, relative + ".py")
                package_file = os.path.join(source_root, relative, "__init__.py")
                if not os.path.isfile(module_file) and not os.path.isfile(package_file):
                    unlisted.setdefault(imported, set()).add(name)
    return unlisted


def _runtime_model_files():
    """The exact twenty ONNX graphs accepted by the player manifest."""
    manifest_path = os.path.join(ROOT, "models", "MANIFEST.json")
    with open(manifest_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    names = []
    for pack in raw.get("packs", {}).values():
        for entry in pack.get("files", []):
            name = entry.get("name")
            if name and name not in names:
                names.append(name)
    return [os.path.join("models", name) for name in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="GitHub_v530")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--complete-runtime", action="store_true",
        help=("include required >100 MB media DLLs, all manifest ONNX graphs, "
              "the offline MatAnyone2 environment and the local TensorRT runtime"))
    args = parser.parse_args()

    dest_root = os.path.join(ROOT, args.dest)
    total, count, missing = 0, 0, []

    selected_files = list(ROOT_FILES) + list(RUNTIME_FILES) + list(EXTRA_FILES)
    if args.complete_runtime:
        selected_files += COMPLETE_ROOT_FILES + _runtime_model_files()

    for name in selected_files:
        src = os.path.join(ROOT, name)
        if not os.path.exists(src):
            if name not in ROOT_FILES_OPTIONAL:
                missing.append(name)
            continue
        total += _copy(src, os.path.join(dest_root, name), args.dry_run)
        count += 1

    selected_trees = list(TREES)
    if args.complete_runtime:
        selected_trees += COMPLETE_TREES
    for tree in selected_trees:
        base = os.path.join(ROOT, tree)
        if not os.path.isdir(base):
            missing.append(tree + "/")
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if not _excluded(
                os.path.relpath(os.path.join(dirpath, d), ROOT) + "/x")]
            for filename in filenames:
                src = os.path.join(dirpath, filename)
                relative = os.path.relpath(src, ROOT)
                if _excluded(relative):
                    continue
                total += _copy(src, os.path.join(dest_root, relative),
                               args.dry_run)
                count += 1

    over = []
    if not args.dry_run and not args.complete_runtime:
        for dirpath, _dirs, filenames in os.walk(dest_root):
            for filename in filenames:
                full = os.path.join(dirpath, filename)
                if os.path.getsize(full) > 100 * 1024 * 1024:
                    over.append((os.path.relpath(full, dest_root),
                                 os.path.getsize(full)))

    print(f"{'would copy' if args.dry_run else 'copied'} {count} files, "
          f"{total/1e6:.0f} MB -> {dest_root}")
    if args.complete_runtime:
        print("complete runtime snapshot: GitHub's 100 MB publication limit "
              "is intentionally not applied")
    if missing:
        print("MISSING (not copied):")
        for name in missing:
            print(f"  {name}")
    if over:
        print("OVER GITHUB'S 100 MB LIMIT -- these will be REJECTED:")
        for name, size in over:
            print(f"  {name}  {size/1e6:.0f} MB")
    unlisted = _unlisted_local_imports()
    if unlisted:
        print("IMPORTED BY SHIPPED CODE BUT NOT IN ROOT_FILES -- the published "
              "tree would raise ImportError:")
        for name, importers in sorted(unlisted.items()):
            print(f"  {name}  <- imported by {', '.join(sorted(importers))}")
    return 1 if (missing or over or unlisted) else 0


if __name__ == "__main__":
    sys.exit(main())
