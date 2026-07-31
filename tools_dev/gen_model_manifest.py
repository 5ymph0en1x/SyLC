# tools_dev/gen_model_manifest.py
"""Regenerates models/MANIFEST.json from the local models/ directory.

Run after re-exporting any graph with da3_to_onnx.py, and again after the
HuggingFace upload to pin the real revision:

    G:\\SyLC-main\\.venv\\Scripts\\python.exe tools_dev\\gen_model_manifest.py
    G:\\SyLC-main\\.venv\\Scripts\\python.exe tools_dev\\gen_model_manifest.py --revision <sha>

The file lists are the two DA3 families, and they are pinned equal to
SyLC_3D_Player's own tables by tests/models/test_manifest_matches_player.py --
a graph the player can resolve but the manifest cannot supply would grey a
preset out with no way for the user to fix it.
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
DEST = os.path.join(MODELS, "MANIFEST.json")

REPO = "Symphoenix/sylc_TRT"
PENDING = "PENDING_UPLOAD"

_SUFFIXES = ("756", "518", "756x406", "756x378", "756x350", "756x322",
             "518x280", "518x266", "518x238", "518x210")

PACKS = {
    "small": ("Small — all three presets",
              tuple(f"da3_small_{s}.onnx" for s in _SUFFIXES)),
    "base": ("Base — maximum quality",
             tuple(f"da3_base_{s}.onnx" for s in _SUFFIXES)),
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=None,
                        help="HuggingFace commit SHA to pin (default: keep "
                             "the current one, or PENDING_UPLOAD)")
    args = parser.parse_args()

    revision = args.revision
    if revision is None and os.path.exists(DEST):
        with open(DEST, encoding="utf-8") as handle:
            revision = json.load(handle).get("revision")
    revision = revision or PENDING

    packs = {}
    for key, (label, names) in PACKS.items():
        files, total = [], 0
        for name in names:
            full = os.path.join(MODELS, name)
            if not os.path.exists(full):
                print(f"MISSING: models/{name}")
                return 1
            size = os.path.getsize(full)
            print(f"  hashing {name} ({size/1e6:.1f} MB)")
            files.append({"path": f"onnx/{key}/{name}", "name": name,
                          "bytes": size, "sha256": _sha256(full)})
            total += size
        packs[key] = {"label": label, "bytes": total, "files": files}
        print(f"{key}: {len(files)} files, {total} bytes")

    with open(DEST, "w", encoding="utf-8") as handle:
        json.dump({"schema": 1, "repo": REPO, "revision": revision,
                   "packs": packs}, handle, indent=2)
        handle.write("\n")
    print(f"wrote {DEST} (revision={revision})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
