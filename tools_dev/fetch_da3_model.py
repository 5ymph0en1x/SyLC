# tools_dev/fetch_da3_model.py
#
# LICENSE CHECK (done 2026-07-28, verify again if you re-run this later):
#   Source weights: https://huggingface.co/depth-anything/DA3-SMALL
#     -> model card license tag: apache-2.0
#   ONNX export used: https://huggingface.co/onnx-community/depth-anything-v3-small
#     -> repo license tag: apache-2.0 (re-export of the Small checkpoint only;
#        does NOT touch DA3-BASE/LARGE/GIANT, which are CC-BY-NC and must
#        NEVER be redistributed by SyLC).
#
"""Fetches a Depth Anything V3 SMALL ONNX export (fp32 I/O) into
models/da3_small.onnx. DA3-Small is Apache-2.0 (verify the model card on first
fetch!); LARGE/GIANT are CC-BY-NC and must NEVER be redistributed by SyLC.

Tries known single-file community export URLs first (CANDIDATES), sha256-gated
against EXPECTED_SHA256 exactly like the HF split-merge route below (a
same-size, wrong-content file -- a moved/replaced release asset, or a
compromised mirror -- is rejected, not silently accepted just for being
>10MB). None of those resolved when this script was last written (2026-07-28:
both devin-lai/Depth-Anything-3-Onnx and MoonCodeMaster/Depth-Anything-3-Onnx
have no release assets published). What DID work is the official ONNX Community
export on Hugging Face, which ships as two files (a small graph + an external
weights blob, per the ONNX "external data" convention). The URLs are pinned to
commit 0b6a7f3bf5595f9950b91389e0da3a0de130324c (the repo's tip as of
2026-07-28, per the HF API) rather than the live "main" branch, so a future
re-run can't silently pull different bytes than what was license-verified
today:
  https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/0b6a7f3bf5595f9950b91389e0da3a0de130324c/onnx/model.onnx
  https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/0b6a7f3bf5595f9950b91389e0da3a0de130324c/onnx/model.onnx_data
Its config.json pins `"transformers.js_config": {"dtype": "fp32", ...}` so the
weights are genuine float32 (not upcast fp16). The merged output's sha256 is
also pinned (EXPECTED_SHA256 below) and checked after every fresh merge.

Merging those two files into the single self-contained models/da3_small.onnx
that Task 2 loads requires the `onnx` pip package (protobuf wrapper) to
combine the external-data blob back into the graph. Per project rule, this
script NEVER pip-installs into the project venv. If `onnx` isn't importable
here, create a disposable venv elsewhere and point ONNX_PY at its python:

    G:\\SyLC-main\\.venv\\Scripts\\python.exe -m venv C:\\temp\\onnx_merge_venv
    C:\\temp\\onnx_merge_venv\\Scripts\\python.exe -m pip install --only-binary=:all: onnx
    C:\\temp\\onnx_merge_venv\\Scripts\\python.exe tools_dev\\fetch_da3_model.py

NOTE on the resulting I/O shape: this is the real multi-view DA3 architecture,
so the ONNX graph's declared shapes are 5D, not the naive [1,3,518,518]:
    input  "pixel_values"    : [batch_size, num_images, 3, height, width]
    output "predicted_depth" : [batch_size, num_images, height, width]
    output "confidence"      : [batch_size, num_images, height, width]
    output "extrinsics"      : [batch_size, num_images, 3, 4]
    output "intrinsics"      : [batch_size, num_images, 3, 3]
Height/width are dynamic (not baked to 518), so Task 2 gets [1,3,518,518] by
resizing to 518x518 and feeding it as pixel_values[1,1,3,518,518] (batch=1,
num_images=1 for monocular use), then reading ONLY "predicted_depth" (ignore
confidence/extrinsics/intrinsics — those exist because the model natively
supports multi-view/video input; a single image is a degenerate 1-view case).
Verified end-to-end with a real onnxruntime CPU inference call: output shape
came back as (1, 1, 518, 518) float32 for a [1,1,3,518,518] float32 input.

If none of this works, fall back to the manual export route (any machine):
  1. git clone https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt
  2. Follow onnx/README.md with a Python 3.12 venv (torch): export the SMALL
     model at 518x518, fp32 inputs/outputs. That route exports a simpler
     monocular [B,3,H,W] -> depth [B,1,H,W] + sky [B,1,H,W] interface instead
     of the 5D multi-view one above, if Task 2 would rather have that shape.
  3. Copy the resulting .onnx to G:\\SyLC-main\\models\\da3_small.onnx
LICENSE: bundle ONLY the Small (Apache-2.0) model. Check the HF model card.
"""
import hashlib
import os
import sys
import urllib.request

DEST = os.path.join(os.path.dirname(__file__), "..", "models", "da3_small.onnx")

# Candidate direct single-file download URLs, best first. None resolved as of
# 2026-07-28 (no release assets published by either repo) — kept so future
# runs retry them cheaply before falling back to the HF split-file path.
# sha256-gated against EXPECTED_SHA256 in _try_single_file_candidates() below --
# a release asset appearing here later is verified, not trusted on size alone.
CANDIDATES = [
    # devin-lai/Depth-Anything-3-Onnx release assets (community export of DA3)
    "https://github.com/devin-lai/Depth-Anything-3-Onnx/releases/latest/download/da3_small.onnx",
    "https://github.com/devin-lai/Depth-Anything-3-Onnx/releases/latest/download/depth_anything_v3_small.onnx",
]

# The path that actually worked: official ONNX Community export, split into a
# graph file + an external-data blob (ONNX "external data" convention).
# Pinned to a specific commit (NOT "main" tip) so a future re-run can't
# silently pull different bytes than what was license-verified on 2026-07-28.
# SHA from https://huggingface.co/api/models/onnx-community/depth-anything-v3-small
# (field "sha"; repo's lastModified was 2025-11-15, well before this pin).
HF_REVISION = "0b6a7f3bf5595f9950b91389e0da3a0de130324c"  # pinned 2026-07-28
HF_GRAPH_URL = f"https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/{HF_REVISION}/onnx/model.onnx"
HF_DATA_URL = f"https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/{HF_REVISION}/onnx/model.onnx_data"

# sha256 of the merged models/da3_small.onnx produced from HF_REVISION above,
# recorded the day this script last completed a successful merge
# (2026-07-28). Checked right after the atomic replace in
# _try_hf_split_merge(): on mismatch the newly-written file is deleted and
# the merge is treated as failed.
EXPECTED_SHA256 = "b622801c69643a7e32fafbba4190b3c6de0a3f06b6d8cc566d4a582035003b65"

MANUAL = """
No candidate URL worked. Manual export route (once, any machine):
  1. git clone https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt
  2. Follow onnx/README.md with a Python 3.12 venv (torch): export the SMALL
     model at 518x518, fp32 inputs/outputs.
  3. Copy the resulting .onnx to G:\\SyLC-main\\models\\da3_small.onnx
LICENSE: bundle ONLY the Small (Apache-2.0) model. Check the HF model card.
"""

MANUAL_MERGE = """
Downloaded the HF split files but could not merge them: `onnx` is not
importable in this interpreter, and this script refuses to pip-install into
the project venv. Create a disposable venv anywhere else and rerun there:

    G:\\SyLC-main\\.venv\\Scripts\\python.exe -m venv C:\\temp\\onnx_merge_venv
    C:\\temp\\onnx_merge_venv\\Scripts\\python.exe -m pip install --only-binary=:all: onnx
    C:\\temp\\onnx_merge_venv\\Scripts\\python.exe tools_dev\\fetch_da3_model.py
"""


def _already_fetched():
    if not os.path.exists(DEST):
        return False
    size = os.path.getsize(DEST)
    if size <= 10_000_000:
        return False
    with open(DEST, "rb") as f:
        head = f.read(1)
    return head == b"\x08"


def _try_single_file_candidates():
    # Download to a sibling temp path and only os.replace() it onto DEST once
    # fully written and size-validated, so a killed process / full disk can
    # never leave a partial file sitting at DEST (which _already_fetched()
    # would otherwise mistake for a complete model on the next run).
    tmp = DEST + ".part"
    for url in CANDIDATES:
        try:
            print(f"[fetch] trying {url}")
            urllib.request.urlretrieve(url, tmp)
            size = os.path.getsize(tmp)
            if size < 10_000_000:
                print(f"[fetch] too small ({size}B), rejecting")
                os.remove(tmp)
                continue
            sha = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
            # F6: >10MB alone used to be accepted as "the model" with no content
            # check -- a moved/replaced release asset (or a compromised mirror)
            # would have been promoted straight to DEST. Gate on the same
            # EXPECTED_SHA256 the HF split-merge route below is checked against.
            if EXPECTED_SHA256 and sha != EXPECTED_SHA256:
                print(f"[fetch] sha256 MISMATCH: got {sha}, expected "
                      f"{EXPECTED_SHA256} — rejecting {url}")
                os.remove(tmp)
                continue
            os.replace(tmp, DEST)
            print(f"[fetch] OK: {size/1e6:.1f} MB sha256={sha}")
            return True
        except Exception as e:
            print(f"[fetch] failed: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
    return False


def _try_hf_split_merge():
    try:
        import onnx
    except ImportError:
        print(MANUAL_MERGE)
        return False

    tmp_dir = os.path.join(os.path.dirname(DEST), ".da3_fetch_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    graph_path = os.path.join(tmp_dir, "model.onnx")
    data_path = os.path.join(tmp_dir, "model.onnx_data")
    # onnx.save() writes to a sibling temp path; only os.replace() promotes it
    # to DEST once the size check (and, below, the sha256 check) passes — an
    # interrupted save can never leave a partial file sitting at DEST.
    merge_tmp = DEST + ".part"
    try:
        print(f"[fetch] downloading {HF_GRAPH_URL}")
        urllib.request.urlretrieve(HF_GRAPH_URL, graph_path)
        print(f"[fetch] downloading {HF_DATA_URL} (~100MB)")
        urllib.request.urlretrieve(HF_DATA_URL, data_path)

        model = onnx.load(graph_path, load_external_data=True)
        onnx.checker.check_model(model)
        onnx.save(model, merge_tmp, save_as_external_data=False)

        size = os.path.getsize(merge_tmp)
        if size < 10_000_000:
            print(f"[fetch] merged model too small ({size}B), rejecting")
            os.remove(merge_tmp)
            return False

        os.replace(merge_tmp, DEST)  # atomic within the same directory/volume

        sha = hashlib.sha256(open(DEST, "rb").read()).hexdigest()
        if EXPECTED_SHA256 and sha != EXPECTED_SHA256:
            print(f"[fetch] sha256 MISMATCH: got {sha}, expected {EXPECTED_SHA256} — deleting")
            os.remove(DEST)
            return False

        print(f"[fetch] OK (merged from HF split export): {size/1e6:.1f} MB sha256={sha}")
        return True
    except Exception as e:
        print(f"[fetch] HF split-merge failed: {e}")
        if os.path.exists(merge_tmp):
            os.remove(merge_tmp)
        return False
    finally:
        for p in (graph_path, data_path):
            if os.path.exists(p):
                os.remove(p)
        if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)


def main():
    os.makedirs(os.path.dirname(DEST), exist_ok=True)

    if _already_fetched():
        size = os.path.getsize(DEST)
        print(f"[fetch] already present: {DEST} ({size/1e6:.1f} MB) — skipping")
        return 0

    if _try_single_file_candidates():
        return 0

    if _try_hf_split_merge():
        return 0

    print(MANUAL)
    return 1


if __name__ == "__main__":
    sys.exit(main())
