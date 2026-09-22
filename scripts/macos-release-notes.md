SyLC 3D Player installer for **Apple Silicon (arm64), macOS 14 or later**.

### Installation

Download `SyLC-7.0.0-macOS-arm64.dmg`, open it, and drag **SyLC 3D Player** to **Applications**.

Includes Python, libmpv, FFmpeg, the MVC decoder, and the **Depth Anything Small 518** model for 2D-to-3D conversion. No separate Python or Homebrew installation is required.

### Changes

- Embedded video playback in the macOS SyLC interface, with fullscreen support and playback controls.
- CoreML depth inference for 2D-to-3D conversion, with side-by-side, top/bottom, and anaglyph output.
- Fixes for shader parameters, depth orientation, texture updates, and temporal depth continuity.
- Bundled application resources and additional model downloads stored outside the application bundle.

### Build status

- Ad hoc signed; **no Developer ID certificate or Apple notarization**. If macOS blocks the app, use System Settings > Privacy & Security > Open Anyway after attempting to open it.
- Observed inference on an Apple M3 Pro was approximately **0.8 depth maps per second**; depth may lag behind video.
- Disk image integrity and the ad hoc signature were verified. The packaged application was not launched for functional testing.
- The installer SHA-256 checksum is attached.
