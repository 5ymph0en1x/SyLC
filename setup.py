import requests
import py7zr
import os
import shutil

FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z"
MPV_URL = "https://sourceforge.net/projects/mpv-player-windows/files/64bit/mpv-0.35.0-x86_64.7z/download"

def download_file(url, filename):
    print(f"Downloading {filename} from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded {filename}")

def extract_7z(filename, extract_path):
    print(f"Extracting {filename}...")
    with py7zr.SevenZipFile(filename, mode='r') as z:
        z.extractall(path=extract_path)
    print(f"Extracted {filename} to {extract_path}")

def main():
    # Create a temp directory
    if not os.path.exists("temp_setup"):
        os.makedirs("temp_setup")

    # Download and extract FFmpeg
    ffmpeg_archive = os.path.join("temp_setup", "ffmpeg.7z")
    ffmpeg_extract_path = os.path.join("temp_setup", "ffmpeg")
    download_file(FFMPEG_URL, ffmpeg_archive)
    extract_7z(ffmpeg_archive, ffmpeg_extract_path)

    # Download and extract mpv
    mpv_archive = os.path.join("temp_setup", "mpv.7z")
    mpv_extract_path = os.path.join("temp_setup", "mpv")
    download_file(MPV_URL, mpv_archive)
    extract_7z(mpv_archive, mpv_extract_path)

    # Copy FFmpeg files
    ffmpeg_bin_path = os.path.join(ffmpeg_extract_path, os.listdir(ffmpeg_extract_path)[0], "bin")
    for item in os.listdir(ffmpeg_bin_path):
        shutil.copy(os.path.join(ffmpeg_bin_path, item), ".")

    # Copy mpv files
    shutil.copy(os.path.join(mpv_extract_path, "mpv-2.dll"), ".")

    # Clean up
    shutil.rmtree("temp_setup")

if __name__ == "__main__":
    main()
