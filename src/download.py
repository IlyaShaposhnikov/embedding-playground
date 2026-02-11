"""
Utilities for downloading and verifying pre-trained embedding models.
All functions include error handling and detailed console feedback.
"""

import os
import gdown
import gzip
import shutil
from typing import Optional

# Expected file sizes for GoogleNews vectors (in bytes)
# Source: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM
GOOGLENEWS_SIZE = 3_644_258_522      # ~3.39 GB (uncompressed .bin)
GOOGLENEWS_GZ_SIZE = 1_647_046_227   # ~1.53 GB (compressed .gz)


def verify_file_size(file_path: str, expected_size: int) -> bool:
    """Check if the file size matches the expected value."""
    if not os.path.exists(file_path):
        return False
    actual_size = os.path.getsize(file_path)
    return actual_size == expected_size


def extract_gzip(gz_path: str, bin_path: str) -> bool:
    """Extract .gz file to uncompressed binary format."""
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(bin_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except (gzip.BadGzipFile, OSError) as e:
        print(f"Extraction failed: {e}")
        return False


def download_word2vec_model(
    force_download: bool = False,
    data_dir: str = "data"
) -> Optional[str]:
    """
    Download GoogleNews pre-trained Word2Vec model (vectors-negative300).
    Implements smart caching: uses existing .bin or .gz files if valid.
    """
    os.makedirs(data_dir, exist_ok=True)

    gz_path = os.path.join(data_dir, "GoogleNews-vectors-negative300.bin.gz")
    bin_path = os.path.join(data_dir, "GoogleNews-vectors-negative300.bin")

    # Clean cache when force_download is requested
    # force_download: If True, ignores existing files and re-downloads
    if force_download:
        for path, name in [(bin_path, "binary"), (gz_path, "compressed")]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed cached {name} file: {path}")

    # Check if valid uncompressed binary already exists
    if os.path.exists(bin_path):
        if verify_file_size(bin_path, GOOGLENEWS_SIZE):
            print(f"Word2Vec binary already exists: {bin_path}")
            return bin_path
        else:
            print(
                "Existing binary has incorrect size. "
                f"Expected {GOOGLENEWS_SIZE}, "
                f"got {os.path.getsize(bin_path)}. Re-downloading..."
            )

    # Check if valid compressed archive exists and extract it
    if os.path.exists(gz_path):
        if verify_file_size(gz_path, GOOGLENEWS_GZ_SIZE):
            print(f"Found valid compressed file, extracting: {gz_path}")
            if extract_gzip(gz_path, bin_path):
                if verify_file_size(bin_path, GOOGLENEWS_SIZE):
                    print(f"Extraction complete: {bin_path}")
                    return bin_path
                else:
                    print(
                        "Extracted file has incorrect size. Re-downloading..."
                    )
            # Fall through to download on extraction failure or size mismatch
        else:
            print(
                "Compressed file has incorrect size. "
                f"Expected {GOOGLENEWS_GZ_SIZE}, "
                f"got {os.path.getsize(gz_path)}. Re-downloading..."
            )

    # Verify sufficient disk space before download (~4.5 GB recommended)
    # +0.5 GB buffer
    required_space = GOOGLENEWS_SIZE + GOOGLENEWS_GZ_SIZE + 500_000_000
    free_space = shutil.disk_usage(data_dir).free
    if free_space < required_space:
        print(f"Insufficient disk space. Need ~{required_space / 1024**3:.1f} "
              f"GB, have {free_space / 1024**3:.1f} GB available.")
        return None

    # Download from Google Drive
    print(
        "Downloading Word2Vec model (GoogleNews-vectors-negative300.bin.gz)..."
    )
    print(
        "This file is ~1.5 GB compressed, ~3.4 GB uncompressed. "
        "May take several minutes."
    )

    url = "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"
    try:
        gdown.download(url, gz_path, quiet=False)
    except Exception as e:
        print(f"Download failed: {e}")
        return None

    # Verify downloaded .gz file size before extraction
    if not os.path.exists(gz_path):
        print("Downloaded file not found.")
        return None

    gz_size = os.path.getsize(gz_path)
    print(f"Downloaded compressed file size: {gz_size / 1024**3:.2f} GB")
    if not verify_file_size(gz_path, GOOGLENEWS_GZ_SIZE):
        print(
            f"Size mismatch: expected {GOOGLENEWS_GZ_SIZE} "
            f"bytes, got {gz_size} bytes."
        )
        return None

    # Extract the archive
    print("Extracting compressed file...")
    if not extract_gzip(gz_path, bin_path):
        return None

    print(f"Extraction complete: {bin_path}")

    # Final verification of uncompressed file
    if verify_file_size(bin_path, GOOGLENEWS_SIZE):
        print("File size matches expected value. Download successful.")
        return bin_path
    else:
        actual = os.path.getsize(bin_path)
        print(
            "Size mismatch after extraction: "
            f"expected {GOOGLENEWS_SIZE}, got {actual}"
        )
        return None
