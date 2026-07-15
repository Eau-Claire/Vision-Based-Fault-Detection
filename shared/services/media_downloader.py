"""
Media downloader with safety checks.

Downloads files from the backend fileUrl with configurable timeout and
maximum file size limits. Supports both Image and Video media types.
"""

import os
import tempfile
import time
from typing import Tuple

import requests

from shared.utils.logging import get_logger
from shared.utils.security import is_safe_url

logger = get_logger("media_downloader")


class DownloadError(Exception):
    """Raised when media download fails."""
    pass


class FileTooLargeError(DownloadError):
    """Raised when file exceeds maximum allowed size."""
    pass


def download_media(
    file_url: str,
    base_url: str = "",
    timeout: int = 60,
    max_size_bytes: int = 500 * 1024 * 1024,
    dest_dir: str = "",
    allow_private_ips: bool = True,
) -> Tuple[bytes, str]:
    """Download media file from URL with safety validation.

    Args:
        file_url: URL or relative path to the media file.
        base_url: Base URL to prepend if file_url is relative.
        timeout: HTTP request timeout in seconds.
        max_size_bytes: Maximum allowed file size in bytes.
        dest_dir: Optional directory to save the file. If empty, returns bytes only.
        allow_private_ips: Whether private/loopback IP addresses are permitted.

    Returns:
        Tuple of (file_bytes, file_extension).

    Raises:
        DownloadError: If download fails.
        FileTooLargeError: If file exceeds max_size_bytes.
    """
    # Resolve relative URL
    if not file_url.startswith("http"):
        if not base_url:
            raise DownloadError(
                f"Cannot download relative URL without base_url: {file_url}"
            )
        full_url = f"{base_url.rstrip('/')}/{file_url.lstrip('/')}"
    else:
        full_url = file_url

    # SSRF & safety validation
    if not is_safe_url(full_url, allow_private_ips=allow_private_ips):
        raise DownloadError(f"URL is unsafe or forbidden: {full_url}")

    logger.info(
        f"Downloading media from: {full_url}",
        extra={"event": "media_download_start"},
    )

    start = time.monotonic()

    try:
        # Stream download to check size progressively
        response = requests.get(full_url, timeout=timeout, stream=True)
        response.raise_for_status()

        # Check Content-Length header if available
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_size_bytes:
            raise FileTooLargeError(
                f"File size {int(content_length)} bytes exceeds "
                f"maximum allowed {max_size_bytes} bytes"
            )

        # Download in chunks with size limit
        chunks = []
        total_bytes = 0
        for chunk in response.iter_content(chunk_size=8192):
            total_bytes += len(chunk)
            if total_bytes > max_size_bytes:
                raise FileTooLargeError(
                    f"Download exceeded maximum size of {max_size_bytes} bytes"
                )
            chunks.append(chunk)

        file_bytes = b"".join(chunks)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Determine file extension from URL or content type
        ext = _guess_extension(full_url, response.headers.get("Content-Type", ""))

        logger.info(
            f"Downloaded {len(file_bytes)} bytes in {elapsed_ms}ms",
            extra={
                "event": "media_download_complete",
                "duration_ms": elapsed_ms,
            },
        )

        return file_bytes, ext

    except FileTooLargeError:
        raise
    except requests.exceptions.Timeout:
        raise DownloadError(
            f"Download timed out after {timeout}s: {full_url}"
        )
    except requests.exceptions.RequestException as e:
        raise DownloadError(f"Failed to download media: {e}")


def save_to_temp_file(
    file_bytes: bytes, extension: str = ".mp4", dest_dir: str = ""
) -> str:
    """Save bytes to a temporary file and return the path.

    Args:
        file_bytes: Raw file content.
        extension: File extension (e.g. '.mp4', '.jpg').
        dest_dir: Directory for temp file. Uses system temp if empty.

    Returns:
        Absolute path to the temporary file.
    """
    kwargs = {"suffix": extension, "delete": False}
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
        kwargs["dir"] = dest_dir

    with tempfile.NamedTemporaryFile(**kwargs) as tmp:
        tmp.write(file_bytes)
        return tmp.name


def _guess_extension(url: str, content_type: str) -> str:
    """Guess file extension from URL path or Content-Type header."""
    # Try URL path first
    url_lower = url.lower().split("?")[0]
    for ext in (".mp4", ".avi", ".mov", ".webm", ".mkv",
                ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"):
        if url_lower.endswith(ext):
            return ext

    # Fallback to Content-Type
    ct = content_type.lower()
    if "video" in ct:
        return ".mp4"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "png" in ct:
        return ".png"

    return ".bin"
