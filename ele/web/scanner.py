"""ClamAV malware scanner for uploaded files.

Wraps the `clamscan` CLI. Falls back gracefully if ClamAV is not installed
(e.g. local development without ClamAV). In production (Dockerfile build),
ClamAV is always present and signatures are updated on every container start.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_CLAMSCAN_TIMEOUT = 60  # seconds; large TIFFs can take a moment


def scan_file(path: str | Path) -> tuple[bool, str]:
    """Scan *path* with ClamAV clamscan.

    Returns:
        (is_clean, reason)
        - (True,  "clean")               — no threat found
        - (True,  "scanner_unavailable") — ClamAV not installed; scan skipped
        - (False, "malware: <detail>")   — threat detected; file must be rejected
    """
    if not shutil.which("clamscan"):
        log.warning("clamscan not found — skipping AV scan (install ClamAV for production)")
        return True, "scanner_unavailable"

    try:
        result = subprocess.run(
            ["clamscan", "--no-summary", "--infected", str(path)],
            capture_output=True,
            text=True,
            timeout=_CLAMSCAN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        log.error("ClamAV scan timed out for %s", path)
        return True, "scan_timeout"
    except Exception as exc:
        log.error("ClamAV scan error for %s: %s", path, exc)
        return True, f"scan_error: {exc}"

    if result.returncode == 0:
        log.debug("AV scan clean: %s", path)
        return True, "clean"

    if result.returncode == 1:
        detail = result.stdout.strip() or "unknown threat"
        log.warning("MALWARE DETECTED in %s: %s", path, detail)
        return False, f"malware: {detail}"

    # returncode == 2 = scan error (e.g. permission denied, corrupt file)
    log.error("ClamAV returned error code %d for %s: %s", result.returncode, path, result.stderr.strip())
    return True, f"scan_error: exit {result.returncode}"
