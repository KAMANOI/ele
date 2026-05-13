"""Replicate Clarity Upscaler wrapper."""
from __future__ import annotations

import logging
import os
import urllib.request
from urllib.parse import urlparse

log = logging.getLogger(__name__)

_MODEL = (
    "philz1337x/clarity-upscaler"
    ":dfad41707589d68ecdccd1dfa600d55a208f9310748e44bfe35b4a6291453d5e"
)
# Replicate output URLs are served from these domains
_ALLOWED_DOMAINS = {"replicate.delivery", "pbxt.replicate.delivery"}


def _validate_replicate_url(url: str) -> None:
    """Raise ValueError (E210) if the URL is not a trusted Replicate domain."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("E210: Replicate returned non-HTTPS URL")
    hostname = parsed.hostname or ""
    if not any(hostname == d or hostname.endswith("." + d) for d in _ALLOWED_DOMAINS):
        raise ValueError(f"E210: Replicate returned untrusted domain: {hostname!r}")


def run_clarity_upscale(input_path: str, output_path: str, scale_factor: int) -> str:
    """Call Clarity Upscaler via Replicate and save result to output_path.

    Requires REPLICATE_API_TOKEN in the environment.
    Raises ValueError with error code E210 on API failure.
    """
    import replicate  # imported lazily so the package is only required when used

    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise ValueError("E210: REPLICATE_API_TOKEN is not set")

    log.info(
        "clarity_upscale: input=%s  scale_factor=%d  output=%s",
        input_path, scale_factor, output_path,
    )

    try:
        with open(input_path, "rb") as f:
            result = replicate.run(
                _MODEL,
                input={"image": f, "scale_factor": scale_factor},
            )
    except Exception as exc:
        raise ValueError(f"E210: Replicate API call failed: {exc}") from exc

    # replicate >= 0.25 returns FileOutput; older versions return a list or str
    if isinstance(result, list):
        result = result[0]
    url = str(result)

    _validate_replicate_url(url)
    urllib.request.urlretrieve(url, output_path)
    log.info("clarity_upscale: saved  output=%s", output_path)
    return output_path
