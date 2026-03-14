"""Web layer request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel


class JobState(BaseModel):
    job_id: str
    original_filename: str
    mode: str
    flow: str
    upload_path: str | None = None
    output_path: str | None = None
    preview_original: str | None = None
    preview_processed: str | None = None
    report: dict | None = None
    metadata: dict | None = None
    export_target: str | None = None
    print_scale: int | None = None
    status: str = "uploaded"
    error: str | None = None
    created_at: str = ""
