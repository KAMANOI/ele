"""ele CLI entry point.

Usage:
  ele input.jpg --mode creator --output out.tiff
  ele input.png --mode pro --output out.tiff
  ele input.jpg --mode print --scale 4 --output out_print.tiff
  ele input.jpg --mode print --scale 2 --print-style natural --output out_natural.tiff
  ele input.jpg --mode print --scale 4 --print-style ai-detail --output out_aidetail.tiff
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ele.config import APP_NAME, APP_VERSION

app = typer.Typer(
    name=APP_NAME,
    help="Pseudo-RAW preprocessing engine for Lightroom / Photoshop / Capture One.",
    add_completion=False,
)
console = Console(stderr=False)
err     = Console(stderr=True)


class ModeChoice(str, Enum):
    free    = "free"
    creator = "creator"
    pro     = "pro"
    print_  = "print"


class PrintStyleChoice(str, Enum):
    natural   = "natural"
    ai_detail = "ai-detail"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"{APP_NAME} {APP_VERSION}")
        raise typer.Exit()


@app.command()
def process(
    input_path: Path = typer.Argument(
        ...,
        help="Input image file (JPEG, PNG, TIFF)",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(
        ..., "--output", "-o",
        help="Output file path (.tiff recommended)",
    ),
    mode: ModeChoice = typer.Option(
        ModeChoice.creator, "--mode", "-m",
        help="Processing mode: free | creator | pro | print",
    ),
    scale: Optional[int] = typer.Option(
        None, "--scale", "-s",
        help="Super-resolution scale for print mode (2, 4, or 6)",
    ),
    print_style: PrintStyleChoice = typer.Option(
        PrintStyleChoice.natural, "--print-style",
        help=(
            "Upscale style for print mode: "
            "natural (photography-first, low-artifact) | "
            "ai-detail (stronger perceived detail, may look synthetic on photos)"
        ),
    ),
    report: bool = typer.Option(
        False, "--report", "-r",
        help="Print degradation report and pipeline metadata after processing",
    ),
    debug_srgb_export: bool = typer.Option(
        False, "--debug-srgb-export",
        help=(
            "Also write an sRGB-gamma-encoded TIFF alongside the normal output "
            "(suffix _srgb_debug.tiff). For diagnosis only — compare with the "
            "ProPhoto TIFF to isolate colour-management issues in your editor."
        ),
    ),
    version: Optional[bool] = typer.Option(
        None, "--version", "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """Convert an image to a pseudo-RAW editable master file."""

    total_steps = 6
    step_labels = [
        "loading image",
        "analyzing degradation",
        "faithful restoration",
        "scene reconstruction",
        "pseudo-RAW reconstruction",
        "exporting",
    ]
    current_step: list[int] = [0]

    def progress_cb(step: int, label: str) -> None:
        current_step[0] = step
        console.print(f"  [{step}/{total_steps}] {label}", highlight=False)

    console.print(f"\n[bold]{APP_NAME}[/bold]  [dim]{input_path}[/dim]  →  [cyan]{output}[/cyan]  [dim](mode={mode.value})[/dim]")

    runner = _get_runner(mode)
    try:
        kwargs: dict = {"scale": scale, "_progress_cb": progress_cb}
        if mode == ModeChoice.print_:
            kwargs["print_style"] = print_style.value
        result = runner(str(input_path), str(output), **kwargs)
    except NotImplementedError as exc:
        err.print(f"\n[yellow]Not implemented:[/yellow] {exc}")
        raise typer.Exit(code=3)
    except Exception as exc:
        err.print(f"\n[red]Error:[/red] {exc}")
        raise typer.Exit(code=2)

    h, w = result.image.shape[:2]
    console.print(
        f"\n[green]✓[/green] Saved: [bold]{result.output_path}[/bold]"
        f"  [dim]{w}×{h} px[/dim]"
    )

    if debug_srgb_export:
        from ele.export.tiff_export import export_tiff_srgb_debug
        try:
            debug_path = export_tiff_srgb_debug(result.image, str(output))
            console.print(
                f"[dim]  debug:[/dim] [yellow]{debug_path}[/yellow]  "
                "[dim](sRGB gamma, no ProPhoto ICC — diagnosis only)[/dim]"
            )
        except Exception as exc:
            err.print(f"[yellow]debug-srgb-export failed:[/yellow] {exc}")

    if report:
        _print_report(result)


def _get_runner(mode: ModeChoice):
    if mode == ModeChoice.free:
        from ele.pipeline.free_pipeline import run_free_pipeline
        return run_free_pipeline
    elif mode == ModeChoice.creator:
        from ele.pipeline.creator_pipeline import run_creator_pipeline
        return run_creator_pipeline
    elif mode == ModeChoice.pro:
        from ele.pipeline.pro_pipeline import run_pro_pipeline
        return run_pro_pipeline
    elif mode == ModeChoice.print_:
        from ele.pipeline.print_pipeline import run_print_pipeline
        return run_print_pipeline
    raise ValueError(f"Unknown mode: {mode}")


def _print_report(result) -> None:
    """Print degradation report and metadata as a rich table."""
    console.print("\n[bold]Degradation Report[/bold]")
    console.print(result.report.summary())

    if result.metadata:
        console.print("\n[bold]Pipeline Metadata[/bold]")
        table = Table(show_header=True, header_style="bold dim")
        table.add_column("Key",   style="dim", no_wrap=True)
        table.add_column("Value")
        for k, v in result.metadata.items():
            table.add_row(str(k), str(v))
        console.print(table)


if __name__ == "__main__":
    app()
