"""DOCX → PDF conversion for the publisher.

Two backends, auto-selected:

  1. docx2pdf (Windows, Word-backed): perfect fidelity. Used for local runs.
  2. libreoffice --headless --convert-to pdf: near-perfect fidelity. Used in
     CI where Word is not installed. Install once: `apt-get install libreoffice`.

Both accept DOCX bytes and return PDF bytes — same signature, no caller-side
branching required.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _convert_via_docx2pdf(docx_bytes: bytes, workdir: Path) -> bytes:
    from docx2pdf import convert  # local import — Windows-only dependency

    in_path = workdir / "resume.docx"
    out_path = workdir / "resume.pdf"
    in_path.write_bytes(docx_bytes)

    convert(str(in_path), str(out_path))

    if not out_path.exists():
        raise RuntimeError(f"docx2pdf produced no output at {out_path}")
    return out_path.read_bytes()


def _convert_via_libreoffice(docx_bytes: bytes, workdir: Path) -> bytes:
    in_path = workdir / "resume.docx"
    in_path.write_bytes(docx_bytes)

    result = subprocess.run(
        [
            "libreoffice", "--headless",
            "--convert-to", "pdf",
            "--outdir", str(workdir),
            str(in_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"libreoffice conversion failed (exit {result.returncode}): "
            f"stdout={result.stdout[:500]} stderr={result.stderr[:500]}"
        )

    out_path = workdir / "resume.pdf"
    if not out_path.exists():
        raise RuntimeError(f"libreoffice ran but no PDF at {out_path}")
    return out_path.read_bytes()


def _pick_backend() -> str:
    """Return 'docx2pdf' on Windows when Word is likely present, else 'libreoffice'."""
    if os.name == "nt":
        try:
            import docx2pdf  # noqa: F401
            return "docx2pdf"
        except ImportError:
            pass
    if shutil.which("libreoffice") or shutil.which("soffice"):
        return "libreoffice"
    # On Windows without docx2pdf, surface a clear error rather than falling through.
    raise RuntimeError(
        "No PDF backend available. Install docx2pdf (Windows + Word) or libreoffice (any OS)."
    )


def convert_docx_to_pdf(docx_bytes: bytes) -> bytes:
    """Convert DOCX bytes to PDF bytes using whichever backend is available."""
    backend = _pick_backend()
    logger.info(f"Converting DOCX → PDF via {backend}")

    with tempfile.TemporaryDirectory(prefix="resume-pdf-") as tmp:
        workdir = Path(tmp)
        if backend == "docx2pdf":
            return _convert_via_docx2pdf(docx_bytes, workdir)
        return _convert_via_libreoffice(docx_bytes, workdir)
