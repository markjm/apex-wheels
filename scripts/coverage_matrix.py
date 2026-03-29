"""Build matrix definitions for NVIDIA Apex wheel builds.

This is the single source of truth for which Python / PyTorch / CUDA
combinations are valid.  It also doubles as a CLI:

    # Emit the GitHub Actions matrix JSON
    python -m scripts.coverage_matrix matrix

    # Resolve the PyTorch CUDA wheel-index tag
    python -m scripts.coverage_matrix torch-cuda 128 2.9
"""

from __future__ import annotations

import json
import sys

# ── Versions to build ──────────────────────────────────────────────────

PYTHON_VERSIONS = ["3.12"]

TORCH_FULL_VERSIONS = [
    "2.8.0",
    "2.9.1",
]

LINUX_CUDA_VERSIONS = ["12.8"]

# ── Compatibility tables ───────────────────────────────────────────────

TORCH_SUPPORT_CUDA_VERSIONS: dict[str, tuple[str, ...]] = {
    "2.7": ("11.8", "12.6", "12.8"),
    "2.8": ("12.6", "12.8", "12.9"),
    "2.9": ("12.6", "12.8", "13.0"),
}

TORCH_SUPPORT_PYTHON_VERSIONS: dict[str, tuple[str, str]] = {
    "2.8": ("3.9", "3.13"),
    "2.9": ("3.10", "3.14"),
}

# ── Helpers ────────────────────────────────────────────────────────────


def _parse_py(version: str) -> tuple[int, int]:
    major, minor = version.split(".")
    return int(major), int(minor)


def _torch_minor(full: str) -> str:
    parts = full.split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else full


def _is_supported_python(torch_minor: str, python: str) -> bool:
    if torch_minor not in TORCH_SUPPORT_PYTHON_VERSIONS:
        return False
    lo, hi = TORCH_SUPPORT_PYTHON_VERSIONS[torch_minor]
    return _parse_py(lo) <= _parse_py(python) <= _parse_py(hi)


# ── Matrix generation ──────────────────────────────────────────────────


def _build_exclude() -> list[dict[str, str]]:
    """Incompatible pairs to feed to ``strategy.matrix.exclude``."""
    exclude: list[dict[str, str]] = []
    for torch_full in TORCH_FULL_VERSIONS:
        minor = _torch_minor(torch_full)
        supported_cuda = TORCH_SUPPORT_CUDA_VERSIONS.get(minor, ())
        for cuda in LINUX_CUDA_VERSIONS:
            if cuda not in supported_cuda:
                exclude.append({"torch-version": torch_full, "cuda-version": cuda})
        for py in PYTHON_VERSIONS:
            if not _is_supported_python(minor, py):
                exclude.append({"torch-version": torch_full, "python-version": py})
    return exclude


def build_matrix_json() -> str:
    """Return the full matrix JSON consumed by GitHub Actions."""
    return json.dumps(
        {
            "linux": {
                "python-version": PYTHON_VERSIONS,
                "torch-version": TORCH_FULL_VERSIONS,
                "cuda-version": LINUX_CUDA_VERSIONS,
            },
            "exclude": _build_exclude(),
        }
    )


# ── PyTorch CUDA index tag ─────────────────────────────────────────────


def torch_cuda_index_tag(cuda_version_int: int, torch_minor: str) -> int:
    """Map a CUDA version int (e.g. 128) to the closest PyTorch wheel index.

    Example: torch_cuda_index_tag(128, "2.9") → 128
    """
    if torch_minor not in TORCH_SUPPORT_CUDA_VERSIONS:
        print(
            f"error: torch minor '{torch_minor}' not in TORCH_SUPPORT_CUDA_VERSIONS "
            f"(known: {', '.join(TORCH_SUPPORT_CUDA_VERSIONS)})",
            file=sys.stderr,
        )
        sys.exit(1)

    supported = [
        int(v.replace(".", "")) for v in TORCH_SUPPORT_CUDA_VERSIONS[torch_minor]
    ]
    cuda_major = str(cuda_version_int)[:2]
    same_major = [v for v in supported if str(v)[:2] == cuda_major]
    if same_major:
        return min(same_major, key=lambda v: abs(v - cuda_version_int))
    return supported[-1]


# ── CLI ────────────────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <matrix|torch-cuda> [args...]", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "matrix":
        print(build_matrix_json())

    elif cmd == "torch-cuda":
        if len(sys.argv) != 4:
            print(
                f"usage: {sys.argv[0]} torch-cuda <cuda_int> <torch_minor>",
                file=sys.stderr,
            )
            sys.exit(1)
        cuda_int = int(sys.argv[2])
        torch_minor = sys.argv[3]
        print(torch_cuda_index_tag(cuda_int, torch_minor))

    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
