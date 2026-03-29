#!/usr/bin/env python3
"""Install a specific CUDA toolkit version on Linux (Ubuntu/Debian).

Replaces the mjun0812/setup-cuda GitHub Action with a pure-Python script
that lives in this repository.  Adds NVIDIA's apt repo and installs the
cuda-toolkit package (Debian/Ubuntu).

Usage:
    python scripts/setup_cuda.py <version>

    <version> is Major.Minor (e.g. "12.8") or Major.Minor.Patch ("12.8.0").

After a successful install the script prints shell exports you can eval,
or - when running inside GitHub Actions - it sets outputs and env vars
directly via GITHUB_OUTPUT / GITHUB_ENV / GITHUB_PATH.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
import shutil
import tempfile
import urllib.request


# ---------------------------------------------------------------------------
# Version resolution
# ---------------------------------------------------------------------------


def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode()


def fetch_available_versions() -> list[str]:
    """Scrape NVIDIA's redistrib directory for available CUDA versions."""
    sources = [
        (
            "https://developer.download.nvidia.com/compute/cuda/redist/",
            re.compile(r"redistrib_([0-9]+\.[0-9]+(?:\.[0-9]+)?)\.json"),
        ),
        (
            "https://developer.download.nvidia.com/compute/cuda/opensource/",
            re.compile(r">([0-9]+\.[0-9]+(?:\.[0-9]+)?)/"),
        ),
    ]
    versions: set[str] = set()
    for url, pattern in sources:
        try:
            html = _fetch_text(url)
            versions.update(pattern.findall(html))
        except Exception as exc:
            print(
                f"[setup-cuda] warning: failed to fetch {url}: {exc}", file=sys.stderr
            )
    return sorted(versions, key=_version_key)


def _version_key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def resolve_version(requested: str) -> str:
    """Resolve a Major.Minor (or full) version to the best available match."""
    available = fetch_available_versions()
    if not available:
        raise RuntimeError("Could not fetch any CUDA version list from NVIDIA")

    if requested in available:
        return requested

    prefix = requested + "."
    matches = [v for v in available if v.startswith(prefix)]
    if matches:
        return matches[-1]

    raise RuntimeError(
        f"CUDA version '{requested}' not found. Available: {', '.join(available[-15:])}"
    )


# ---------------------------------------------------------------------------
# Linux distro detection
# ---------------------------------------------------------------------------


def _parse_os_release() -> dict[str, str]:
    for path in ("/etc/os-release", "/usr/lib/os-release"):
        if os.path.isfile(path):
            info: dict[str, str] = {}
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line:
                        k, v = line.split("=", 1)
                        info[k] = v.strip('"')
            return info
    raise RuntimeError("Cannot detect Linux distribution (no os-release)")


def _is_debian_based(os_info: dict[str, str]) -> bool:
    return os_info.get("ID", "") == "debian" or "debian" in os_info.get("ID_LIKE", "")


def _target_os_name(os_info: dict[str, str]) -> str:
    distro_id = os_info.get("ID", "").lower()
    version = os_info.get("VERSION_ID", "")
    if distro_id == "ubuntu":
        return f"{distro_id}{version.replace('.', '')}"
    return f"{distro_id}{version.split('.')[0]}"


# ---------------------------------------------------------------------------
# Network install (Debian/Ubuntu apt)
# ---------------------------------------------------------------------------


def _run(cmd: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"  $ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, text=True)


def _sudo(cmd: str, **kw) -> subprocess.CompletedProcess[str]:  # type: ignore[type-arg]
    prefix = "" if os.geteuid() == 0 else "sudo "
    return _run(f"{prefix}{cmd}", **kw)


def install_network(version: str) -> str:
    """Install CUDA via NVIDIA's apt repo (Debian/Ubuntu)."""
    os_info = _parse_os_release()
    if not _is_debian_based(os_info):
        raise RuntimeError(
            f"Network install only supports Debian-based distros, got {os_info.get('ID')}"
        )

    arch = platform.machine()
    if arch == "aarch64":
        repo_arch = "sbsa"
    elif arch in ("x86_64", "amd64"):
        repo_arch = "x86_64"
    else:
        raise RuntimeError(f"Unsupported architecture: {arch}")

    target_os = _target_os_name(os_info)
    repo_base = f"https://developer.download.nvidia.com/compute/cuda/repos/{target_os}/{repo_arch}"

    print(f"[setup-cuda] Fetching repo file listing from {repo_base}/ ...")
    listing = _fetch_text(f"{repo_base}/")

    keyring_debs = sorted(re.findall(r"(cuda-keyring[\w.\-]+\.deb)", listing))
    if not keyring_debs:
        raise RuntimeError(f"No cuda-keyring .deb found at {repo_base}/")
    keyring_deb = keyring_debs[-1]

    with tempfile.TemporaryDirectory() as tmpdir:
        local_deb = os.path.join(tmpdir, keyring_deb)
        print(f"[setup-cuda] Downloading {keyring_deb} ...")
        urllib.request.urlretrieve(f"{repo_base}/{keyring_deb}", local_deb)
        _sudo(f"dpkg -i {local_deb}")

    _sudo("apt-get update -qq")

    major_minor = ".".join(version.split(".")[:2])
    mm_dash = major_minor.replace(".", "-")
    _sudo(f"apt-get install -y cuda-toolkit-{mm_dash} libnccl-dev")

    cuda_path = "/usr/local/cuda"
    if not os.path.isdir(cuda_path):
        raise RuntimeError(f"CUDA install succeeded but {cuda_path} not found")
    return cuda_path


# ---------------------------------------------------------------------------
# Local (.run) install
# ---------------------------------------------------------------------------


def _find_run_installer_url(version: str) -> str:
    """Find the .run installer URL by checking NVIDIA's md5sum manifest."""
    md5_url = f"https://developer.download.nvidia.com/compute/cuda/{version}/docs/sidebar/md5sum.txt"
    text = _fetch_text(md5_url)

    arch = platform.machine()
    if arch in ("x86_64", "amd64"):
        suffix = "_linux.run"
    elif arch == "aarch64":
        suffix = "_linux_sbsa.run"
    else:
        raise RuntimeError(f"Unsupported architecture: {arch}")

    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            fname = parts[-1]
            if fname.endswith(suffix):
                return f"https://developer.download.nvidia.com/compute/cuda/{version}/local_installers/{fname}"

    raise RuntimeError(f"Could not find .run installer for CUDA {version} ({suffix})")


def install_local(version: str) -> str:
    """Download and run the CUDA .run installer."""
    url = _find_run_installer_url(version)
    print(f"[setup-cuda] Downloading {url} ...")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_run = os.path.join(tmpdir, f"cuda_{version}_linux.run")
        urllib.request.urlretrieve(url, local_run)
        os.chmod(local_run, 0o755)
        _sudo(f"sh {local_run} --silent --override --toolkit")

    cuda_path = "/usr/local/cuda"
    if not os.path.isdir(cuda_path):
        raise RuntimeError(f"CUDA install succeeded but {cuda_path} not found")
    return cuda_path


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------


def _gh_actions_env(name: str, value: str) -> None:
    """Append to GITHUB_ENV if running inside Actions."""
    env_file = os.environ.get("GITHUB_ENV")
    if env_file:
        with open(env_file, "a") as f:
            f.write(f"{name}={value}\n")


def _gh_actions_path(entry: str) -> None:
    """Append to GITHUB_PATH if running inside Actions."""
    path_file = os.environ.get("GITHUB_PATH")
    if path_file:
        with open(path_file, "a") as f:
            f.write(f"{entry}\n")


def _gh_actions_output(name: str, value: str) -> None:
    """Append to GITHUB_OUTPUT if running inside Actions."""
    out_file = os.environ.get("GITHUB_OUTPUT")
    if out_file:
        with open(out_file, "a") as f:
            f.write(f"{name}={value}\n")


def set_env(cuda_path: str, version: str) -> None:
    bin_dir = os.path.join(cuda_path, "bin")
    lib_dir = os.path.join(cuda_path, "lib64")

    os.environ["CUDA_HOME"] = cuda_path
    os.environ["CUDA_PATH"] = cuda_path
    os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    _gh_actions_env("CUDA_HOME", cuda_path)
    _gh_actions_env("CUDA_PATH", cuda_path)
    _gh_actions_env("LD_LIBRARY_PATH", os.environ["LD_LIBRARY_PATH"])
    _gh_actions_path(bin_dir)
    _gh_actions_output("version", version)
    _gh_actions_output("cuda-path", cuda_path)

    print(f"[setup-cuda] CUDA_HOME={cuda_path}")
    print(f"[setup-cuda] PATH prepended with {bin_dir}")
    print(f"[setup-cuda] LD_LIBRARY_PATH prepended with {lib_dir}")

    # Also emit eval-able exports for non-Actions callers
    print()
    print("# Paste the following into your shell if needed:")
    print(f'export CUDA_HOME="{cuda_path}"')
    print(f'export CUDA_PATH="{cuda_path}"')
    print(f'export PATH="{bin_dir}:$PATH"')
    print(f'export LD_LIBRARY_PATH="{lib_dir}:$LD_LIBRARY_PATH"')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Install NVIDIA CUDA toolkit")
    parser.add_argument("version", help="CUDA version, e.g. 12.8 or 12.8.0")
    parser.add_argument(
        "--method",
        choices=("auto", "network", "local"),
        default="auto",
        help="Install method (default: auto — tries network, falls back to local)",
    )
    args = parser.parse_args()

    print(f"[setup-cuda] Requested CUDA {args.version}, method={args.method}")
    version = resolve_version(args.version)
    print(f"[setup-cuda] Resolved to CUDA {version}")

    cuda_path: str | None = None

    if args.method == "network":
        cuda_path = install_network(version)
    elif args.method == "local":
        cuda_path = install_local(version)
    else:
        try:
            cuda_path = install_network(version)
        except Exception as exc:
            print(f"[setup-cuda] Network install failed: {exc}", file=sys.stderr)
            print(
                "[setup-cuda] Falling back to local .run installer ...", file=sys.stderr
            )
            cuda_path = install_local(version)

    set_env(cuda_path, version)

    nvcc = shutil.which("nvcc", path=os.environ["PATH"])
    if nvcc:
        _run(f"{nvcc} --version", check=False)

    print(f"\n[setup-cuda] Done — CUDA {version} installed at {cuda_path}")


if __name__ == "__main__":
    main()
