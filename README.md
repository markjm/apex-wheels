# apex-wheels

Pre-built wheels for [NVIDIA Apex](https://github.com/NVIDIA/apex) with all
CUDA extensions enabled.

Building Apex from source takes a long time and requires a CUDA toolkit.  This
repository automates the process with GitHub Actions and publishes the resulting
wheels as GitHub Release assets.

## Wheel naming

Wheels follow the local-version convention used by
[flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels):

```
apex-[YY.MM]+cu[CUDA]torch[PyTorch]-cp[Python]-cp[Python]-linux_x86_64.whl
```

The base version comes from Apex's `YY.MM` git tags (matching NVIDIA NGC
container releases).  Builds from commits between tags get a `.devN` suffix.

| Apex checkout        | Wheel version example                |
| -------------------- | ------------------------------------ |
| On tag `25.09`       | `apex-25.09+cu128torch2.9-…`        |
| 3 commits past 25.09 | `apex-25.09.dev3+cu128torch2.9-…`  |

## Install

Find the wheel matching your environment from the
[Releases](../../releases) page, then install directly:

```bash
pip install https://github.com/markjm/apex-wheels/releases/download/v25.09/apex-25.09+cu128torch2.9-cp312-cp312-linux_x86_64.whl
```

Or download first:

```bash
wget <wheel-url>
pip install ./apex-25.09+cu128torch2.9-cp312-cp312-linux_x86_64.whl
```

## Extensions included

The wheels are built with **all** extensions enabled:

| Environment variable     | Extension                          |
| ------------------------ | ---------------------------------- |
| `APEX_CPP_EXT`           | Core C++ extension                 |
| `APEX_CUDA_EXT`          | Core CUDA extensions (amp, syncbn, fused layer norm, …) |
| `APEX_ALL_CONTRIB_EXT`   | Every contrib extension (xentropy, fmha, fast layer norm, distributed adam/lamb, …) |

Extensions that cannot be built for a particular CUDA / cuDNN / NCCL
combination are silently skipped by the upstream `setup.py`.

## Building locally

```bash
./build_linux.sh <python-version> <torch-version> <cuda-version> [apex-commit]
# Example:
./build_linux.sh 3.12 2.9.1 12.8 25.09
```

To override the target GPU architectures (default: `8.0 8.6 9.0 10.0 12.0+PTX`):

```bash
TORCH_CUDA_ARCH_LIST="8.9" ./build_linux.sh 3.12 2.9.1 12.8 25.09
```

## Triggering a release

Push a `v`-prefixed tag that matches an Apex tag to build and release wheels.
The `v` prefix is stripped to derive the Apex git ref (e.g. `v25.09` → Apex
tag `25.09`):

```bash
git tag v25.09
git push origin v25.09
```

Or use **workflow_dispatch** from the Actions tab to build an arbitrary Apex
commit on demand.

## Customizing the matrix

Edit `scripts/coverage_matrix.py` to change which Python / PyTorch / CUDA
combinations are built.
