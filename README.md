# apex-wheels

Pre-built wheels for [NVIDIA Apex](https://github.com/NVIDIA/apex) with some supported extensions enabled.


Per the upstream README, the expectation is that you build this project from source to work with the specific torch/cuda/python/extensions you want. While that definitely is the ideal approach for an optimized build, it can be a pain to set up. This repository is focused on providing wheels for a common case using common stuff in a selection of CUDA/PyTorch/Python versions. This makes it easier to try things out across a range of environments without needing to build from source each time.

The project structure and code is heavily inspired by [flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels), so thanks to them for the inspiration!


## Install

Find the wheel matching your environment from the
[Releases](https://github.com/markjm/apex-wheels/releases) page, then install directly:

```bash
pip install https://github.com/markjm/apex-wheels/releases/download/c6374ac/apex-0.1+g6374ac.cu128torch2.9-cp312-cp312-linux_x86_64.whl
```

Waiting for the evolution of https://peps.python.org/pep-0817/ 🙏

## Extensions included

The wheels are built with the following extensions enabled:

| Environment variable     | Extension                          |
| ------------------------ | ---------------------------------- |
| `APEX_CPP_EXT`           | Core C++ extension                 |
| `APEX_CUDA_EXT`          | Core CUDA extensions (amp, syncbn, fused layer norm, …) |
| `APEX_FAST_MULTIHEAD_ATTN` | Fast multihead attention |

I am open to adding additional extensions if there is high demand. This list is just what I personally care about and know builds successfully.

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

I manually trigger a release when I want to build a new wheel. I use the following command to trigger a release. May look to automate if folks are interested in this.
