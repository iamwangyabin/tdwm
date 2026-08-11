# syntax=docker/dockerfile:1.7

# Linux/amd64 only: this is the official PyTorch CUDA 12.1 runtime image.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime@sha256:ac7c098a81512e719afa5d2d497f812d7db3498f340a4b819c69cb7b3b257126

LABEL org.opencontainers.image.title="TDWM"
LABEL org.opencontainers.image.source="https://github.com/iamwangyabin/tdwm"
LABEL tdwm.pytorch.version="2.4.1"
LABEL tdwm.cuda.version="12.1"
LABEL tdwm.stable-worldmodel.version="0.1.1"

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    MUJOCO_GL=egl \
    SDL_VIDEODRIVER=dummy \
    STABLEWM_HOME=/workspace/cache/stable_worldmodel \
    TDWM_RUN_ROOT=/workspace/runs

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libglfw3 \
        libosmesa6 \
        libsm6 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        swig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/tdwm

COPY docker/constraints-cu121.txt /tmp/tdwm-constraints.txt

# Install the exact upstream platform first so this layer remains cached while
# TDWM source files change. The constraints keep the CUDA/PyTorch stack and the
# training/data packages on the versions verified for this repository.
RUN python -m pip install \
        "pip==24.3.1" \
        "setuptools==75.8.0" \
        "wheel==0.45.1" \
    && python -m pip install \
        --constraint /tmp/tdwm-constraints.txt \
        "stable-worldmodel[all]==0.1.1" \
    && python -m pip check

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

# TDWM itself has no dependency source separate from pyproject.toml. Its only
# declared runtime dependency was installed and verified in the previous layer.
RUN python -m pip install --no-deps . \
    && python -m pip check \
    && python -c "import importlib.metadata as m, sys, torch, torchvision; import stable_pretraining, stable_worldmodel as swm; from packaging.version import Version; expected={'stable-worldmodel':'0.1.1','stable-pretraining':'0.1.8','torch':'2.4.1','torchvision':'0.19.1','lightning':'2.4.0','transformers':'4.50.3','datasets':'2.20.0','numpy':'2.0.2','pillow':'11.3.0','pyarrow':'20.0.0','scikit-learn':'1.7.0','lancedb':'0.37.1','pylance':'10.0.0'}; actual={name:Version(m.version(name)).base_version for name in expected}; assert actual==expected,(actual,expected); assert sys.version_info[:2]==(3,11),sys.version; assert torch.version.cuda=='12.1',torch.version.cuda; print('TDWM container verified:', actual, 'CUDA', torch.version.cuda, 'SWM', swm.__file__)"

RUN mkdir -p "${STABLEWM_HOME}" "${TDWM_RUN_ROOT}"

CMD ["bash"]
