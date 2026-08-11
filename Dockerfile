# syntax=docker/dockerfile:1.7

# Supply the CUDA-enabled project image that is already used by the training
# platform. TDWM deliberately does not select or install PyTorch/CUDA.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="TDWM"
LABEL org.opencontainers.image.source="https://github.com/iamwangyabin/tdwm"
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

COPY docker/constraints.txt /tmp/tdwm-constraints.txt

# Record the base image's accelerator packages. The final build check requires
# these exact versions to remain installed, so dependency resolution cannot
# silently replace the platform-provided PyTorch stack.
RUN python -c "import importlib.metadata as m, json, sys, torch, torchvision; assert sys.version_info >= (3,10),sys.version; assert torch.version.cuda is not None,'base image must provide a CUDA-enabled torch'; base={'torch':m.version('torch'),'torchvision':m.version('torchvision'),'cuda':torch.version.cuda}; json.dump(base,open('/tmp/tdwm-base-torch.json','w')); open('/tmp/tdwm-base-constraints.txt','w').write('torch=='+base['torch']+'\ntorchvision=='+base['torchvision']+'\n')"

# Install the exact upstream platform first so this layer remains cached while
# TDWM source files change. PyTorch, torchvision, and CUDA come only from the
# supplied base image and are intentionally absent from the constraints file.
RUN python -m pip install \
        "pip==24.3.1" \
        "setuptools==75.8.0" \
        "wheel==0.45.1" \
    && python -m pip install \
        --constraint /tmp/tdwm-constraints.txt \
        --constraint /tmp/tdwm-base-constraints.txt \
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
    && python -c "import importlib.metadata as m, json, sys, torch, torchvision; import stable_pretraining, stable_worldmodel as swm; from packaging.version import Version; expected={'stable-worldmodel':'0.1.1','stable-pretraining':'0.1.7','lightning':'2.4.0','transformers':'4.50.3','datasets':'2.20.0','numpy':'1.26.4','pillow':'11.3.0','pyarrow':'20.0.0','scikit-learn':'1.7.0','lancedb':'0.37.1','pylance':'10.0.0'}; actual={name:Version(m.version(name)).base_version for name in expected}; base=json.load(open('/tmp/tdwm-base-torch.json')); final={'torch':m.version('torch'),'torchvision':m.version('torchvision'),'cuda':torch.version.cuda}; assert actual==expected,(actual,expected); assert final==base,(final,base); assert sys.version_info >= (3,10),sys.version; print('TDWM container verified:',actual,'base accelerator stack preserved:',final,'SWM',swm.__file__)"

RUN mkdir -p "${STABLEWM_HOME}" "${TDWM_RUN_ROOT}"

CMD ["bash"]
