FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PBSMT_ROOT=/opt/pbsmt

WORKDIR /opt/pbsmt

RUN apt-get update && apt-get install -y --no-install-recommends \
    automake \
    build-essential \
    ca-certificates \
    cmake \
    csh \
    curl \
    gawk \
    git \
    libboost-all-dev \
    libbz2-dev \
    liblzma-dev \
    libtool \
    perl \
    pkg-config \
    python3 \
    python3-pip \
    unzip \
    wget \
    xz-utils \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir sacrebleu

RUN mkdir -p tools

RUN git clone --depth 1 https://github.com/moses-smt/mosesdecoder.git tools/mosesdecoder && \
    cd tools/mosesdecoder && \
    git tag -f local-build && \
    ./bjam -j"$(nproc)" --without-tcmalloc --no-xmlrpc-c

RUN git clone --depth 1 https://github.com/kpu/kenlm.git tools/kenlm && \
    mkdir -p tools/kenlm/build && \
    cd tools/kenlm/build && \
    cmake -Wno-dev .. >/dev/null && \
    make -j"$(nproc)" >/dev/null

RUN git clone --depth 1 https://github.com/clab/fast_align.git tools/fast_align && \
    mkdir -p tools/fast_align/build && \
    cd tools/fast_align/build && \
    cmake -Wno-dev .. >/dev/null && \
    make -j"$(nproc)" >/dev/null

RUN git clone --depth 1 https://github.com/moses-smt/giza-pp.git tools/giza-pp && \
    make -C tools/giza-pp/GIZA++-v2 -j"$(nproc)" && \
    make -C tools/giza-pp/mkcls-v2 -j"$(nproc)"

RUN mkdir -p tools/bin && \
    cp tools/giza-pp/mkcls-v2/mkcls tools/bin/ && \
    cp tools/giza-pp/GIZA++-v2/GIZA++ tools/bin/ && \
    cp tools/giza-pp/GIZA++-v2/snt2cooc.out tools/bin/ && \
    cp tools/fast_align/build/fast_align tools/bin/ && \
    cp tools/fast_align/build/atools tools/bin/

ENV PBSMT_TOOLS=/opt/pbsmt/tools
ENV PATH="/opt/pbsmt/tools/mosesdecoder/bin:/opt/pbsmt/tools/kenlm/build/bin:/opt/pbsmt/tools/bin:${PATH}"

WORKDIR /workspace
