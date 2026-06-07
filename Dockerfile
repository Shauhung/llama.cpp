FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN mkdir build && cd build && \
    cmake .. -DLLAMA_GGML_BACKEND_DL=OFF && \
    cmake --build . --config Release --target llama-server

RUN mkdir -p /src/out && \
    find build -name "*.so*" -exec cp {} /src/out/ \; && \
    cp build/bin/llama-server /src/out/

FROM ubuntu:24.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/out/* /app/

ENV LD_LIBRARY_PATH=/app

EXPOSE 8080

ENTRYPOINT ["/app/llama-server"]
