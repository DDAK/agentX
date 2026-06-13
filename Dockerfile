# ── build stage ───────────────────────────────────────────────────────────────
FROM rust:1.95-bookworm AS builder

WORKDIR /build

# Cache dependency compilation by copying manifests first.
COPY Cargo.toml Cargo.lock ./

# Create stub sources so `cargo build` can resolve the dependency graph.
RUN mkdir -p src && echo "fn main(){}" > src/main.rs && echo "" > src/lib.rs
RUN cargo build --release 2>/dev/null; rm -f target/release/deps/agentx*

# Copy actual sources and build for real.
COPY src ./src
COPY tests ./tests
RUN cargo build --release

# ── runtime stage ─────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

# Install minimal runtime deps (TLS certs for outbound HTTPS, netcat for healthcheck).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for safety.
RUN useradd -m -u 1000 agentx
USER agentx

WORKDIR /workspace

COPY --from=builder /build/target/release/agentx /usr/local/bin/agentx

ENTRYPOINT ["agentx"]
