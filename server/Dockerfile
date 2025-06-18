FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /server

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking
ENV UV_LINK_MODE=copy

# Install dependencies first for better caching
# RUN --mount=type=cache,target=/root/.cache/uv \
#     --mount=type=bind,source=uv.lock,target=uv.lock \
#     --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
#     uv sync --locked --no-install-project --no-install-workspace 

# Copy project files
COPY . /server/

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Add venv binaries to PATH
ENV PATH="/server/.venv/bin:$PATH"
ENV PYTHONPATH=/server

# Reset entrypoint
ENTRYPOINT []

# Run FastAPI application with multiple workers
CMD ["fastapi", "run", "--host", "0.0.0.0", "--workers", "4", "app/main.py"]