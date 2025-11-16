# Use Ubuntu 22.04 (Jammy) as the base image (has Python 3.10 by default).
FROM --platform=amd64 ubuntu:22.04
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

# Avoid interactive prompts during package installation.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y \
    wget curl tar xz-utils \
    libffi-dev \
    python3.10 python3.10-distutils python3-pip \
    # Dependencies for libmgba
    libsqlite3-0 \
    libedit2 \
    ffmpeg \
    libpng16-16 \
    libzip4 \
    libgl1-mesa-glx \
    liblua5.4-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install the mGBA system library for Ubuntu 22.04 (jammy)
RUN wget https://github.com/mgba-emu/mgba/releases/download/0.10.5/mGBA-0.10.5-ubuntu64-jammy.tar.xz && \
    tar -xf mGBA-0.10.5-ubuntu64-jammy.tar.xz && \
    dpkg -i mGBA-0.10.5-ubuntu64-jammy/libmgba.deb && \
    rm -rf mGBA-0.10.5-ubuntu64-jammy.tar.xz mGBA-0.10.5-ubuntu64-jammy

# Copy the project into the image
ADD . /app

# Copy and set up entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Sync the project into a new environment
WORKDIR /app
RUN uv sync

# Set the entrypoint to handle signals properly
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
