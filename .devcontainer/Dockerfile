FROM python:3.12.3-slim

# Set environment variable to avoid tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Create a non-privileged user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --uid "${UID}" \
    appuser

# Install Poetry and configure it
RUN pip install poetry && \
    poetry config virtualenvs.create false

# Switch to root user to install system dependencies
USER root

# Install system dependencies, including OpenSSL, ffmpeg, and Git
USER root
RUN apt-get update && \
    apt-get install -y \
    gcc \
    git \
    git-lfs \
    ffmpeg \
    libssl-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy application code into the container
COPY . /app
WORKDIR /app

# Switch back to the non-privileged user
USER appuser
