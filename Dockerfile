# Uses micromamba to create the conda env from environment.yml
FROM mambaorg/micromamba:1.5.10

# Keep things predictable
WORKDIR /app

# Copy only what we need to build the environment first (better caching)
COPY --chown=micromamba:micromamba environment.yml /app/environment.yml

# Create the environment (name must match your file: "scrap")
RUN micromamba create -y -f /app/environment.yml && \
    micromamba clean -a -y

# Use the conda env for all subsequent commands
ENV MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["/usr/local/bin/_entrypoint.sh", "/bin/bash", "-lc"]

# Copy the code (NOT data/ or output/ because they are dockerignored/mounted)
COPY --chown=micromamba:micromamba . /app

# Default command (compose can override)
CMD python pipeline_runner.py --help
