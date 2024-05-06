FROM continuumio/miniconda3

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Check if GPU is available and set CUDA version accordingly
RUN if nvidia-smi; then \
    CUDA_VERSION=11.2; \
    else \
    CUDA_VERSION=""; \
    fi

# Create Conda environment with Python 3.8 and PyTorch (and optional CUDA)
RUN conda create -n pytorch_env python=3.8 pytorch torchvision torchaudio cudatoolkit=$CUDA_VERSION -c pytorch -y && \
    conda clean -afy

SHELL ["conda", "run", "-n", "pytorch_env", "/bin/bash", "-c"]

WORKDIR /app

# Define script path from environment variable (optional)
# ENV SCRIPT_PATH=/path/to/your/script.py

# Copy script from volume mount (example)
COPY . /app

# Run your Python script
# Use environment variable if defined (adjust path if needed)
CMD ["python", "${SCRIPT_PATH:-/app/your_script.py}"]
