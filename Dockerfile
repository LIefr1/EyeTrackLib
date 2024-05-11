FROM continuumio/miniconda3

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN conda env create --file environment.yml

SHELL ["conda", "run", "-n", "eye-tracking"]

WORKDIR /app

# Copy script from volume mount (example)
COPY . .

# Run your Python script
# Use environment variable if defined (adjust path if needed)
CMD ["python", "main.py"]
