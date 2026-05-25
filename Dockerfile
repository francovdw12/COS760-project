# CPU-only image for COS760 cross-lingual embedding project.
# Build: docker compose build
# Run:   docker compose run --rm cos760 rq1
#        docker compose run --rm cos760 rq2
#        docker compose run --rm cos760 viz1
#        docker compose run --rm cos760 viz2
#        docker compose run --rm cos760 all

FROM python:3.11-slim

# Build-time system deps for fasttext-wheel compilation and VecMap (git not needed at runtime).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU-only wheel before requirements.txt so it satisfies the torch dep.
# NOTE: The +cpu local-version suffix only exists from 2.6.0 onward on the PyTorch index.
# For older releases the wheel is served without the suffix but is still CPU-only.
RUN pip install --no-cache-dir \
    torch==2.7.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the Python dependencies.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source and data (see .dockerignore for exclusions).
COPY . .

# Ensure the entrypoint is executable.
RUN chmod +x docker-entrypoint.sh

# Generated artifacts land in these directories; they can be bind-mounted
# by docker-compose.yml so results are available on the host after a run.
RUN mkdir -p results outputs/ner embeddings data/subsets

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["help"]
