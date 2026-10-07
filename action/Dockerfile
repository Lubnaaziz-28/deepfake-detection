FROM python:3.11-slim

# REPO_REF is a branch name (e.g. main) or tag whose archive pip installs.
ARG REPO_REF=main

RUN pip install --no-cache-dir \
    "https://github.com/Lubnaaziz-28/deepfake-detection/archive/refs/heads/${REPO_REF}.tar.gz"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
