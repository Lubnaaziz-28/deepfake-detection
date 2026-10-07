FROM python:3.11-slim

ARG REPO_REF=main

RUN pip install --no-cache-dir \
    "deepfake-scan @ git+https://github.com/Lubnaaziz-28/deepfake-detection.git@${REPO_REF}"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
