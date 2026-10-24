# --------------------------------------------------------------------------- #
# Groundwire public demo for a Hugging Face Docker Space.
#
# The Space repo holds only this file and README.md: the gateway itself is
# installed from GitHub, so pushing to GitHub + "Factory rebuild" in the
# Space settings deploys the latest code.
#
# Space secrets (Settings -> Variables and secrets):
#   GROQ_API_KEY                optional; without it answers come from the backup mock
#   GROUNDWIRE_APPROVER_TOKEN   optional; overrides the public demo token below
# --------------------------------------------------------------------------- #
FROM python:3.12-slim

ARG GROUNDWIRE_REPO=https://github.com/apmg001/groundwire
ARG GROUNDWIRE_REF=main

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Hugging Face runs Spaces as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Tarball install: no git needed in the image.
RUN pip install "${GROUNDWIRE_REPO}/archive/refs/heads/${GROUNDWIRE_REF}.tar.gz"
ADD --chown=user ${GROUNDWIRE_REPO}/raw/${GROUNDWIRE_REF}/config/demo.yaml config/demo.yaml

USER user
ENV HOME=/home/user \
    GROUNDWIRE_CONFIG_PATH=/home/user/app/config/demo.yaml \
    GROUNDWIRE_HOST=0.0.0.0 \
    GROUNDWIRE_PORT=7860 \
    GROUNDWIRE_WORKERS=1 \
    GROUNDWIRE_LOG_JSON=true \
    # Public by design: the demo page hands this token to every visitor
    # (dashboard.public_demo), and the demo's tools are fake. Never reuse it.
    GROUNDWIRE_APPROVER_TOKEN=public-demo-reviewer-token

EXPOSE 7860
ENTRYPOINT ["groundwire"]
