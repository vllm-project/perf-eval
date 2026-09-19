ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG ARM
COPY . /opt/dsv4-ab
RUN uv venv --system-site-packages --no-python-downloads --python 3.12 /opt/dsv4-ab/.venv \
    && /opt/dsv4-ab/.venv/bin/python /opt/dsv4-ab/apply_overlay.py "${ARM}"
ENV PATH="/opt/dsv4-ab/.venv/bin:${PATH}"
