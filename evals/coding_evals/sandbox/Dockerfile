# Sandbox image for grading model-generated code.
# Build once per EC2 instance:  docker build -t coding-eval-sandbox:latest coding_eval/sandbox
FROM python:3.11-slim

# pytest-json-report is required: we read structured results from a JSON file
# rather than scraping stdout, because a solution can print fake pytest output.
RUN pip install --no-cache-dir pytest==8.2.0 pytest-json-report==1.5.0 \
 && rm -rf /root/.cache

COPY runner.py /opt/runner.py
RUN chmod 0555 /opt/runner.py

# Scratch space. The container is run with --read-only plus a tmpfs on /work,
# so nothing a solution writes survives the run.
RUN mkdir -p /work && chmod 0777 /work
WORKDIR /work

# No network, no root: enforced again at `docker run` time by the host wrapper.
USER nobody

ENTRYPOINT []
CMD ["python", "/opt/runner.py"]
