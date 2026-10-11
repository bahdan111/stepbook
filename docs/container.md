# Running in a container

A run executes real shell commands as you, so a container is the safest place for it. stepbook detects a container on
its own; elsewhere it refuses to start unless you set `STEPBOOK_SANDBOX=1` or pass `--unsandboxed-i-accept-the-risk`.

For a run with a model, the image needs your project's dependencies, stepbook (`uv tool install stepbook`) and a
logged-in agent CLI, and the container needs network access to the model. For Codex steps, start the container with
`--security-opt seccomp=unconfined`: Codex reads files through its own sandbox, which Docker's default profile blocks.

## Check the install offline

This recipe only checks that stepbook installs and runs: it has no network, and a scripted agent repairs a failing
test in a small fixture repository.

Copy the source (without `.git`, environments or run logs) into `sandbox-copy/` and save this as
`sandbox-copy/Dockerfile.smoke`:

```dockerfile
FROM python:3.12-slim
COPY . /opt/stepbook
WORKDIR /opt/stepbook
RUN pip install .
WORKDIR /work
```

Build it with `docker build --tag stepbook-smoke --file sandbox-copy/Dockerfile.smoke sandbox-copy` (needs network
for dependencies), then run:

```sh
docker run --rm --network none --mount "type=bind,src=$PWD/sandbox-copy,dst=/work" \
  --workdir /work stepbook-smoke \
  stepbook run checklist --workdir /work/tests/fixtures/offline/toy-repo \
  --goal "Make the failing test pass without editing tests" \
  --check "tests=python -m unittest discover -s tests -q" --constraint "Do not edit tests" \
  --command "python /work/tests/fixtures/offline/scripted_agent.py" \
  --steps 8 --no-checkpoint --log-dir /work/.stepbook/smoke
```
