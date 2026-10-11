# Contributing

## Set up

```sh
git clone https://github.com/bahdan111/stepbook && cd stepbook
uv sync --all-extras
```

Python 3.11 or newer and uv (https://docs.astral.sh/uv/). `uv sync` installs the package and the development tools
into `.venv`.

## Check your change

```sh
uv run ruff format --check
uv run ruff check
uv run mypy
uv run pytest -q
```

These are the same checks CI runs. The tests run offline and call no model; `uv run ruff format` fixes the
formatting. To run ruff on every commit, install the hooks once with `uvx pre-commit install`.

## Pull requests

`main` accepts no direct pushes. Work on a branch, open a pull request, and it is squash-merged once CI passes; CI
runs on every update to the pull request and again on `main` after the merge.

## The repository is the plugin

Claude Code and Codex install this repository as it is: `.claude-plugin/`, `.codex-plugin/` and `.agents/plugins/`
hold the manifests and catalogs, `hooks/` and `skills/stepbook/` are what the agents load, and the plugins run the
`stepbook` command from PyPI (or the one the user installed). There is nothing to build. Try your checkout in Claude
Code with `claude --plugin-dir .`. Keep the root free of other agent-loaded folders (`commands/`, `agents/`,
`.mcp.json`); built-in task skills belong in `stepbook/templates/`. Tests guard both.

## Versions and releases

The version lives in `pyproject.toml`, both plugin manifests and the onboarding skill's `uvx stepbook@X.Y.Z`; change
all of them with `scripts/bump-version.sh X.Y.Z`, and a test checks they agree. Pushing a tag `vX.Y.Z` runs the release
workflow: the checks, a PyPI upload through PyPI trusted publishing (configured once on pypi.org for this repository's
`release.yml` and its `pypi` environment) and a GitHub release with generated notes.

## What to keep stable

The `stepbook` command, its flags and exit codes, the step JSON (`reasoning`, `state_patch`, `action`), the files in a
run folder (`state.json`, `events.jsonl`) and the skill folder layout are the interfaces people build on. A change to
any of them needs a note in the pull request.
