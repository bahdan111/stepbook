# Notes for coding agents working on stepbook

- Set up with `uv sync --all-extras`. Check every change with `uv run ruff format --check`, `uv run ruff check`,
  `uv run mypy` and `uv run pytest -q` (offline, no model calls); run one test file with `uv run pytest -q tests/<file>`.
- The repository root is the Claude Code and Codex plugin. Do not add `commands/`, `agents/` or `.mcp.json` at the
  root; built-in skills live in `stepbook/templates/`.
- Keep stable: the `stepbook` command, its flags and exit codes, the step JSON (`reasoning`, `state_patch`, `action`),
  the run folder files (`state.json`, `events.jsonl`, `launch.json`) and the skill folder layout. Tests pin rendered
  prompts (`tests/golden/`) and journal bytes; regenerate them only for a deliberate change and say why.
- Never decide behaviour by matching free text (model prose, tool output). Use exit codes, JSON fields or typed values.
- Work on a branch and open a pull request; `main` accepts no direct pushes and merges only with CI green.
- Change the version only with `scripts/bump-version.sh X.Y.Z`.
