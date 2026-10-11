# Running stepbook without an agent chat

## A plan and a launch

Write the plan in the chosen skill's structure, for example `.stepbook/plan.json` for `checklist`:

```json
{"checklist": {
  "billing": {"text": "Move the billing service to the new HTTP client", "status": "todo",
              "done_when": "The billing check passes"},
  "docs": {"text": "Update the 40 docs pages that mention the old client", "status": "todo",
           "done_when": "No page mentions the old client"}},
 "facts": {"inventory": "docs/migration/sites.txt lists every call site"}}
```

Checklist items are keyed by id; `status` is `todo`, `doing`, `done`, `blocked` or `skipped`. Make one item per unit you
want to track (a service, the docs set), not per file, and keep long inventories in a file the plan names. Then, inside
a container or worktree:

```sh
stepbook run checklist --workdir /path/to/disposable-checkout \
  --plan .stepbook/plan.json --goal "Make the test suite pass without editing tests" \
  --check "tests=python -m pytest -q" --constraint "Do not edit tests" \
  --steps 30 --log-dir .stepbook/my-run
```

Without `--executor`, each step runs the Claude CLI with Opus. The plan may carry the goal, checks and constraints
itself; a command-line value replaces the plan's only when you give it. For `eval-loop`, the plan records the eval
commands, the metric and its direction, files to keep fixed, the case set, how many times to repeat runs (two by
default) and the ideas to try. `stepbook run --help` lists the routine options; `--help-all` lists every option.

## Other agents

**Codex.** `--executor codex` starts a fresh Codex call per step with your Codex login:

```sh
stepbook run checklist --workdir /path/to/disposable-checkout \
  --plan .stepbook/plan.json --goal "Make the test suite pass without editing tests" \
  --check "tests=python -m pytest -q" --constraint "Do not edit tests" \
  --steps 30 --log-dir .stepbook/codex-run --executor codex
```

Each Codex step reads files through Codex's own read-only sandbox, which needs Linux user namespaces. Docker's default
security profile blocks them, so in a container every read inside a step fails and Codex can stop early. Start the
container with `--security-opt seccomp=unconfined`, or run Codex steps from a worktree on the host. Claude and
OpenCode steps don't need this.

**OpenCode.** The same, with OpenCode's own login (any provider it supports; `--model` takes `provider/model`). Each
step runs as a read-only OpenCode agent: it can read files but not edit them or run commands; stepbook runs the
action it returns.

```sh
stepbook run checklist --workdir /path/to/disposable-checkout \
  --plan .stepbook/plan.json --goal "Make the test suite pass without editing tests" \
  --check "tests=python -m pytest -q" --constraint "Do not edit tests" \
  --steps 30 --log-dir .stepbook/opencode-run --executor opencode --model openai/gpt-5.6-sol
```

**Any other agent.** `--command "<cmd>"` runs `<cmd>` once per step. It gets the prompt on stdin and prints one JSON
object: `reasoning`, `state_patch` (a merge patch) and `action` (a shell command, `check`, `done` or `stop <reason>`),
plus optional `usage` with `input_tokens`, `output_tokens` and `cost_usd`. `{schema_file}` and `{system_prompt_file}`
are substituted in the command, which runs in the `--workdir` folder (give a script outside it an absolute path). Exit
`124` if your agent's own turn timed out, so stepbook does not send the call again. A ten-line agent that only runs
the checks, saved as `agent.py` in the working folder:

```python
import json
import sys

prompt = sys.stdin.read()
assert prompt, "Expected a rendered prompt on stdin"
reply = {
    "reasoning": "Run the supplied checks; make no edits.",
    "state_patch": {}, "action": "check",
}
print(json.dumps(reply))
```

```sh
stepbook run checklist --workdir /path/to/disposable-checkout \
  --goal "Inspect checks only" --check "tests=python -m pytest -q" \
  --command "python agent.py" --steps 1 --no-checkpoint \
  --log-dir .stepbook/contract-run
```

## Example: a docs migration

"Replace `v1/orders` with `v2/orders` on 120 pages and keep links green." The agent first finds the page inventory and
the checker commands in the repository, records a passing baseline, and seeds four checklist items: scope, baseline,
replacements, final checks. At the end it shows the covered pages, the content diff and a diff of the checker's own
sources against the baseline.

```sh
stepbook run checklist --workdir /path/to/disposable-checkout \
  --plan .stepbook/plan.json --goal "Migrate the inventoried docs API references" \
  --check "links=python tools/check_links.py" --check "names=python tools/check_names.py" \
  --constraint "Preserve check sources, configuration and unrelated content" \
  --steps 60 --log-dir .stepbook/docs-run
```

## Limits, timeouts and cost

A step is one unit of work: the agent hands back one action, such as a fix followed by its tests, a chain of commands
or a whole eval run, and stepbook runs it, records the result and commits the changes. `--steps` counts all steps
across resumes; without it, a run continues until the agent stops or you pause it.

Each shell command is stopped after `--cmd-timeout` seconds, 2 hours by default. Raise it if a test or eval run can
take longer, for example `--cmd-timeout 14400` for 4 hours.

Steps use your agent subscription, and how much quota a task uses is not predictable. Reported USD is the list-price
value of the tokens, not a bill, and there is no dollar ceiling. stepbook refuses to start if the agent CLI would bill
an API key; pass `--allow-api-billing` (or set `STEPBOOK_ALLOW_API_BILLING=1`) to allow it. `--runs-max` caps eval
runs: with an adapter stepbook refuses the extra run, and otherwise the agent counts runs itself, so nothing stops an
extra one. `--state-max-chars` abbreviates older state in the prompt on a best-effort basis; the full state stays on
disk.
