---
name: stepbook
description: Onboard and run long-horizon tasks with durable evidence and restartable state, including measured improvement loops, procedures and test-fixing sweeps; also relays a chat message starting with @loop to the running loop. Requires a measured result, operator checks or a checklist with evidence; not a substitute for unbounded product design.
allowed-tools: Bash, Read, Glob
---

# stepbook: long-horizon tasks without a growing transcript

`stepbook` runs a *skill* (instructions plus a JSON state schema) one step at a time. Each step sees the
instructions, the validated state, the latest observation, a short digest of recent steps and the operator's notes.
The runtime validates every state patch and journals every step, so a run survives crashes and compaction.

Use the harness when the task has a machine-checkable result or a known long checklist. When success is undefined,
agree what "done" means first.

## Running stepbook from your agent

Commands below call plain `stepbook`. Use it when `command -v stepbook` finds it (the user installed it with
`uv tool install stepbook`); otherwise run `uvx stepbook@0.1.0` in its place, which needs `uv`. If neither works, ask
the user to install uv (https://docs.astral.sh/uv/) or run `uv tool install stepbook`. Then, per agent:

- **Claude Code:** steps run on Opus by default; pass `--model` only if the user picks another controller. If the chat
  itself runs a weaker model, ask the user to switch to Opus before you make strategic choices for them. Login check:
  `claude auth status` shows a claude.ai login.
- **Codex:** add `--executor codex` to `run` and `author` and leave out `--model`, so each step runs on Codex with the
  user's Codex login. Login check: `codex login status`. In a Docker container, Codex's own read sandbox needs the
  container started with `--security-opt seccomp=unconfined`; without it every in-step read fails, so tell the user,
  or use a host worktree.
- **Any other agent with a shell** (OpenCode, for example): for OpenCode add `--executor opencode --model
  <provider>/<model>`; login check: `opencode auth list`.

Use the user's existing login. Never configure an API key, collect tokens or switch billing. stepbook refuses to start
when the agent CLI would bill an API key; never pass `--allow-api-billing` yourself, only when the user asks for it.
Check the login before a model-backed launch. Reported USD is the list-price value of the tokens, not a bill; say so
when you report it. Application models (for
example Haiku writing answers inside the user's eval) belong to their config; `--model` does not change them.

Operator notes reach a running loop in one of two ways. Where the plugin's hooks run (Claude Code, or Codex after the
user trusts the plugin's hooks), the user types `@loop <note>` in chat. Everywhere else, relay the user's note with
`stepbook say "<note>"` and tell the user it was delivered only after the command succeeds. Run `stepbook pause` only
when the user asks to pause.

## Onboard any long-horizon task

Do this in the chat before launching.

1. **Understand how done is checked:** a measured result, check commands, or a checklist with evidence. Find the
   commands yourself in the README, code, build config and CI; ask the user only for scope and decisions you cannot
   discover. For docs or migration work, record the affected files, generated-source rules and a baseline result.
2. **Pick the skill:** use a built-in skill by name: `eval-loop` for a measured result, `checklist` for a
   procedure or checklist (failing tests included: make the test command a check). Draft with `author` only in two
   cases. If the user names a skill they already have (a folder with a `SKILL.md`, for example under
   `.claude/skills/`), keep their text: `stepbook author ./<name>-state --from <that folder>` adds only a state
   schema, an initial state and a short section on running it step by step. Show the user the added section and the
   check ids in its initial state, and pass matching `--check "id=<command>"` values at launch. If neither built-in
   skill fits, draft one with `stepbook author ./drafted-skill "<procedure>"` and review it with the user.
3. **Draft the plan** as a JSON merge patch on that skill's initial state (see the skill sections below). Omit
   unchanged defaults: `null` in a merge patch deletes a key.
   Check the plan with `stepbook validate <skill-dir> --plan .stepbook/plan.json` before showing it.
4. **Show a plain summary** as a chat message, then ask for go or edits: what will be done, how done is checked, the
   models, the limits and anything risky. A multiple-choice prompt alone hides the plan. Explain cost in plain words:
   the subscription login, quota use unknown, the step limit and timeouts that are enforced, the agent-counted run cap
   that is not, and no dollar ceiling. Do not ask twice for an approval the user already gave.
5. **On go**, write `.stepbook/plan.json` in the working directory and launch from there with
   `--plan .stepbook/plan.json`. Pass the goal with `--goal`, checks with `--check "id=<command>"` and protected
   rules with `--constraint`; command-line values win over the plan. Add your agent's flags from "Running stepbook
   from your agent". Launch in the background, keep the process handle and the log directory, and tell the user how
   to follow the run. Leave `plan.json` unchanged afterwards, because a resume reads it again; send later changes as
   operator notes.

**Size the run in the plan summary.** A step is one unit of work: the step's agent reads the state and the project,
then returns one action, which can be a fix followed by its tests, a chain of commands or a whole eval run. Allow
about one step per checklist item or eval trial plus a few to get started. Propose a `--steps` limit, say that
`--steps` counts all steps across resumes, and raise `--cmd-timeout` (default 2 hours) if a check or eval command
can take longer.

## Shell mode: container, or a fresh worktree with explicit consent

Shell mode refuses to start outside a container (it looks for container markers) unless `STEPBOOK_SANDBOX=1` is
declared for an environment that really is isolated; never set it just to get past the refusal. Outside a container,
tell the user:

> The agent will run shell commands as you, with your access to files, credentials and the network. The guards and
> a separate Git worktree are not isolation and cannot prevent effects elsewhere on your machine.

Offer a container first: an image with the project's dependencies and a logged-in agent CLI (the README's "Check
the install in a container" recipe shows the stepbook part). As the local alternative,
offer a fresh sibling Git worktree on a new branch, and show the paths, branch, goal, checks, limits and exact
command. Launch it only after an explicit yes to this local route in this conversation; a generic "go" or an earlier
approval does not count. Never use it on the user's main working tree.

Before creating the worktree, check the source repository for staged, unstaged and untracked changes. Stop while
there are changes the user has not chosen to include; transfer only the files they name, and commit those in the
new worktree before launch. Record the source branch, HEAD and status. Then, from the source repository root:

```sh
git worktree add ../<repo>-stepbook-<date> -b stepbook-worktree/<date>
```

The new worktree has none of the project's installed dependencies or virtual environments: install what the checks
need there, or record the install command in the plan, before the baseline. Keep checkpoints and guards on: the run
creates its own `stepbook/<date>-<time>` branch and commits after each changing step; that is the branch the user
reviews and merges. The checklist launch, for example (plus your agent's flags):

```sh
stepbook run checklist --workdir . --plan .stepbook/plan.json --goal "<reviewed goal>" --check "tests=<reviewed check command>" --constraint "<reviewed boundary>" --steps 12 --unsandboxed-i-accept-the-risk --log-dir .stepbook/local-run
```

`--unsandboxed-i-accept-the-risk` only skips the sandbox check. Afterwards, show the harness branch's diff against the
recorded HEAD, the check results and what is left; confirm the original tree is unchanged. The user reviews and
merges; do not merge or publish for them. Copy `.stepbook/` out of the worktree (for example to
`../<repo>-stepbook-<date>-logs`), then remove the worktree without forcing it:

```sh
git worktree remove ../<repo>-stepbook-<date>
```

## eval-loop: improve a measured result

The default for "improve a metric of my project", however many commands the eval takes. No adapter is needed.
Read the README and eval code; find the command sequence, where results land, the primary metric and its direction,
per-case outcomes, the files to keep fixed, and the case set. Check cheaply that it runs (a limited or dry mode if
there is one); run a full paid eval only inside the agreed budget.

The plan holds `facts.eval` (ordered commands, results and how to read them, metric direction, rough duration),
`facts.fixed_files`, `facts.case_set` (count and fingerprint), `protocol` (baseline and trial repeats),
`budget.runs_max` from the user's words (omit it when they gave none), and 5 to 8 ideas under `hypotheses`, each with
`kind`, `rationale`, `origin` (`user`, `docs` or `agent`) and `status: "todo"`, plus `how` when known. The loop adds
its own ideas in the same shape. In the summary, list the ideas and where each came from. Pass fixed files as
`--constraint` and repeat the budget in `--goal`. The run cap is counted by the agent, not enforced by the host.

```sh
stepbook run eval-loop --workdir /path/to/disposable-checkout --plan .stepbook/plan.json --goal "<what to improve, budget in the user's words>" --constraint "<fixed file or protected rule>" --steps 80
```

## checklist: a procedure or checklist

For a procedure, seed `checklist` with items keyed by id, each `{text, status: "todo", done_when}` (statuses are
`todo`, `doing`, `done`, `blocked` and `skipped`), and put known context in `facts`. Make one item per unit the user
wants to track (a service, a docs set, a batch of cases), not one per file; keep long inventories in a repository file
that `facts` names:

```sh
stepbook run checklist --workdir /path/to/disposable-checkout --plan .stepbook/plan.json --goal "<what to do>" --check "tests=<cmd>" --check "lint=<cmd>" --constraint "<rule>" --steps 80
```

The host reruns the checks on `done`; failing checks end the run with exit 1.

## While it runs

- **Progress:** `stepbook status` prints the result line, what the harness recorded, what the agent reported, and the
  exact resume command. `stepbook report <log-dir>` prints the full report.
- **Steer:** a note (see "Running stepbook from your agent") is appended to `<log-dir>/inbox.txt` and read before
  the next step; `@loop pause` or `stepbook pause` stops cleanly. Confirm a relayed note in one line and do not act
  on it yourself.
- **Resume:** `stepbook run --resume <log-dir>` continues with the recorded launch (no folder: the active, else the
  last run). Add `--steps N` to raise the limit, `--reopen "<note>"` for a finished run, and
  `--unsandboxed-i-accept-the-risk` again if the launch had it. A fresh chat, or another agent, adds its own executor
  flags from "Running stepbook from your agent".
- **Recover:** if resume reports an unresolved action, run `stepbook recover <log-dir>`, check that the old process
  and its children have stopped, and inspect the outputs. Only for a verified outcome, append a receipt with
  `stepbook recover <log-dir> --action-id "<id>" --receipt ./receipt.json --reason "<how you verified it>"` (fields
  `result`, `info`, `done`). Never rerun a possibly paid action and never invent an exit status.

## Optional: the strict adapter route

Use `eval-loop-adapter` when the user wants host-computed keep/revert verdicts and a host-enforced eval budget. Start with an
onboarding brief:

```markdown
## Improvement-loop onboarding

**Fit:** suitable / not yet suitable, and why.
**Found:** evaluator command, result format, configuration target, working-tree status.
**Measure:** primary and guardrail metrics, any limit metric and its stated ceiling, development-set size, protected files, gaps.
**Known constraints:** output rules the scorer enforces.
**First investigation or experiment:** evidence to inspect or a reversible change.
**Run plan:** models, repeats, approximate eval count, log location.
**Before I act:** the smallest decision needed from the user.
```

An inspection-only request does not authorize a run. Read the scoring code and list the output rules it enforces;
never invent a metric or a ceiling the repository does not state.

- If the eval already writes one JSON or CSV row per case with an id and a pass/fail field, scaffold the no-code
  adapter with `stepbook init ./stepbook-loop --config-only` and fill `stepbook-loop/eval.json`. Paths in it are
  relative to its own folder. Declare a limit metric as
  `"metric_fields": {"latency_s": {"direction": "lower", "max": 1.2}}` and usage through `cost_artifact`.
- Otherwise scaffold a Python adapter with `stepbook init ./stepbook-loop` and fill `run_eval()`. State the missing
  glue before writing it.
- Verify it before any model call: `stepbook adapter check "stepbook adapter generic stepbook-loop/eval.json"`.
- Adapter loops change configuration only; for code changes, use `eval-loop` in shell mode.

```sh
stepbook run stepbook-loop/skill --adapter "python stepbook-loop/adapter.py" --goal "<objective>" --constraint "<human boundary>" --runs-max 20 --steps 80 --evidence-root ./config.json --evidence-root ./dev-results --evidence-root ./dev --log-dir .stepbook/adapter-run
```

Here `--runs-max` is enforced by the host. Grant evidence roots for the config, the run
artifacts, dev documents and source, and never a directory with held-out answers.

## What to tell the user

- `state.json` is the agent's readable checkpoint. The journal (`events.jsonl`) and the result files show what
  actually ran, and every step's reasoning is kept there.
- A saved receipt lets a run continue without redoing work; an uncertain outcome needs reconciling. Do not promise
  recovery from every crash.
- Guards, worktrees and evidence roots are not isolation; executable adapters and shell runs belong in a container
  or a consented disposable worktree.
