# Runs: files, controls and recovery

## The run folder

A run lives in `.stepbook/<skill>-<timestamp>/` under the launch directory (or `--log-dir`).

| File | Purpose |
|---|---|
| `state.json` | The agent's readable checkpoint: facts, progress and decisions. |
| `events.jsonl` | Append-only journal of steps, actions and results; a resume rebuilds state from it. |
| `launch.json` | The launch arguments, which `stepbook run --resume` reuses. |
| `inbox.txt` | Your notes, read before the next step; a line `pause` stops the run cleanly. |
| `report.md` | The report written when the run ends. |

## Controls

In Claude Code, a message starting with `@loop ` goes to the inbox (`@loop pause` pauses). From a shell:

```sh
stepbook status
stepbook say "Do not change the tests"
stepbook pause
stepbook report .stepbook/my-run
```

`say`, `pause` and `status` act on the run started from the current directory (`.stepbook/ACTIVE`, else
`.stepbook/LAST_RUN`); from anywhere else, pass `--log-dir <run>`. `status` starts with a one-line result, then the
checklist and what stepbook recorded, and ends with the exact resume command.

| Exit | Meaning |
|---|---|
| 0 | Goal met: `done` with all checks passing, or a normal adapter finish. |
| 1 | Checks failed. |
| 2 | Refused: invalid arguments, or a launch or resume stepbook would not start. |
| 3 | Ended incomplete: the agent stopped, the step limit was reached, the run was paused or it aborted. |

A `done` with no checks configured exits 0 on the agent's word, and `status` says "No checks configured".

## Resume

```sh
stepbook run --resume .stepbook/my-run --steps 60
```

A resume reuses the launch recorded in the run folder; without a folder it takes the active run, else the last one.
Add a higher `--steps` if the limit was reached, and `--reopen "<why>"` for a finished run. If the launch needed
`--unsandboxed-i-accept-the-risk`, pass it again. To change any other setting, repeat the whole launch with
`--resume --log-dir .stepbook/my-run`. Leave `plan.json` unchanged: a resume applies it again and then replays the
journal, so no progress is lost.

## Hand a run to another agent

Pause it and wait until the old process exits (code 3). Then resume it with that agent's flags:
`stepbook run --resume .stepbook/my-run --executor codex`, or `--executor opencode --model <provider>/<model>`. Each
agent needs stepbook installed.

## Recover a crashed run

A crash keeps every journaled step, and a resume never replays a command whose result was recorded. If resume stops at
an action with no recorded result, run `stepbook recover .stepbook/my-run` to see it. Check that the old processes
have stopped, and look at the command's output. If you can confirm what happened, save it as a receipt and append it:

```json
{"result": "Inventory written: 120 pages; completion marker shows exit 0.", "info": {"ok": true, "rc": 0}, "done": false}
```

```sh
stepbook recover .stepbook/my-run --action-id "<id from recover>" --receipt receipt.json --reason "Process tree stopped; marker shows exit 0"
```

Then resume as above. If you cannot confirm the outcome, stop and investigate; never invent an exit status or rerun a
command that may have cost money.
