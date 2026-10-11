# The strict adapter route (`eval-loop-adapter`)

`eval-loop-adapter` runs in adapter mode, where the agent gets no shell. It can only try changes your adapter knows how to
apply (usually configuration values), run the eval through the adapter, and keep or revert. stepbook requires repeated
runs before a keep, compares them with the baseline, and refuses eval runs beyond `--runs-max`. Choose it when an eval
run is expensive or noisy, or when the agent must not have a shell; otherwise use `eval-loop`.

## The adapter

The adapter is a small script with commands to describe the experiment, apply a change, run the eval and keep or
revert. Each command answers with its exit code:

| Command | Exit code and output |
|---|---|
| `describe` | Free text for the agent; stepbook never parses it |
| `apply <id>` | 0 applied, 1 refused (the output is the reason the agent sees), 3 applied but the change also needs code edits (the output says what) |
| `run <label>` | 0 and exactly one JSON result on stdout |
| `decide <id> keep\|revert` | 0 recorded, 1 refused |
| `status` | JSON: `applied`, `decided`, `phase`, `phases_complete`, `hypotheses` (a list of `{id, kind}`) |

stepbook decides only from exit codes and the JSON of `run` and `status`.

## Set one up

Scaffold one, fill in `run_eval()`, check it, then launch (inside isolation):

```sh
stepbook init my-loop
stepbook adapter check "python my-loop/adapter.py"
stepbook run my-loop/skill --adapter "python my-loop/adapter.py" --runs-max 20 --steps 60
```

If your eval already writes one JSON or CSV row per case with a pass/fail field, `stepbook init my-loop --config-only`
creates an `eval.json` for the bundled generic adapter instead, with no code to write. The agent makes each keep or
revert decision; `status` and the report show it next to the verdict stepbook computed and flag every decision that
goes against it.
