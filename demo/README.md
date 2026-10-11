# Demo workloads

Small projects for trying stepbook and for the figures in the main README. You don't need any of them to use
stepbook.

| Folder | What it is |
|---|---|
| `tatqa/` | A small retrieval QA pipeline over TAT-QA (tables and text from financial reports; Zhu et al., ACL 2021). `adapter.py` exposes it to the `eval-loop-adapter` skill as four commands: `describe`, `apply <h>`, `run <label>` and `decide <h> keep\|revert`. `config.json` holds every setting the agent may change. The example step in the README's "One step in detail" diagram comes from a run on it. |
| `fixtests/` | A test-repair workload: a 30-function package, and `mutants.py`, which injects known bugs that break known tests. `generate.py` builds larger variants. |
| `fixtests_big/` | Output of `python demo/fixtests/generate.py --modules 20 --out demo/fixtests_big --seed 0`: 20 modules, 100 tests and `mutants.json`, one verified bug per function. The coding task behind the OpenCode figure feeds these bugs in one at a time. |

Run the TAT-QA demo from a checkout. It needs the `demo` extra and a logged-in `claude` CLI, which the pipeline uses
to answer. The first run downloads the TAT-QA dev set into `tatqa/data/`.

```sh
uv sync --extra demo
uv run stepbook run eval-loop-adapter --adapter "python demo/tatqa/adapter.py" --steps 24
```
