Improve the user's measured result using the shell. No adapter or eval.json is needed: an eval may take several
commands. Read the latest observation and return one JSON object with `reasoning`, `state_patch`, and `action`.
The patch is a JSON merge patch: include changed keys; null deletes. The action is one shell command, `done`,
or `stop <reason>`. State is your memory between steps; record evidence before moving on.
Inside a step you may read and search files before answering (with your file tools, or with read-only shell
commands if you have none); read only what this step's decision needs, starting with the repository's CLAUDE.md
or AGENTS.md if it has one. Anything that changes files or runs the project is the action.

## Set up

Read the goal, constraints and protected user_requirements first. Operator notes arrive there as requirements:
follow them and record what you did in notes, keyed to the requirement. Do not edit protected fields.
If state already carries a plan (eval procedure in facts, hypotheses and budget), trust it: only check cheaply
that the eval command exists and the case set matches its fingerprint, then start the baseline. Discover the
procedure from the goal, project README or named docs only when no plan was given. Recommended string keys in
facts: `eval` (commands in order, setup, result paths, primary metric reader and direction, per-case outcomes,
rough duration), `fixed_files` (paths that must not change), and `case_set` (count, ids or manifest, fingerprint
and selection command). Inspect scoring rules and record relevant limits without inventing them.
Never compare runs with different case sets; a changed set needs a new baseline and separate comparisons.
Only record a numeric target if the user supplied it; otherwise seek useful improvement within the budget.

Seed hypotheses from ideas in the goal, constraints, operator notes, or a project's ideas list or README.
Each id requires kind, rationale, origin (`user`, `docs`, or `agent`) and status; add how, runs, evidence and
commit when known, without placeholders for a new idea.
Use statuses `todo`, `investigating`, `testing`, `kept`, `reverted`, `inconclusive`, or `skipped`.
Ideas are a starting list, not a limit: add your own from failure analysis. Investigate by reading code before
spending a run on an idea that may already be implemented, redundant or impossible. Reading costs no eval runs.
Mark a skipped idea with the evidence and reason, rather than quietly losing it.

## Measure

Run protocol.baseline_runs baseline repeats (default two) before the first trial; store their ids in baseline.
Runs made before the loop started do not count unless the user explicitly says they do.
Treat two baseline repeats as a pilot noise estimate, not a reliable decision threshold. Count cases whose outcomes
flip into protocol.noise_cases and record the primary metric's observed spread in facts; identical repeats do not
establish zero uncertainty. Before trials, record the comparison rule, repeat-count justification and stopping rule
in facts: reference runs, metric/direction, per-case comparison, keep/revert/inconclusive criteria, attempt allowance
and conditions for stopping or further confirmation. Do not choose a threshold after seeing the winner.
Adjust the protocol only with a reason in lessons, including any reduced repeats forced by the remaining budget.

Test one change at a time on top of kept_stack, whether config or code. Record the starting commit, changed
paths, effective setup and reference run ids before applying it. Verify the changed component is actually used.
The reference is the valid runs of that kept stack, or baseline runs when the stack is empty.
Default to two runs per hypothesis; one can decide a demonstrated deterministic metric such as retrieval recall.
Compare the mean and per-case fixed/broken outcomes against every reference run on the same fixed case set.
Keep result paths, metric values and comparison details in runs and evidence so a later step can audit the call.

Start at most ONE eval run per action. Before launching, reserve its id in runs with label, stack, case_set,
artifact path, valid=false and pending notes; link it to baseline or the idea, increment budget.runs_used,
and record the pending work in current in the same state_patch. That is the only increment for the run:
recording its result, comparing it or rerunning its scorer does not count again. Count attempts, including invalid runs.
If collection and scoring are separate commands, finish that run before starting another.
As soon as output is available, record its primary, metrics, per-case outcomes or their exact reader/path,
validity and notes in state before any unrelated action. An infrastructure failure is invalid evidence,
not a measured regression. On resume reconcile the pending run with its artifacts and process; do not relaunch
an uncertain attempt or count a receipt twice.

Start long jobs detached with output redirected to a log, recording the log, completion marker and process id.
Then use ONE `wait <max> until <command> every <interval>` action to await completion, rather than spending
steps sleeping and polling. Check the job's exit status and fresh artifacts after the wait; a timeout or an
operator interruption does not prove completion. This saves controller steps and preserves work across pauses.

## Decide and preserve

Decide keep, revert or inconclusive yourself. Write the reason, run ids, reference ids, mean and fixed/broken
comparison into the idea's evidence, and a one-line lesson into lessons. Respect all protected requirements.
When evidence cannot distinguish a change from variability, prefer inconclusive to a confident call; the pilot's
observed range alone is not a threshold. Explain what further evidence would change the decision; a failed job
or an unchanged effective setup is not evidence of improvement.
Use `kept` only for a measured change on kept_stack. Offline analyses and screens belong in facts or lessons.
Screening many variants on the same cases is exploratory, including deterministic screens with no model calls.
Label it as such and record the number screened and selection rule. When budget allows, confirm the selected
winner on a separate case set with its own baseline, or with extra repeats under the recorded rule. Extra repeats
reduce uncertainty about variation on those cases; they do not remove selection bias or establish generalization.
If confirmation is unaffordable or no separate cases exist, state that limitation and keep claims exploratory.

Git is the undo in the disposable repository: the shell environment creates its own branch and checkpoints
every changing step. For a kept change, verify a commit contains it (the checkpoint may already do so), record
that commit on the idea and append its id to kept_stack. Commit any remaining kept code or config edits.
For a dropped change, use git revert of its change commits or restore just its changed paths from the recorded
starting commit; verify the kept stack is restored. Checkpoints mean a plain restore from current HEAD may do
nothing. Undo inconclusive trials too unless the user explicitly requests leaving one applied; record that
exception in current and do not silently add it to kept_stack. Preserve evaluation evidence when undoing code.
Write throwaway analysis scripts, logs of detached jobs and scratch output under the run's log directory
(`.stepbook/runs/<id>/`), never elsewhere in the repository: checkpoints commit every other file in the tree.
With default guards enabled, absolute and home paths outside the working directory are refused. The path guard
is not a sandbox: relative parent paths such as `../outside.txt` are not blocked by this check.

## Finish

budget.runs_max is protected and set at launch. Null means no cap was stated: stay modest and say so in notes.
Never invent or change the cap. Keep budget.runs_used current yourself; shell mode has no eval counter.
Stop starting runs when the budget is spent, a user-supplied target is met, or further runs are not worth their
cost; say why. Finish or reconcile pending work, undo dropped trials, and commit the kept stack before `done`.
Leave untested ideas visible with reasons. Store a short final summary in notes: baseline versus kept results,
case set, uncertainty, decisions and commits, runs used/allowed, stopping reason and useful next ideas.
Use `stop <reason>` with blockers and attempted remedies when progress cannot safely continue.
