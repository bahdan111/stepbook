"""A deterministic command executor: read stdin, emit one envelope, never call a model."""

import json
import re
import shlex
import sys


def envelope(prompt: str) -> dict:
    """Use the checklist's persisted state to select the next fixed action."""
    state_text = prompt.split("Skill Execution State:\n```json\n", 1)[1]
    state = json.loads(state_text.split("\n```", 1)[0])
    stage = state.get("facts", {}).get("offline_stage", "inspect")
    patch = {"facts": {"offline_stage": "check_before"}}
    if stage in {"fix", "done"}:
        observation = prompt.split("<<<OBSERVATION\n", 1)[1].split("\nOBSERVATION>>>", 1)[0]
        checks = dict(re.findall(r"^  (\S+): (pass|fail)   \(", observation, re.MULTILINE))
        if "tests" not in checks:
            raise ValueError("Expected an observed result for the tests check")
        patch["checks_status"] = checks
    if stage == "inspect":
        action = "cat toy_math/__init__.py tests/test_add.py"
        reasoning = "Inspect the function and its unchanged test."
        patch["current"] = "inspect the implementation and test"
    elif stage == "check_before":
        action = "check"
        reasoning = "Run the operator's test before changing the implementation."
        patch["current"] = "run tests before the fix"
        patch["facts"]["offline_stage"] = "fix"
    elif stage == "fix":
        code = (
            "from pathlib import Path; "
            "p = Path('toy_math/__init__.py'); "
            "s = p.read_text(); "
            "assert 'return left - right' in s; "
            "p.write_text(s.replace('return left - right', 'return left + right'))"
        )
        action = "python3 -c " + shlex.quote(code)
        reasoning = "The fixed script replaces subtraction with addition in the implementation."
        patch["current"] = "fix the implementation after checking tests"
        patch["facts"]["offline_stage"] = "check_after"
    elif stage == "check_after":
        action = "check"
        reasoning = "Verify the implementation using the operator's unchanged check."
        patch["current"] = "verify the implementation fix"
        patch["facts"]["offline_stage"] = "done"
    elif stage == "done":
        passed = all(result == "pass" for result in patch["checks_status"].values())
        action = "done" if passed else "check"
        reasoning = "Finish after passing checks." if passed else "Checks failed; rerun before claiming completion."
        patch["current"] = "finished; all checks passed" if passed else "checks failed; rerun tests"
        patch["facts"]["offline_stage"] = "done"
    else:
        raise ValueError(f"Unknown offline script stage: {stage!r}")
    return {"reasoning": reasoning, "state_patch": patch, "action": action}


if __name__ == "__main__":
    print(json.dumps(envelope(sys.stdin.read())))
