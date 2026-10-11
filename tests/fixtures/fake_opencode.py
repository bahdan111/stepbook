"""Offline OpenCode executable: check the subprocess contract and replay JSONL."""

import json
import os
import sys
import time
from pathlib import Path

argv = sys.argv[1:]
assert argv[:3] == ["run", "--format", "json"], argv
options = argv[3:]
assert len(options) % 2 == 0, argv
flags = dict(zip(options[::2], options[1::2], strict=True))
assert set(flags) <= {"-m", "--variant", "--agent"}, argv
assert "--agent" in flags
config = json.loads(os.environ["OPENCODE_CONFIG_CONTENT"])
agent = config["agent"][flags["--agent"]]
assert agent["mode"] == "primary" and agent["description"]
permission = os.environ.get("FAKE_OPENCODE_PERMISSION", "deny")
assert agent["permission"] == dict.fromkeys(("edit", "bash", "webfetch"), permission)
assert config["autoupdate"] is False and config["share"] == "disabled"
prompt = sys.stdin.read()
assert "Skill output schema:\n" in prompt
assert prompt.endswith("Return exactly one JSON object matching this schema, no prose and no code fence.\n")
with Path(os.environ["FAKE_OPENCODE_LOG"]).open("a") as log:
    log.write(json.dumps({"argv": argv, "prompt": prompt, "config": config, "cwd": os.getcwd()}) + "\n")
time.sleep(float(os.environ.get("FAKE_OPENCODE_SLEEP", "0")))
if os.environ.get("FAKE_OPENCODE_SMOKE"):
    observation = prompt.split("<<<OBSERVATION\n", 1)[1].split("\nOBSERVATION>>>", 1)[0]
    action = "done" if "verified: pass" in observation else "check"
    answer = {"reasoning": "offline check", "state_patch": {}, "action": action}
    print(json.dumps({"type": "text", "sessionID": "smoke", "part": {"text": json.dumps(answer)}}))
    print(json.dumps({"type": "step_finish", "sessionID": "smoke", "part": {"tokens": {}, "cost": 0}}))
else:
    print(Path(os.environ["FAKE_OPENCODE_FIXTURE"]).read_text(), end="")
if message := os.environ.get("FAKE_OPENCODE_STDERR"):
    print(message, file=sys.stderr)
sys.exit(int(os.environ.get("FAKE_OPENCODE_EXIT", "0")))
