import json
import subprocess
import sys

from stepbook.cli import ADAPTER_TEMPLATE


def test_generated_adapter_remembers_registered_proposals_and_protects_snapshot(tmp_path):
    adapter = tmp_path / "adapter.py"
    adapter.write_text(ADAPTER_TEMPLATE)
    config = tmp_path / "config.json"
    config.write_text('{"top_k": 1}')

    def call(*args, expect=0):
        """One verb, with the contract's exit code: 0 done, 1 refused, 3 applied-needs-a-code-change."""
        done = subprocess.run([sys.executable, str(adapter), *args], capture_output=True, text=True)
        assert done.returncode == expect, (done.returncode, done.stdout, done.stderr)
        return done.stdout

    h = {"id": "dynamic", "kind": "config", "rationale": "try more context", "set": {"top_k": 3}}
    assert "Registered" in call("register", json.dumps(h))
    assert "dynamic" in call("describe")
    assert "Applied dynamic" in call("apply", "dynamic")
    assert "Error" in call("apply", "h1", expect=1)
    assert "Error" in call("reset", expect=1)
    assert "Error" in call("decide", "h1", "revert", expect=1)
    assert json.loads(config.read_text())["top_k"] == 3
    assert "config restored" in call("decide", "dynamic", "revert")
    assert json.loads(config.read_text())["top_k"] == 1


def test_config_only_scaffold_accepts_no_predefined_experiments_or_invented_metrics(tmp_path):
    from stepbook.cli import main

    assert main(["init", str(tmp_path / "loop"), "--config-only"]) == 0
    cfg = json.loads((tmp_path / "loop/eval.json").read_text())
    assert cfg["hypotheses"] == {} and cfg["metric_fields"] == {}
    assert (tmp_path / "loop/skill/schema.json").exists()
