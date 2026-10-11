import pytest

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.runtime import loop as runtime
from stepbook.runtime.loop import StepResult


class Environment:
    def reset(self):
        return "ready"

    def step(self, action):
        return StepResult("worked")

    def score(self):
        return 0


def skill():
    return Skill(
        "callback-test",
        "do the work",
        {"type": "object", "properties": {"memo": {"type": "string"}}},
        {"memo": ""},
        write_once_paths=("memo",),
    )


@pytest.mark.parametrize("runtime_class, module", [(Runtime, runtime)])
def test_controller_guard_checks_current_inbox(monkeypatch, runtime_class, module):
    seen = []
    check_control = module.check_control

    def record(log, env=None):
        seen.append(log)
        return check_control(log, env)

    monkeypatch.setattr(module, "check_control", record)
    executor = ScriptedExecutor(lambda system, user: Envelope("work", {}, "work", cost_usd=0.25))

    assert runtime_class(skill(), executor, Environment()).run(3).steps == 3
    assert len(seen) >= 3 and all(log is None for log in seen)
