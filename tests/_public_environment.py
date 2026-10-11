"""Deterministic environment for public runtime accounting tests."""

from stepbook import StepResult


class FixtureEnv:
    def __init__(self, horizon=100, seed=0, **kwargs):
        self.horizon = horizon
        self.steps = 0

    def reset(self):
        self.steps = 0
        return "Inspect the fixture."

    def step(self, action):
        self.steps += 1
        return StepResult("Fixture action completed.", done=self.steps >= self.horizon)

    def score(self):
        return 0.0
