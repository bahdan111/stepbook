"""stepbook runtime.

Implements the runtime from "SKILL.state: Scalable Long-Horizon Agent Skills"
(Badhe, Tiwari, Chung; arXiv 2608.26263, CC BY 4.0): at every step the model sees only
the immutable skill instructions, the current structured state and the latest observation,
emits a validated state patch plus one action, and its reasoning is discarded.
"""

from .contracts.runtime import Environment, RunResult, StepResult
from .executors.base import Envelope, StepExecutor
from .executors.claude import ClaudeCLIExecutor
from .executors.command import CommandExecutor, ScriptedExecutor
from .runtime.loop import Runtime
from .state.merge import count_deletions, diff, merge
from .state.skill import Skill
from .state.validator import PatchError, Validator

__all__ = [
    "merge",
    "diff",
    "count_deletions",
    "Validator",
    "PatchError",
    "Envelope",
    "StepExecutor",
    "ClaudeCLIExecutor",
    "CommandExecutor",
    "ScriptedExecutor",
    "Skill",
    "Runtime",
    "RunResult",
    "Environment",
    "StepResult",
]
