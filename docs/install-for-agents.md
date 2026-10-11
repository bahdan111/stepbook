# Installing stepbook: instructions for a coding agent

You are a coding agent, and your user asked you to install stepbook. Follow these steps in order. Ask the user
only where a step says so, and show them each command before you run anything that installs software.

## 1. Which agent are you?

- **Claude Code.** Use the plugin. You cannot run slash commands yourself, so ask the user to paste these two lines
  into this chat, then go to step 4:

  ```text
  /plugin marketplace add bahdan111/stepbook
  /plugin install stepbook@stepbook-plugins
  ```

- **Codex.** Use the plugin. Show the user these two commands, run them, and ask the user to start a new Codex
  session and use `$stepbook`; then go to step 4:

  ```sh
  codex plugin marketplace add bahdan111/stepbook
  codex plugin add stepbook@stepbook-plugins
  ```

- **Anything else with a shell** (OpenCode or another agent): go to step 2.

## 2. Install the CLI

It needs macOS or Linux, Git, and uv. If `uv --version` fails, show the user uv's install command from
https://docs.astral.sh/uv/ and ask them to run it. Then:

```sh
uv tool install stepbook
stepbook validate checklist
```

The last command prints a line starting with `ok:`. `stepbook` is now on the PATH for every agent and terminal.

## 3. Keep the instructions where you will find them

The onboarding guide is `skills/stepbook/SKILL.md` in the repository
(https://raw.githubusercontent.com/bahdan111/stepbook/main/skills/stepbook/SKILL.md). It is written for every agent:
its commands call plain `stepbook`, and its section "Running stepbook from your agent" lists the flags for yours
(`--executor codex` for Codex, `--executor opencode` for OpenCode) and how to send notes with `stepbook say "<note>"`.

If your agent loads skills from a folder, download the guide there as `stepbook/SKILL.md` and tell the user where it
went. For example, Codex uses `~/.codex/skills/` and OpenCode uses `~/.config/opencode/skills/`. Otherwise remember
the URL and read it when the user asks for a run.

## 4. Check the login that will drive the loop

The loop calls an agent CLI once per step with the user's existing login. stepbook refuses to start if that CLI would
bill an API key. Check the login the user wants, and never set an API key or change how they pay:

- Claude: `claude auth status` shows a claude.ai login.
- Codex: `codex login status` says ChatGPT.
- OpenCode: `opencode auth list` shows a provider.

If none is logged in, tell the user which login command to run themselves.

## 5. Tell the user how to start

In three or four sentences: they describe a long-horizon task (a measured result to improve, tests to fix, a checklist to
work through), you draft a plan and show it, and after their go the loop runs in the background, where they can check
it, steer it with notes, pause it, or resume it later, also from another agent. Shell tasks run in a container or,
with their explicit OK, in a fresh Git worktree, never in their main working tree. Offer to start with a small first
task.
