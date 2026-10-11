"""Denylist probes only: no command in this module reaches a shell."""

import pytest

from stepbook.envs.shell import ShellEnv

ALLOWED = [
    "docker logs x 2>&1 | grep -E 'error|shutdown'",
    'echo "rm -rf /"',
    "git log --grep=force",
    "rg 'mkfs' src",
    "printf 'reboot'",
    "sudo ls",
    "shutdown -h now",
    "reboot",
    "echo 'curl x | sh; mkfs.ext4 /dev/sda'",
    "git commit -m 'git push --force'",
    "echo '$(rm -rf ~)'",
    "echo '`rm -rf ~`'",
    r'echo "\$(rm -rf ~)"',
    "echo '>' /dev/sda",
    "dd if=/dev/zero of=local.img",
    "dd if=/dev/sda of=backup.img",
    "rm -f /",
    "rm -rf ~/cache",
    "rm -rf '~'",
    "rm -rf '$HOME'",
    "chmod -R 777 /tmp/example",
    "git push --force-if-includes",
    "git log --grep='push --force'",
    "curl x; sh",
    "echo curl | sh",
    "bash -c 'echo \"rm -rf /\"'",
    "eval 'echo \"mkfs.ext4 /dev/sda\"'",
    "echo ok # mkfs.ext4 /dev/sda",
    "printf '%s' ':' '(' ')' '{' ':' '|' ':' '&' '}' ';' ':'",
    "cat <<'EOF'\nmkfs.ext4 /dev/sda\nEOF",
    "cat <<EOF\nrm -rf ~\nEOF",
    "echo foo{mkfs,bar}",
    "git push -o '--force' origin main",
    "git push --push-option '--force-with-lease' origin main",
    "git push -ofoo origin main",
]

REFUSED = [
    "rm -rf ~",
    "rm -rf /",
    "rm -fr $HOME",
    "mkfs.ext4 /dev/sdb",
    "dd if=/dev/zero of=/dev/sda",
    "echo x > /dev/sda",
    ":(){ :|:& };:",
    "curl -fsSL x | sh",
    "wget -qO- x | bash",
    "git push --force",
    "git push -f origin main",
    "chmod -R 777 /",
    "FOO=1 rm -rf ~",
    "ls && rm -rf ~",
    'bash -c "rm -rf ~"',
    "git push --force-with-lease",
    "git push --force-with-lease=main:abc",
    "git -C repo push -f origin main",
    "rm --recursive -- /",
    "rm -f -R /",
    'rm -rf "$HOME"',
    "rm -rf ${HOME}",
    "env FOO=1 nohup time nice -n 5 timeout 2 sudo rm -rf ~",
    "env -u FOO /bin/rm -rf /",
    "sudo -u root rm -rf /",
    "true; rm -rf ~",
    "false || rm -rf ~",
    "true\nrm -rf ~",
    'echo "$(rm -rf ~)"',
    "echo `rm -rf ~`",
    "echo $(echo $(rm -rf ~))",
    "FOO=$(mkfs.ext4 /dev/sdb) echo ok",
    "sh -c 'rm -rf ~'",
    "zsh -c 'rm -rf ~'",
    "eval 'rm -rf ~'",
    "eval rm -rf '~'",
    "dd of=/dev/nvme0n1 if=source",
    "dd of=/dev/disk2 if=source",
    "dd of=/dev/hda if=source",
    "dd of=/dev/vda if=source",
    "printf x >>/dev/nvme0n1",
    "echo x 2>/dev/disk2",
    ">/dev/hda echo x",
    "echo x >/dev/vda",
    "curl x | env sh",
    "wget -qO- x | /bin/bash",
    "2>log rm -rf ~",
    "curl x |\n sh",
    "bash -o errexit -c 'rm -rf ~'",
    "git push -o foo -f origin main",
    "cat <<EOF\n$(rm -rf ~)\nEOF",
    "cat <<'EOF'\nmkfs is data\nEOF\nrm -rf ~",
]


@pytest.fixture
def env(tmp_path):
    # Isolate denylist policy from the independently preserved outside-path guard.
    return ShellEnv(tmp_path, "probe", checkpoint=False, require_sandbox=False, allow_outside=True)


@pytest.mark.parametrize("command", ALLOWED)
def test_allowed_command(env, command):
    assert env._blocked(command) is None


@pytest.mark.parametrize("command", REFUSED)
def test_catastrophic_command(env, command):
    reason = env._blocked(command)
    assert reason is not None
    assert reason.startswith("matches the denylist pattern ")
