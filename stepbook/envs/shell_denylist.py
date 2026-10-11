"""Static shell denylist matching; inspect syntax without executing commands."""

from __future__ import annotations

import re
from dataclasses import dataclass

# These names are part of the refusal observation, not regexes over arbitrary text.
DENY = (
    "recursive-rm-root-or-home",
    "mkfs",
    "disk-device-write",
    "fork-bomb",
    "download-pipe-shell",
    "git-force-push",
    "chmod-recursive-777-root",
)
_DISK = re.compile(r"/dev/(?:sd|nvme|disk|hd|vd)[^/]*$")
_ASSIGNMENT = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*=")
_OPERATORS = re.compile(r"[0-9]*(?:>>|<<-?|>&|<&|>\||[<>])|&&|\|\||\|&|[;|&()\n]|[{}](?=\s|;|$)")


@dataclass
class _ShellToken:
    text: str
    raw: str
    operator: bool = False


class _ShellLexer:
    """Small, non-executing lexer; preserve quoted words and inspect substitutions.

    This deliberately does not implement shell expansion, aliases or control flow.
    Conditional branches are candidate commands; the sandbox remains the boundary.
    """

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.substitutions: list[list[_ShellToken]] = []

    def tokens(self, end: str = "") -> list[_ShellToken]:
        result = []
        heredocs: list[tuple[_ShellToken, bool]] = []
        while self.pos < len(self.source):
            char = self.source[self.pos]
            if end and char == end:
                self.pos += 1
                break
            if char in " \t\r":
                self.pos += 1
            elif char == "#":
                newline = self.source.find("\n", self.pos)
                self.pos = newline if newline >= 0 else len(self.source)
            elif match := _OPERATORS.match(self.source, self.pos):
                self.pos = match.end()
                result.append(_ShellToken(match[0].lstrip("0123456789"), match[0], True))
                if match[0] == "\n":
                    self.heredocs(heredocs)
                    heredocs = []
            else:
                word = self.word(end)
                if result and result[-1].operator and result[-1].text in {"<<", "<<-"}:
                    heredocs.append((word, result[-1].text == "<<-"))
                result.append(word)
        return result

    def heredocs(self, documents: list[tuple[_ShellToken, bool]]) -> None:
        for delimiter, strip_tabs in documents:
            start = self.pos
            while self.pos < len(self.source):
                line_end = self.source.find("\n", self.pos)
                stop = line_end if line_end >= 0 else len(self.source)
                line = self.source[self.pos : stop]
                body_end = self.pos
                self.pos = min(stop + 1, len(self.source))
                if (line.lstrip("\t") if strip_tabs else line) == delimiter.text:
                    break
            else:
                body_end = self.pos
            if delimiter.raw == delimiter.text:
                self.heredoc_expansions(self.source[start:body_end])

    def heredoc_expansions(self, body: str) -> None:
        lexer = _ShellLexer(body)
        while lexer.pos < len(body):
            if body[lexer.pos] == "\\":
                lexer.escape('"')
            elif body[lexer.pos] == "`" or body.startswith("$(", lexer.pos):
                lexer.expansion()
            else:
                lexer.pos += 1
        self.substitutions.extend(lexer.substitutions)

    def expansion(self) -> str:
        """Consume a substitution, keeping its commands separate from its output."""
        backtick = self.source[self.pos] == "`"
        self.pos += 1 if backtick else 2
        self.substitutions.append(self.tokens("`" if backtick else ")"))
        # Unknown output must not accidentally form a known command name.
        return "${command-output}"

    def word(self, end: str) -> _ShellToken:
        start = self.pos
        value = ""
        quote = ""
        while self.pos < len(self.source):
            char = self.source[self.pos]
            if not quote and (char.isspace() or char in ";|&()<>" or (end and char == end)):
                # Braces in ${HOME} are word characters, handled below.
                break
            if char == quote:
                quote = ""
                self.pos += 1
            elif char in "\"'" and not quote:
                quote = char
                self.pos += 1
            elif char == "\\" and quote != "'":
                value += self.escape(quote)
            elif quote != "'" and (char == "`" or self.source.startswith("$(", self.pos)):
                value += self.expansion()
            elif self.source.startswith("${", self.pos):
                close = self.source.find("}", self.pos + 2)
                stop = close + 1 if close >= 0 else len(self.source)
                value += self.source[self.pos : stop]
                self.pos = stop
            else:
                value += char
                self.pos += 1
        return _ShellToken(value, self.source[start : self.pos])

    def escape(self, quote: str) -> str:
        self.pos += 1
        if self.pos == len(self.source):
            return "\\"
        char = self.source[self.pos]
        self.pos += 1
        if char == "\n":
            return ""
        if quote == '"' and char not in '$`"\\':
            return "\\" + char
        return char


def _command_words(tokens: list[_ShellToken]) -> tuple[list[_ShellToken], bool]:
    """Remove redirects (including their operands), independently checking writes."""
    words = []
    disk_write = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.operator and token.text in {">", ">>", ">|", "<", ">&", "<&", "<<", "<<-"}:
            if i + 1 < len(tokens):
                target = tokens[i + 1].text
                disk_write |= token.text in {">", ">>", ">|"} and bool(_DISK.fullmatch(target))
            i += 2
        else:
            words.append(token)
            i += 1
    return words, disk_write


def _skip_options(words: list[_ShellToken], takes_value: set[str]) -> list[_ShellToken]:
    i = 0
    while i < len(words) and words[i].text.startswith("-"):
        option = words[i].text
        i += 1
        if option == "--":
            break
        if option in takes_value:
            i += 1
    return words[i:]


def _unwrap(words: list[_ShellToken]) -> list[_ShellToken]:
    wrappers = {
        "env": {"-u", "--unset", "-C", "--chdir"},
        "nohup": set(),
        "time": {"-o", "--output", "-f", "--format"},
        "nice": {"-n", "--adjustment"},
        "timeout": {"-s", "--signal", "-k", "--kill-after"},
        "sudo": {"-u", "--user", "-g", "--group", "-h", "--host", "-p", "--prompt", "-C", "-T"},
    }
    while words:
        if _ASSIGNMENT.match(words[0].raw):
            words = words[1:]
            continue
        name = words[0].text.rsplit("/", 1)[-1]
        if name not in wrappers:
            break
        words = _skip_options(words[1:], wrappers[name])
        if name == "timeout":
            words = words[1:]
    return words


def _options_and_operands(words: list[_ShellToken]) -> tuple[list[str], list[_ShellToken]]:
    options, operands = [], []
    ended = False
    for word in words:
        if word.text == "--" and not ended:
            ended = True
        elif word.text.startswith("-") and not ended:
            options.append(word.text)
        else:
            operands.append(word)
    return options, operands


def _recursive(options: list[str], letters: str) -> bool:
    return "--recursive" in options or any(
        not option.startswith("--") and any(letter in option[1:] for letter in letters) for option in options
    )


def _simple_rule(words: list[_ShellToken]) -> str | None:
    if not words:
        return None
    name = words[0].text.rsplit("/", 1)[-1]
    args = words[1:]
    options, operands = _options_and_operands(args)
    if name == "rm" and _recursive(options, "rR"):
        homes = {"~", "$HOME", "${HOME}", '"$HOME"', '"${HOME}"'}
        if any(word.text == "/" or word.raw in homes for word in operands):
            return DENY[0]
    if name.startswith("mkfs"):
        return DENY[1]
    if name == "dd" and any(word.text.startswith("of=") and _DISK.fullmatch(word.text[3:]) for word in args):
        return DENY[2]
    if name == "git" and _force_push(args):
        return DENY[5]
    if name == "chmod" and _recursive(options, "R"):
        if operands and operands[0].text == "777" and any(word.text == "/" for word in operands[1:]):
            return DENY[6]
    return None


def _force_push(args: list[_ShellToken]) -> bool:
    args = _skip_options(args, {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env"})
    if not args or args[0].text != "push":
        return False
    options = iter(args[1:])
    for word in options:
        option = word.text
        if option == "--":
            break
        if option in {"-o", "--push-option", "--repo", "--receive-pack", "--exec"}:
            next(options, None)
        elif option in {"--force", "--force-with-lease"} or option.startswith("--force-with-lease="):
            return True
        elif option.startswith("-") and not option.startswith("--") and "f" in option[1:].split("o", 1)[0]:
            return True
    return False


def _nested_command(words: list[_ShellToken]) -> str | None:
    if not words:
        return None
    name = words[0].text.rsplit("/", 1)[-1]
    if name == "eval":
        args = words[1:]
        if args and args[0].text == "--":
            args = args[1:]
        return " ".join(word.text for word in args)
    if name in {"bash", "sh", "zsh"}:
        return _shell_source(words[1:])
    return None


def _shell_source(args: list[_ShellToken]) -> str | None:
    options = iter(args)
    for word in options:
        option = word.text
        if option in {"-o", "+o", "-O", "+O", "--rcfile", "--init-file"}:
            next(options, None)
        elif not option.startswith("-") or option == "--":
            break
        elif not option.startswith("--") and "c" in option[1:]:
            source = next(options, None)
            return source.text if source is not None else None
    return None


def _token_rule(tokens: list[_ShellToken], nested: bool) -> str | None:
    # Match the classic definition AND invocation as syntax, never as a quoted string.
    signature = [":", "(", ")", "{", ":", "|", ":", "&", "}", ";", ":"]
    texts = [token.text if token.operator or token.text == ":" else "" for token in tokens]
    if any(texts[i : i + len(signature)] == signature for i in range(len(texts))):
        return DENY[3]
    command: list[_ShellToken] = []
    download = False
    for token in [*tokens, _ShellToken(";", ";", True)]:
        if not token.operator or token.text not in {";", "&&", "||", "|", "|&", "\n", "&", "(", ")", "{", "}"}:
            command.append(token)
            continue
        if not command and token.text == "\n":
            continue  # A newline after a pipe continues that pipeline.
        words, disk_write = _command_words(command)
        words = _unwrap(words)
        name = words[0].text.rsplit("/", 1)[-1] if words else ""
        if disk_write:
            return DENY[2]
        if download and name in {"sh", "bash"}:
            return DENY[4]
        if rule := _simple_rule(words):
            return rule
        if nested and (source := _nested_command(words)):
            if rule := _deny_rule(source, nested=False):
                return rule
        download = token.text in {"|", "|&"} and name in {"curl", "wget"}
        command = []
    return None


def _deny_rule(source: str, *, nested: bool = True) -> str | None:
    lexer = _ShellLexer(source)
    tokens = lexer.tokens()
    for part in [tokens, *lexer.substitutions]:
        if rule := _token_rule(part, nested):
            return rule
    return None


def denylist_reason(cmd: str) -> str | None:
    """Return the environment's refusal reason, or None when no deny rule matches."""
    if rule := _deny_rule(cmd):
        return f"matches the denylist pattern {rule!r}"
    return None
