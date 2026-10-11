#!/usr/bin/env bash
# One version. Rewrites it in pyproject.toml, in both plugin manifests (the agents read it there) and in every
# `uvx stepbook@X.Y.Z` pin in the onboarding skill. Nothing else carries a version.
# usage: scripts/bump-version.sh X.Y.Z
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION="${1:-}"
if [[ ! $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$ ]]; then
  echo "usage: scripts/bump-version.sh X.Y.Z" >&2
  exit 2
fi
MANIFESTS=(.claude-plugin/plugin.json .codex-plugin/plugin.json)
SKILL=skills/stepbook/SKILL.md

# Rewrite in place without depending on GNU or BSD `sed -i`.
edit() {
  local file=$1 expression=$2
  sed -E "$expression" "$file" >"$file.bump-version"
  mv "$file.bump-version" "$file"
}

edit pyproject.toml "s/^version = \".*\"\$/version = \"$VERSION\"/"
for manifest in "${MANIFESTS[@]}"; do
  edit "$manifest" "s/(\"version\": )\"[^\"]*\"/\\1\"$VERSION\"/"
done
written=("pyproject.toml" "${MANIFESTS[@]}")
if [[ -f $SKILL ]]; then
  edit "$SKILL" "s/stepbook@[0-9A-Za-z.+-]+/stepbook@$VERSION/g"
  written+=("$SKILL")
fi

# Report success only when every place carries the new version.
grep -qx "version = \"$VERSION\"" pyproject.toml
for manifest in "${MANIFESTS[@]}"; do
  grep -q "\"version\": \"$VERSION\"" "$manifest"
done
echo "version $VERSION: ${written[*]}"
