#!/bin/sh
# Points this checkout's git hooks at scripts/git-hooks (the local CI push gate).
set -eu
top="$(git rev-parse --show-toplevel)"
chmod +x "$top"/scripts/git-hooks/*
git -C "$top" config core.hooksPath scripts/git-hooks
echo "git hooks: core.hooksPath = scripts/git-hooks (pre-push runs the local CI gate)"
