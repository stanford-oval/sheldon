#!/usr/bin/env bash
# Fetch ARC-AGI-1 and ARC-AGI-2 task files into data/.
# Idempotent: re-running refreshes the clones and re-copies.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${SHELDON_DATA:-$ROOT/data}"
TMP="$DATA/.src"
mkdir -p "$TMP"

fetch() {  # name url
  local name="$1" url="$2"
  if [ -d "$TMP/$name/.git" ]; then
    echo "== updating $name"
    git -C "$TMP/$name" pull --ff-only --quiet
  else
    echo "== cloning $name"
    git clone --depth 1 --quiet "$url" "$TMP/$name"
  fi
}

install_split() {  # name src_dir
  local name="$1" src="$2"
  for split in training evaluation; do
    if [ -d "$src/$split" ]; then
      mkdir -p "$DATA/$name/$split"
      cp "$src/$split"/*.json "$DATA/$name/$split/"
      echo "   $name/$split: $(ls "$DATA/$name/$split" | wc -l) tasks"
    else
      echo "   $name/$split: NOT FOUND in $src" >&2
    fi
  done
}

fetch arc-agi-1 https://github.com/fchollet/ARC-AGI.git
install_split arc-agi-1 "$TMP/arc-agi-1/data"

fetch arc-agi-2 https://github.com/arcprize/ARC-AGI-2.git
install_split arc-agi-2 "$TMP/arc-agi-2/data"

echo
echo "Counts above are what was on disk, not what any paper claims."
echo "Verify against the current ARC Prize release before quoting a number."
echo "Sources kept in $TMP — safe to delete."
