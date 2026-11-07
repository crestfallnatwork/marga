#!/usr/bin/env bash
# Run clang-tidy with the checked-in configuration. Rules: STYLE.md section 3.10.
#
# Severity comes from .clang-tidy's WarningsAsErrors: `bugprone-*` and `performance-*` are errors;
# `modernize-*` and `readability-*` are advisory warnings.
#
# Usage:
#   scripts/lint.sh                lint every tracked C/C++ file
#   scripts/lint.sh --staged       lint the staged C/C++ files
#   scripts/lint.sh FILE...        lint specific files
#
# A full-tree pass runs one clang-tidy process per file, capped by PICTOR_LINT_JOBS (default 4);
# clang-tidy itself has no parallel mode and per-TU startup dominates the wall time.
#
# Requires a configured build (cmake -S . -B build) so that compile_commands.json exists.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

clang_tidy="${PICTOR_CLANG_TIDY:-$(command -v clang-tidy || true)}"
if [[ -z "$clang_tidy" ]]; then
    echo "lint.sh: clang-tidy not found; activate the gpujoin env or set PICTOR_CLANG_TIDY" >&2
    exit 1
fi

build_dir="${PICTOR_BUILD_DIR:-build}"
if [[ ! -f "$build_dir/compile_commands.json" ]]; then
    echo "lint.sh: $build_dir/compile_commands.json not found; configure the build first" >&2
    exit 1
fi

mode=all
files=()
for arg in "$@"; do
    case "$arg" in
        --staged) mode=staged ;;
        *) files+=("$arg") ;;
    esac
done

if [[ ${#files[@]} -eq 0 ]]; then
    if [[ "$mode" == staged ]]; then
        while IFS= read -r f; do files+=("$f"); done < <(
            git diff --cached --name-only --diff-filter=ACMR | grep -E '\.(hpp|cpp|h|cc)$' || true)
    else
        while IFS= read -r f; do files+=("$f"); done < <(git ls-files '*.hpp' '*.cpp' '*.h' '*.cc')
    fi
fi

if [[ ${#files[@]} -eq 0 ]]; then
    exit 0
fi

jobs="${PICTOR_LINT_JOBS:-4}"
if ! [[ "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "lint.sh: PICTOR_LINT_JOBS must be a positive integer" >&2
    exit 2
fi

# clang-tidy has no --jobs; run one process per file (bounded by memory) and replay the logs in
# file order so the diagnostics stay deterministic.
tmp=$(mktemp -d "$build_dir/.lint.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
export PICTOR_LINT_TMP="$tmp" PICTOR_LINT_TIDY="$clang_tidy" PICTOR_LINT_BUILD="$build_dir"

printf '%s\0' "${files[@]}" | xargs -0 -n1 -P "$jobs" bash -c '
    f="$1"
    key=$(printf "%s" "$f" | cksum | cut -d" " -f1)
    "$PICTOR_LINT_TIDY" -p "$PICTOR_LINT_BUILD" --quiet "$f" >"$PICTOR_LINT_TMP/$key.log" 2>&1
    echo $? >"$PICTOR_LINT_TMP/$key.status"
' _

status=0
for f in "${files[@]}"; do
    key=$(printf "%s" "$f" | cksum | cut -d" " -f1)
    if [[ -f "$tmp/$key.log" ]]; then
        cat "$tmp/$key.log"
    fi
    if [[ ! -f "$tmp/$key.status" || "$(cat "$tmp/$key.status")" -ne 0 ]]; then
        status=1
    fi
done
exit "$status"
