#!/usr/bin/env bash
# Format C++ sources with clang-format. Rules: STYLE.md section 3.1.
#
# Usage:
#   scripts/format.sh              format every tracked C/C++ file
#   scripts/format.sh --check      report files that need formatting (exit 1)
#   scripts/format.sh --staged     format only the staged hunks (git-clang-format)
#   scripts/format.sh FILE...      format specific files
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

clang_format="${PICTOR_CLANG_FORMAT:-$(command -v clang-format || true)}"
if [[ -z "$clang_format" ]]; then
    echo "format.sh: clang-format not found; activate the gpujoin env or set PICTOR_CLANG_FORMAT" >&2
    exit 1
fi

tracked=()
while IFS= read -r f; do tracked+=("$f"); done < <(git ls-files '*.hpp' '*.cpp' '*.h' '*.cc' '*.cu')

mode=all
files=()
for arg in "$@"; do
    case "$arg" in
        --staged) mode=staged ;;
        --check) mode=check ;;
        *) files+=("$arg") ;;
    esac
done

if [[ ${#files[@]} -gt 0 ]]; then
    "$clang_format" -i "${files[@]}"
    exit 0
fi

case "$mode" in
    all)
        if [[ ${#tracked[@]} -gt 0 ]]; then
            "$clang_format" -i "${tracked[@]}"
        fi
        ;;
    check)
        status=0
        for f in "${tracked[@]}"; do
            if ! "$clang_format" --dry-run -Werror "$f" >/dev/null 2>&1; then
                echo "needs formatting: $f"
                status=1
            fi
        done
        exit "$status"
        ;;
    staged)
        # git-clang-format returns 1 on its normal completion path (even when it changed
        # nothing); only propagate real failures.
        status=0
        git clang-format --binary "$clang_format" --staged --force || status=$?
        if [[ "$status" -gt 1 ]]; then
            exit "$status"
        fi
        ;;
esac
