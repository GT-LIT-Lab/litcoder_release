#!/usr/bin/env bash
set -u
set -o pipefail

ZIP_FILE="$(cd "$(dirname "$0")" && pwd)/analysis_litcoder.zip"
EXTRACT_DIR="$(cd "$(dirname "$0")" && pwd)/analysis_litcoder_figures"

if [ ! -f "$ZIP_FILE" ]; then
  echo "Zip file not found: $ZIP_FILE" >&2
  exit 1
fi

rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"

echo "Unzipping $ZIP_FILE -> $EXTRACT_DIR"
unzip -q -o "$ZIP_FILE" -d "$EXTRACT_DIR" | cat

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python is not installed (python3/python not found in PATH)." >&2
  exit 1
fi

# Iterate over .py files in a portable way (no mapfile/arrays)
count=0
fail_count=0
failed_scripts=""

while IFS= read -r -d '' script; do
  count=$((count+1))
  echo "=== Running: $script ==="
  script_dir="$(dirname "$script")"
  script_base="$(basename "$script")"
  (
    cd "$script_dir" >/dev/null 2>&1 &&
    "$PYTHON_BIN" "$script_base"
  )
  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo "FAILED ($exit_code): $script"
    failed_scripts="${failed_scripts}${script} (${exit_code})"$'\n'
    fail_count=$((fail_count+1))
  else
    echo "OK: $script"
  fi

done < <(find "$EXTRACT_DIR" -type f -name "*.py" ! -path "*/__MACOSX/*" ! -name "._*" -print0)

if [ $count -eq 0 ]; then
  echo "No .py files found in $EXTRACT_DIR"
  exit 0
fi

echo
echo "Ran ${count} file(s). Failures: $fail_count"
if [ $fail_count -gt 0 ]; then
  printf '%s' "$failed_scripts"
  exit 1
fi 