#!/usr/bin/env bash
set -euo pipefail

# This script is a wrapper for the canary_runner.py tool.
# It passes all command-line arguments directly to the Python script.

echo "Executing canary analysis..."

# "$@" is a special shell variable that passes all arguments from this script
# directly to the python command.
python ./tools/canary_runner.py "$@"