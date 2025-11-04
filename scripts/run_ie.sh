#!/usr/bin/env bash
set -euo pipefail

# Make our override take precedence over installed RecSim
export PYTHONPATH="$(pwd)/overrides:${PYTHONPATH:-}"

# Run the entrypoint
python3 code.py
