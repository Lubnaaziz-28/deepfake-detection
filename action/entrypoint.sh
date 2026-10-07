#!/bin/sh
set -e

# Append conditional inputs as CLI flags.
if [ "${FAIL_ON_DEEPFAKE:-true}" = "true" ]; then
  set -- "$@" --fail-on-deepfake
fi
if [ -n "${MODEL:-}" ]; then
  set -- "$@" --model "$MODEL"
fi

# Run the scan; a non-zero exit means deepfakes were flagged.
set +e
deepfake-scan "$@"
status=$?
set -e

# Publish step outputs for docker actions via GITHUB_OUTPUT.
report="${REPORT:-deepfake-report.json}"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  flagged=0
  if [ -f "$report" ]; then
    flagged=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['summary']['deepfakes'])" "$report" 2>/dev/null || echo 0)
  fi
  echo "flagged=${flagged}" >> "$GITHUB_OUTPUT"
  echo "report-path=${report}" >> "$GITHUB_OUTPUT"
fi

exit $status
