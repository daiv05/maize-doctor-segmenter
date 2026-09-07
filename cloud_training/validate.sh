#!/usr/bin/env bash
set -euo pipefail

# validate y test son la misma evaluación sobre el mismo split retenido: se delega en
# evaluate_test.sh para que exista una sola implementación y un solo guard.
exec bash "$(dirname "$0")/evaluate_test.sh" "$@"
