#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# read environments
set -a
source "$ROOT/.env"
set +a

# init fnm
export PATH="$HOME/.local/share/fnm:$PATH"
eval "$(fnm env --shell bash)"

# run zenn cli
exec npx --yes zenn-cli@latest "$@"
