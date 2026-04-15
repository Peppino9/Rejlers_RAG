#!/bin/sh
set -e
# Newer nginx Docker images pass the whole container env to envsubst; that can break "$$uri" escapes.
# Restrict substitution to the two vars we actually substitute in templates.
export NGINX_ENVSUBST_FILTER='^(PORT|NGINX_BACKEND)$'
# Railway injects PORT. VITE_* may only exist at build time; nginx needs a runtime upstream (NGINX_BACKEND).
_b="${NGINX_BACKEND:-${VITE_API_BASE_URL:-}}"
export NGINX_BACKEND="${_b%/}"
unset _b
if [ -z "$NGINX_BACKEND" ]; then
  echo "frontend: set NGINX_BACKEND or VITE_API_BASE_URL to your API base, e.g. https://your-api.up.railway.app (no path, no trailing slash)" >&2
  exit 1
fi
exec /docker-entrypoint.sh "$@"
