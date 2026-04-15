#!/bin/sh
set -e
# Railway injects PORT. VITE_* may only exist at build time; nginx needs a runtime upstream (NGINX_BACKEND).
_b="${NGINX_BACKEND:-${VITE_API_BASE_URL:-}}"
export NGINX_BACKEND="${_b%/}"
unset _b
if [ -z "$NGINX_BACKEND" ]; then
  echo "frontend: set NGINX_BACKEND or VITE_API_BASE_URL to your API base, e.g. https://your-api.up.railway.app (no path, no trailing slash)" >&2
  exit 1
fi
exec /docker-entrypoint.sh "$@"
