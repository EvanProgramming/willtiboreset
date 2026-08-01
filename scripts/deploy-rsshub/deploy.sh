#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Deploying RSSHub ==="
cd "$SCRIPT_DIR"

if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo "ERROR: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

docker compose pull
docker compose up -d

echo ""
echo "=== RSSHub deployed ==="
echo "Check status: docker compose -f $SCRIPT_DIR/docker-compose.yml logs -f"
echo "Test URL: http://YOUR_SERVER_IP:1200/twitter/user/thsottiaux"
