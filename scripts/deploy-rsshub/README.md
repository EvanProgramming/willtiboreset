# Self-Hosted RSSHub for WillTiboReset

This directory contains the Docker Compose configuration for a self-hosted RSSHub instance used to fetch Tibo's X (Twitter) posts.

## Prerequisites

- A server with Docker and Docker Compose installed.
- A Twitter/X account (a spare/throwaway account is recommended) for cookie-based authentication.

## Deployment Steps

1. Copy this directory to your server:

   ```bash
   scp -r scripts/deploy-rsshub root@YOUR_SERVER_IP:/opt/
   ```

2. SSH into your server and run the deploy script:

   ```bash
   ssh root@YOUR_SERVER_IP
   cd /opt/deploy-rsshub
   ./deploy.sh
   ```

3. Verify RSSHub is running:

   ```bash
   curl http://localhost:1200
   ```

## Configure Twitter/X Cookie

RSSHub requires a valid Twitter/X login cookie to fetch user timelines.

1. Log in to Twitter/X in your browser (use the spare account).
2. Open DevTools (F12) → Application → Cookies → `https://x.com`.
3. Find the `auth_token` and `ct0` cookies.
4. Edit `/opt/deploy-rsshub/docker-compose.yml` and fill in:

   ```yaml
   TWITTER_AUTH_TOKEN: "your_auth_token"
   TWITTER_COOKIE: "auth_token=your_auth_token; ct0=your_ct0"
   ```

5. Restart RSSHub:

   ```bash
   docker compose up -d
   ```

## Update GitHub Secret

After RSSHub is running, update the `TIBO_RSS_URLS` secret in your WillTiboReset GitHub repository:

```text
http://YOUR_SERVER_IP:1200/twitter/user/thsottiaux
```

If your server has a domain with HTTPS, use:

```text
https://rsshub.yourdomain.com/twitter/user/thsottiaux
```

## Test the Feed

```bash
curl "http://YOUR_SERVER_IP:1200/twitter/user/thsottiaux"
```

You should see an RSS/XML response containing Tibo's recent posts.
