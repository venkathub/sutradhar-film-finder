#!/bin/bash
# Nightly off-box Langfuse backup (P3, DEC-P3-7 — the compose stack ships NO backups).
# Postgres logical dump + ClickHouse BACKUP + MinIO event-media sync, tarred to /root/backups
# and (when HF_BACKUP_REPO + HF_TOKEN are configured in /root/.backup_env) pushed off-box to a
# private Hugging Face dataset repo. Retention: last 7 local archives.
set -euo pipefail

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORKDIR="/root/backups/${STAMP}"
LANGFUSE_DIR="/root/langfuse"
mkdir -p "${WORKDIR}"
cd "${LANGFUSE_DIR}"

# shellcheck disable=SC1091
[ -f /root/.backup_env ] && . /root/.backup_env

# 1. Postgres logical dump (users, orgs, projects, API keys, prompts …)
docker compose exec -T postgres pg_dumpall -U postgres | gzip > "${WORKDIR}/postgres.sql.gz"

# 2. ClickHouse: traces/observations/scores tables
docker compose exec -T clickhouse clickhouse-client \
  --query "BACKUP DATABASE default TO Disk('backups', '${STAMP}.zip')" || \
  echo "clickhouse BACKUP failed (check backups disk config)" >&2
docker compose cp clickhouse:/backups/"${STAMP}".zip "${WORKDIR}/clickhouse.zip" || true

# 3. MinIO: event/media object store
docker compose cp minio:/data "${WORKDIR}/minio-data" || true

tar -C "${WORKDIR}" -czf "/root/backups/langfuse-${STAMP}.tar.gz" .
rm -rf "${WORKDIR}"

# 4. Off-box push (private HF dataset repo; skipped when not configured)
if [ -n "${HF_BACKUP_REPO:-}" ] && [ -n "${HF_TOKEN:-}" ]; then
  pip show huggingface_hub >/dev/null 2>&1 || pip install -q huggingface_hub
  python3 - "$STAMP" << 'PYEOF'
import os, sys
from huggingface_hub import HfApi
stamp = sys.argv[1]
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj=f"/root/backups/langfuse-{stamp}.tar.gz",
    path_in_repo=f"langfuse-backups/langfuse-{stamp}.tar.gz",
    repo_id=os.environ["HF_BACKUP_REPO"],
    repo_type="dataset",
)
PYEOF
fi

# 5. Retention: keep the last 7 local archives
ls -1t /root/backups/langfuse-*.tar.gz | tail -n +8 | xargs -r rm -f
echo "backup ${STAMP} complete"
