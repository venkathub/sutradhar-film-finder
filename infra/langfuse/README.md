# infra/langfuse — self-hosted Langfuse v3 on AIC Cloud (DEC-P3-7)

Self-hosted tracing backend for the whole eval/serving stack. One `essential-8gb` AIC Cloud
VPS (4 vCPU / 8 GB / 80 GB NVMe, dedicated IPv4, ₹799/mo — the cheapest tier *with* a public
IPv4) runs the pinned Langfuse v3 compose stack, published over an **outbound cloudflared
tunnel** (AIC's managed edge firewall opens only SSH — inbound 443/80 are blocked and
unmodifiable, verified live 2026-07-03; see the DEC-P3-7 amendments). Langfuse Cloud free
tier is the documented fallback (DEC-P3-7 option A).

## One command

```bash
make langfuse-up        # idempotent from-scratch bootstrap; safe to re-run any time
```

`provision.py` converges from **any** intermediate state:

| Phase | Steps (each check-then-act) |
|---|---|
| 1 — instance (AIC API) | find `sutradhar-obs-01` → running? skip · stopped? start · absent? wallet pre-check → resolve `essential-8gb` from the live catalogue → **checkout is dashboard-only** (verified live 2026-07-03: the API returns 403 for checkout — the script prints exact one-time purchase instructions: plan `essential-8gb`, name `sutradhar-obs-01`, OS `ubuntu-24.04`, attach your SSH key; re-run then finds the instance and continues) |
| 2 — configure (SSH) | 4 GB swap → Docker → **Docker-in-LXC nesting gate** (Essential VPS is LXC; a hard failure stops with an escalation message — AIC support / KVM-class product, never fought blindly) → clone pinned tag `v3.203.3` → secrets generated **once** (all `# CHANGEME` rotated; headless init with pinned org/project/user + project keys) → `docker compose up -d` → **cloudflared quick tunnel** (systemd; outbound-only public HTTPS — no inbound ports exist on this tier) → ufw (22 + external SSH NAT port + 443/80) → `AUTH_DISABLE_SIGNUP=true` → nightly backup cron → health gate **through the tunnel edge** |

Already-satisfied steps are detected and skipped; **no destructive operation without the
explicit `--recreate` flag**. The script ends by printing the `LANGFUSE_HOST` +
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` lines for the laptop `.env`.

Needs in `.env`: `AICCLOUD_API_KEY` (dashboard → Settings → API Keys). The tunnel URL is
minted per cloudflared start (quick tunnel, no domain needed); upgrade to a stable named
tunnel when a domain exists (planned with the P6 static surface). Committed trace exports
are the durable evidence either way.

## Backups (the compose stack ships none)

`backup_langfuse.sh` (installed by the bootstrap, cron `15 2 * * *`): Postgres `pg_dumpall`
+ ClickHouse `BACKUP` + MinIO data copy → tarball; pushed **off-box** to a private HF dataset
repo when `/root/.backup_env` provides `HF_BACKUP_REPO` + `HF_TOKEN`; 7-archive local
retention.

## Evidence longevity

Benchmark-cited traces are **exported (JSON via `sutradhar.obs.tracing.export_trace`) and
committed** with the run artifact + a screenshot — standing evidence never depends on VPS
uptime. Ops thereafter: AIC API `stats` / `start|stop|restart` / `upgrade` (vertical, to
`essential-16gb` on OOM/disk pressure).

## Tests

`tests/test_langfuse_provision.py` drives the full bootstrap against a **fake AIC API + fake
SSH transcript** (mock-tested like `extract_session`): fresh-state runs every step in order;
re-run is a no-op; partial state resumes at the right step; instance-exists never calls
checkout; destructive paths require `--recreate`; the LXC nesting gate escalates. CI never
spends money or opens a connection.
