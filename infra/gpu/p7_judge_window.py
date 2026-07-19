"""P7 task-20 judge leg (DEC-P7-7): one ephemeral judge+sidecar session that rejudges
the fresh capture run (coherence + RAGAS) and destroys itself.

Reuses the PROVEN serving-benchmark Session-B topology verbatim
(``build_judge_sidecar_startup_script`` + smoke-disambiguated endpoints with a
mandatory BGE-M3 sidecar — never the broken ``--task embed`` path), with the runner
swapped for ``evals/rejudge_run.py --run <id> --with-ragas``. Teardown guaranteed in
``finally``; artifacts are laptop-side (the run JSON mutates in place, judge block
recorded), so a dead box loses nothing.

Usage:  uv run python infra/gpu/p7_judge_window.py <generation_run_id>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jarvis import (  # noqa: E402
    DEFAULT_HEALTH_TIMEOUT_S,
    SIDECAR_SCRIPT,
    _default_deps,
    _resolve_serve_endpoints,
    build_judge_sidecar_startup_script,
)

from sutradhar.config import get_settings  # noqa: E402


def main(run_id: str) -> int:
    settings = get_settings()
    deps = _default_deps()
    judge_model = settings.require("judge_model")
    client = deps.create_client(settings)
    instance = None
    exit_code = 1
    try:
        deps.log(f"[p7-judge] creating {settings.gpu_type} to serve {judge_model} + sidecar …")
        script = build_judge_sidecar_startup_script(
            settings.require("hf_token"),
            judge_model,
            settings.embed_model,
            settings.rerank_model,
            SIDECAR_SCRIPT.read_text(encoding="utf-8"),
        )
        instance = deps.create_instance(client, settings, script)
        judge_settings = settings.model_copy(update={"llm_model": judge_model})
        judge_url, embed_url, _ = _resolve_serve_endpoints(
            instance,
            deps,
            judge_settings,
            timeout_s=DEFAULT_HEALTH_TIMEOUT_S,
            poll_interval_s=15.0,
            client=client,
            require_sidecar=True,  # RAGAS must embed against BGE-M3, never gpt-oss
        )
        if judge_url is None or embed_url is None:
            raise RuntimeError("judge or its BGE-M3 sidecar never became healthy")
        deps.log(f"  judge window UP ({judge_url}); rejudging {run_id} …")
        proc = subprocess.run(
            [
                sys.executable,
                "evals/rejudge_run.py",
                "--run",
                run_id,
                "--with-ragas",
            ],
            env={
                **__import__("os").environ,
                "JUDGE_BASE_URL": judge_url,
                "JUDGE_MODEL": judge_model,
                "EMBED_BASE_URL": embed_url,
            },
        )
        exit_code = proc.returncode
        deps.log(f"  rejudge exit={exit_code}")
    finally:
        machine_id = getattr(instance, "machine_id", None)
        if machine_id is not None:
            try:
                deps.log(f"  destroying instance {machine_id} …")
                deps.destroy_instance(client, machine_id)
            except Exception as exc:  # noqa: BLE001
                deps.log(f"  TEARDOWN FAILED: {exc} — run `make gpu-nuke`")
        try:
            deps.remove_scripts(client)
        except Exception as exc:  # noqa: BLE001
            deps.log(f"  script cleanup failed (non-fatal): {exc}")
    return exit_code


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
